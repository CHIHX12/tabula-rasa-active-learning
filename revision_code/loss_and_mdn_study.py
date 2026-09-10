"""
Controlled study of the PC-BAN training objective and the MDN head.

Answers, with numbers rather than assertion:
  * surrogate study  -- how were lambda_MSE = 0.10 and lambda_rec = 0.02 chosen?
                     -> full one-factor-at-a-time sensitivity sweep.
  * MDN/loss study -- is the MSE term necessary, given the NLL already
                     contains it in some form?  -> lambda_MSE = 0 arm.
  * MDN/loss study -- why a Gaussian mixture if only the mean and variance are
                     used?  -> K = 1 (single Gaussian) vs K > 1, scored on
                     accuracy AND on the quality of the uncertainty that the
                     acquisition function actually consumes.

Protocol: the labelled set is a random draw of n = 220 compositions (the same
budget the active-learning campaigns end with); the model is scored on the
5854 never-seen compositions, i.e. the same out-of-pool metric used everywhere
else in the revision.  5 independent seeds.
"""
from __future__ import annotations

import itertools
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import norm
from sklearn.metrics import r2_score

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.process_model import PCBAN  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(exist_ok=True)

N_LAB = 220
N_BOOT = 5
N_MC = 20
EPOCHS = 300


def load():
    df = pd.read_csv(ROOT / "data" / "OER_database.csv")
    df.columns = [c.strip().lstrip("﻿") for c in df.columns]
    return (df[["Ni", "Fe", "Co", "Ce"]].values.astype(np.float32),
            df["J10"].values.astype(np.float32))


def fit_ensemble(Xtr, ytr, lam_mse, lam_rec, K, seed):
    """Bootstrap ensemble at the final architecture tier."""
    mean_, scale_ = float(ytr.mean()), float(ytr.std()) + 1e-8
    y_sc = (ytr - mean_) / scale_
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    models = []
    t0 = time.time()
    for _ in range(N_BOOT):
        b = rng.integers(0, len(Xtr), size=len(Xtr))
        m = PCBAN(n_elem=4, embed_dim=32, triplet_rank=4, proj_dim=30,
                  n_components=K, hidden_dims=[128, 64], dropout=0.25).to(DEVICE)
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-3)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=1e-5)
        Xt = torch.tensor(Xtr[b], dtype=torch.float32, device=DEVICE)
        yt = torch.tensor(y_sc[b], dtype=torch.float32, device=DEVICE)
        m.train()
        for _ in range(EPOCHS):
            loss, *_ = m.full_loss(Xt, yt, lam_mse=lam_mse, lam_rec=lam_rec)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step(); sch.step()
        m.eval()
        models.append(m)
    return models, mean_, scale_, time.time() - t0


def evaluate(models, mean_, scale_, Xte, yte):
    Xt = torch.tensor(Xte, dtype=torch.float32, device=DEVICE)
    preds = []
    with torch.no_grad():
        for m in models:
            m.train()                      # MC dropout, as in the AL loop
            for _ in range(N_MC):
                preds.append(m.predict_mean(Xt).cpu().numpy() * scale_ + mean_)
            m.eval()
    P = np.asarray(preds)
    mu, sd = P.mean(0), P.std(0)
    err = yte - mu
    # Gaussian NLL of the held-out data under the epistemic predictive law
    sd_c = np.maximum(sd, 1e-6)
    nll = float(np.mean(0.5 * np.log(2 * np.pi * sd_c**2) + 0.5 * (err / sd_c) ** 2))
    cover95 = float(np.mean(np.abs(err) <= 1.959964 * sd_c))
    # does sigma rank the errors?  (Spearman between |err| and sigma)
    from scipy.stats import spearmanr
    rho = float(spearmanr(np.abs(err), sd_c).correlation)
    return dict(r2=float(r2_score(yte, mu)),
                rmse=float(np.sqrt(np.mean(err**2))),
                mae=float(np.mean(np.abs(err))),
                nll=nll, coverage95=cover95, sigma_error_spearman=rho,
                mean_sigma=float(sd.mean()))


def main():
    X, y = load()
    N = len(X)
    seeds = [0, 1, 2, 3, 4]

    grid = []
    for lm in [0.0, 0.01, 0.05, 0.10, 0.25, 0.50, 1.00]:
        grid.append(dict(kind="lam_mse", lam_mse=lm, lam_rec=0.02, K=10))
    for lr_ in [0.0, 0.005, 0.01, 0.02, 0.05, 0.10, 0.50]:
        grid.append(dict(kind="lam_rec", lam_mse=0.10, lam_rec=lr_, K=10))
    for K in [1, 2, 4, 6, 8, 10, 16]:
        grid.append(dict(kind="K", lam_mse=0.10, lam_rec=0.02, K=K))

    # de-duplicate the shared default point
    seen, uniq = set(), []
    for g in grid:
        key = (g["lam_mse"], g["lam_rec"], g["K"])
        if key in seen:
            g = dict(g, dup_of=key)
        seen.add(key)
        uniq.append(g)

    results = []
    for gi, g in enumerate(uniq):
        rows = []
        for s in seeds:
            rng = np.random.default_rng(1000 + s)
            lab = rng.choice(N, size=N_LAB, replace=False)
            mask = np.ones(N, bool); mask[lab] = False
            models, mean_, scale_, secs = fit_ensemble(
                X[lab], y[lab], g["lam_mse"], g["lam_rec"], g["K"], seed=s)
            m = evaluate(models, mean_, scale_, X[mask], y[mask])
            m["train_seconds"] = secs
            rows.append(m)
        agg = {k: (float(np.mean([r[k] for r in rows])),
                   float(np.std([r[k] for r in rows]))) for k in rows[0]}
        rec = dict(g, per_seed=rows, agg=agg)
        results.append(rec)
        print(f"[{gi+1:2d}/{len(uniq)}] kind={g['kind']:8s} "
              f"lam_mse={g['lam_mse']:.3f} lam_rec={g['lam_rec']:.3f} K={g['K']:2d} | "
              f"R2={agg['r2'][0]:.4f}+-{agg['r2'][1]:.4f} "
              f"RMSE={agg['rmse'][0]:.2f} NLL={agg['nll'][0]:.3f} "
              f"cov95={agg['coverage95'][0]:.3f} rho={agg['sigma_error_spearman'][0]:.3f}",
              flush=True)
        json.dump(results, open(OUT / "loss_mdn_study.json", "w"), indent=1)

    print("DONE")


if __name__ == "__main__":
    main()
