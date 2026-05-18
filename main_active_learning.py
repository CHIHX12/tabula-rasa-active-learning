"""
Active Learning — Prospective Dynamic Landscape Learning (v6)
=============================================================
Key improvements over v5:
  (1) MC Dropout + Bootstrap Ensemble -> more reliable uncertainty sigma
  (2) Three-point batch query (Stratified Batch Query) -> full landscape coverage
        Point A: LCBDS  -> find lowest J10 (exploitation)
        Point B: Max sigma -> fill most uncertain region (exploration)
        Point C: Max mu   -> understand highest J10 region (contrastive exploration)
  (3) beta annealing + diversity penalty + surprise curiosity (inherited from v5)
  (4) Bootstrap OOB evaluation (implicit validation, no separate test set needed)

Core design principle (prospective AL):
  Day 1: only 20 labeled experiments, no prior knowledge of the landscape
  Each round: one new experiment -> add to training set -> update model
  Evaluation: Bootstrap OOB (implicit validation) without a pre-split test set
  Termination: after sufficient labels, perform final held-out evaluation

  -> This reflects the real-world scenario of learning from scratch

Key differences from retrospective AL:
  - No pre-split test set (which would require knowing all 6074 points upfront)
  + Bootstrap OOB automatically provides implicit validation each round
  + Final evaluation uses remaining unlabeled points (never queried)

Tracked metrics:
  (1) best_history : lowest J10 found so far
  (2) oob_r2       : Bootstrap OOB R^2 (landscape understanding depth)
  (3) mean_sigma   : mean uncertainty over unlabeled pool (remaining unknowns)

Compared strategies: LCBDS (three-point batch) vs. Greedy vs. Random
Evaluation: break_exp (experiments to break threshold) + best_mV + OOB R^2
"""

import sys
import json
import argparse
import warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from sklearn.metrics import r2_score
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

warnings.filterwarnings("ignore")

ROOT   = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from src.data.loader import load_data, fit_target_scaler
from src.process_model import PCBAN

DEVICE  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ELEM    = ["Ni", "Fe", "Co", "Ce"]
RES_DIR = ROOT / "results"; RES_DIR.mkdir(exist_ok=True)
FIG_DIR = ROOT / "figures"; FIG_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────
# Space-filling initialization
# ─────────────────────────────────────────────

def diverse_init(X_oracle: np.ndarray, n_init: int, seed: int) -> list:
    """
    K-Means space-filling initialization:
      Partition composition space into n_init clusters, select one
      representative point per cluster.
      Guarantees initial points cover diverse regions of composition space.
      J10 values are completely unknown at this stage.
    """
    km     = KMeans(n_clusters=n_init, random_state=seed, n_init=10)
    labels = km.fit_predict(X_oracle)
    rng    = np.random.default_rng(seed)
    init_idx = []
    for c in range(n_init):
        cluster_pts = np.where(labels == c)[0]
        init_idx.append(int(rng.choice(cluster_pts)))
    return init_idx


# ─────────────────────────────────────────────
# Adaptive model tier selection
# ─────────────────────────────────────────────

def model_tier(n: int) -> int:
    """Return architecture tier (0=smallest, 3=largest) based on labeled set size."""
    if n < 40:  return 0
    elif n < 70:  return 1
    elif n < 120: return 2
    else:         return 3


# ─────────────────────────────────────────────
# Capacity-adaptive model
# ─────────────────────────────────────────────

def adaptive_model(n_data: int, dropout: float = 0.25) -> PCBAN:
    if n_data < 40:
        cfg = dict(embed_dim=8,  triplet_rank=2, proj_dim=12,
                   n_components=4,  hidden_dims=[32, 16])
    elif n_data < 70:
        cfg = dict(embed_dim=16, triplet_rank=2, proj_dim=20,
                   n_components=6,  hidden_dims=[64, 32])
    elif n_data < 120:
        cfg = dict(embed_dim=24, triplet_rank=3, proj_dim=24,
                   n_components=8,  hidden_dims=[96, 48])
    else:
        cfg = dict(embed_dim=32, triplet_rank=4, proj_dim=30,
                   n_components=10, hidden_dims=[128, 64])
    return PCBAN(n_elem=4, dropout=dropout, **cfg).to(DEVICE)


# ─────────────────────────────────────────────
# Bootstrap Ensemble + OOB evaluation
# ─────────────────────────────────────────────

def bootstrap_train_oob(X_tr: np.ndarray, y_tr_sc: np.ndarray,
                        scale: float, mean_: float,
                        n_models: int = 3, n_epochs: int = 250,
                        lr: float = 1e-3, seed_offset: int = 0):
    """
    Train bootstrap ensemble and compute OOB R^2.
    Each model is trained on a bootstrap resample:
      ~63% of samples used for training
      ~37% of samples are OOB (Out-of-Bag) = implicit validation set
    """
    n   = len(X_tr)
    rng = np.random.default_rng(seed_offset)

    models    = []
    oob_preds = np.full((n_models, n), np.nan)

    for m_idx in range(n_models):
        boot_idx = rng.integers(0, n, size=n)
        oob_mask = np.ones(n, dtype=bool)
        oob_mask[np.unique(boot_idx)] = False

        X_boot = X_tr[boot_idx]
        y_boot = y_tr_sc[boot_idx]

        model = adaptive_model(n)
        opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=n_epochs, eta_min=1e-5)

        X_t = torch.tensor(X_boot, dtype=torch.float32, device=DEVICE)
        y_t = torch.tensor(y_boot, dtype=torch.float32, device=DEVICE)

        model.train()
        for _ in range(n_epochs):
            loss, *_ = model.full_loss(X_t, y_t, lam_mse=0.10, lam_rec=0.02)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()

        model.eval()
        models.append(model)

        if oob_mask.sum() > 0:
            X_oob = torch.tensor(X_tr[oob_mask], dtype=torch.float32, device=DEVICE)
            with torch.no_grad():
                pred_sc = model.predict_mean(X_oob).cpu().numpy()
                oob_preds[m_idx, oob_mask] = pred_sc * scale + mean_

    y_mv     = y_tr_sc * scale + mean_
    oob_mean = np.nanmean(oob_preds, axis=0)
    valid    = ~np.isnan(oob_mean)
    oob_r2   = float(r2_score(y_mv[valid], oob_mean[valid])) if valid.sum() >= 5 else float("nan")

    return models, oob_r2


# ─────────────────────────────────────────────
# Warm start (same tier: inherit previous weights)
# ─────────────────────────────────────────────

def warm_start_bootstrap(prev_models: list,
                         X_tr: np.ndarray, y_tr_sc: np.ndarray,
                         scale: float, mean_: float,
                         n_models: int = 3, n_epochs: int = 60,
                         lr: float = 3e-4, seed_offset: int = 0):
    """
    Warm start: when architecture tier is unchanged, continue training
    from previous model weights. Only 60 epochs needed (fine-tuning).
    """
    n   = len(X_tr)
    rng = np.random.default_rng(seed_offset)
    models    = []
    oob_preds = np.full((n_models, n), np.nan)

    for m_idx in range(n_models):
        boot_idx = rng.integers(0, n, size=n)
        oob_mask = np.ones(n, dtype=bool)
        oob_mask[np.unique(boot_idx)] = False

        model = adaptive_model(n)
        model.load_state_dict(prev_models[m_idx % len(prev_models)].state_dict())

        opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=n_epochs, eta_min=1e-6)

        X_t = torch.tensor(X_tr[boot_idx], dtype=torch.float32, device=DEVICE)
        y_t = torch.tensor(y_tr_sc[boot_idx], dtype=torch.float32, device=DEVICE)

        model.train()
        for _ in range(n_epochs):
            loss, *_ = model.full_loss(X_t, y_t, lam_mse=0.10, lam_rec=0.02)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()

        model.eval()
        models.append(model)

        if oob_mask.sum() > 0:
            X_oob = torch.tensor(X_tr[oob_mask], dtype=torch.float32, device=DEVICE)
            with torch.no_grad():
                pred_sc = model.predict_mean(X_oob).cpu().numpy()
                oob_preds[m_idx, oob_mask] = pred_sc * scale + mean_

    y_mv     = y_tr_sc * scale + mean_
    oob_mean = np.nanmean(oob_preds, axis=0)
    valid    = ~np.isnan(oob_mean)
    oob_r2   = float(r2_score(y_mv[valid], oob_mean[valid])) if valid.sum() >= 5 else float("nan")
    return models, oob_r2


# ─────────────────────────────────────────────
# Knowledge distillation (tier upgrade: Teacher -> Student)
# ─────────────────────────────────────────────

def distill_bootstrap(teacher_models: list,
                      X_tr: np.ndarray, y_tr_sc: np.ndarray,
                      X_pool: np.ndarray, scale: float, mean_: float,
                      n_models: int = 3, n_epochs: int = 300,
                      alpha: float = 0.5, lr: float = 1e-3,
                      seed_offset: int = 0):
    """
    Knowledge distillation: on tier upgrade, new larger Student learns from old Teacher.
      Task Loss    : fit real J10 labels (hard labels)
      Distill Loss : imitate Teacher predictions on unlabeled pool (soft labels)
    """
    n   = len(X_tr)
    rng = np.random.default_rng(seed_offset)

    # Soft labels from Teacher on unlabeled pool (eval mode, no dropout)
    with torch.no_grad():
        mu_teacher, _ = ensemble_predict(teacher_models, X_pool, scale, mean_)
        mu_teacher_sc = torch.tensor(
            (mu_teacher - mean_) / scale, dtype=torch.float32, device=DEVICE)
    X_pool_t = torch.tensor(X_pool, dtype=torch.float32, device=DEVICE)

    models    = []
    oob_preds = np.full((n_models, n), np.nan)

    for m_idx in range(n_models):
        boot_idx = rng.integers(0, n, size=n)
        oob_mask = np.ones(n, dtype=bool)
        oob_mask[np.unique(boot_idx)] = False

        student = adaptive_model(n)
        opt     = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=1e-3)
        sched   = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=n_epochs, eta_min=1e-5)

        X_t = torch.tensor(X_tr[boot_idx], dtype=torch.float32, device=DEVICE)
        y_t = torch.tensor(y_tr_sc[boot_idx], dtype=torch.float32, device=DEVICE)

        student.train()
        for _ in range(n_epochs):
            task_loss, *_ = student.full_loss(X_t, y_t, lam_mse=0.10, lam_rec=0.02)
            pi_s, mu_s, _ = student(X_pool_t)
            mu_student_pool = (pi_s * mu_s).sum(-1)
            distill_loss = F.mse_loss(mu_student_pool, mu_teacher_sc)
            loss = alpha * task_loss + (1.0 - alpha) * distill_loss
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            opt.step(); sched.step()

        student.eval()
        models.append(student)

        if oob_mask.sum() > 0:
            X_oob = torch.tensor(X_tr[oob_mask], dtype=torch.float32, device=DEVICE)
            with torch.no_grad():
                pred_sc = student.predict_mean(X_oob).cpu().numpy()
                oob_preds[m_idx, oob_mask] = pred_sc * scale + mean_

    y_mv     = y_tr_sc * scale + mean_
    oob_mean = np.nanmean(oob_preds, axis=0)
    valid    = ~np.isnan(oob_mean)
    oob_r2   = float(r2_score(y_mv[valid], oob_mean[valid])) if valid.sum() >= 5 else float("nan")
    return models, oob_r2


def ensemble_predict(models: list, X: np.ndarray, scale: float, mean_: float):
    """Ensemble prediction (eval mode): used for distillation teacher soft labels."""
    X_t   = torch.tensor(X, dtype=torch.float32, device=DEVICE)
    preds = []
    with torch.no_grad():
        for model in models:
            pred_sc = model.predict_mean(X_t).cpu().numpy()
            preds.append(pred_sc * scale + mean_)
    preds = np.array(preds)
    return preds.mean(0), preds.std(0)


def mc_ensemble_predict(models: list, X: np.ndarray, scale: float, mean_: float,
                         n_mc: int = 20):
    """
    MC Dropout + Bootstrap Ensemble prediction:
       - Each bootstrap model runs n_mc forward passes in train() mode (Dropout active)
       - sigma = std over n_models * n_mc predictions
       - More reliable and larger sigma than pure bootstrap -> effective exploration

    Compared to pure bootstrap:
      pure bootstrap: sigma from B model differences (too small, ~1-4 mV)
      MC Dropout:     sigma from B*n_mc stochastic passes (larger, more reliable)
    """
    X_t   = torch.tensor(X, dtype=torch.float32, device=DEVICE)
    preds = []
    with torch.no_grad():
        for model in models:
            model.train()          # Enable Dropout (MC Dropout)
            for _ in range(n_mc):
                pred_sc = model.predict_mean(X_t).cpu().numpy()
                preds.append(pred_sc * scale + mean_)
            model.eval()           # Restore eval mode
    preds = np.array(preds)        # (n_models * n_mc, N)
    return preds.mean(0), preds.std(0)


# ─────────────────────────────────────────────
# Prospective pool-based active learning (core)
# ─────────────────────────────────────────────

def run_pool_al(X_oracle: np.ndarray, y_oracle: np.ndarray,
                n_init: int = 20, n_iter: int = 100,
                beta: float = 2.5, beta_min: float = 0.2,
                gamma: float = 8.0,
                delta: float = 12.0,
                surp_radius: float = 0.1,
                surp_thresh_sigma: float = 2.0,
                n_mc: int = 20,
                n_query: int = 3,
                seed: int = 42,
                strategy: str = "lcb", n_bootstrap: int = 3) -> dict:
    """
    Prospective pool-based active learning (three-point batch query):
      - Initial n_init points: K-Means space-filling
      - Each round: +n_query points with three objectives:
          (A) LCBDS   -> find lowest J10 (exploitation)
          (B) Max sigma -> fill uncertain region (exploration)
          (C) Max mu    -> understand barrier regions (contrastive)

    params:
      n_query           : queries per iteration
      beta              : initial beta (exploration weight)
      beta_min          : final beta (exploitation weight)
      gamma             : diversity penalty weight (mV equivalent)
      delta             : surprise bonus weight (mV equivalent)
      surp_radius       : surprise decay radius (composition space distance)
      surp_thresh_sigma : surprise detection threshold (multiples of sigma)
      n_mc              : MC Dropout samples per model
    """
    rng   = np.random.default_rng(seed)
    N     = len(X_oracle)

    init_idx = diverse_init(X_oracle, n_init, seed)
    labeled  = set(init_idx)
    X_tr     = X_oracle[init_idx].copy()
    y_tr     = y_oracle[init_idx].copy()

    best_history     = [float(y_tr.min())]
    oob_r2_history   = []
    sigma_history    = []
    beta_history     = []
    surprise_pts     = []   # [(x_point, residual_mV), ...]  surprise event list
    surprise_history = []   # [(iter, residual_mV), ...]     surprise occurrence log
    prev_models      = None
    prev_tier        = None

    for it in range(n_iter):
        # Dynamic scaler: use only currently labeled data
        mean_ = float(y_tr.mean())
        scale = float(y_tr.std()) + 1e-8
        y_tr_sc   = (y_tr - mean_) / scale
        unlabeled = list(set(range(N)) - labeled)
        X_pool    = X_oracle[unlabeled]
        curr_tier = model_tier(len(X_tr))

        # beta annealing: cosine decay from beta to beta_min
        beta_t = beta_min + (beta - beta_min) * np.cos(np.pi * it / (2 * n_iter))
        beta_history.append(float(beta_t))

        # Three training modes: scratch / distill / warm-start
        if prev_models is None:
            models, oob_r2 = bootstrap_train_oob(
                X_tr, y_tr_sc, scale, mean_,
                n_models=n_bootstrap, n_epochs=250,
                seed_offset=seed * 10000 + it)
            mode = "scratch"

        elif curr_tier != prev_tier:
            models, oob_r2 = distill_bootstrap(
                prev_models, X_tr, y_tr_sc, X_pool, scale, mean_,
                n_models=n_bootstrap, n_epochs=300, alpha=0.5,
                seed_offset=seed * 10000 + it)
            mode = "distill"

        else:
            models, oob_r2 = warm_start_bootstrap(
                prev_models, X_tr, y_tr_sc, scale, mean_,
                n_models=n_bootstrap, n_epochs=60, lr=3e-4,
                seed_offset=seed * 10000 + it)
            mode = "warm"

        prev_models = models
        prev_tier   = curr_tier
        oob_r2_history.append(oob_r2)

        # MC Dropout ensemble prediction (more reliable sigma)
        mu_pool, sig_pool = mc_ensemble_predict(models, X_pool, scale, mean_, n_mc=n_mc)
        sigma_history.append(float(sig_pool.mean()))

        # Three-point batch query: compute shared LCBDS base scores
        dist_to_labeled = cdist(X_pool, X_tr, metric="euclidean")
        min_dist        = dist_to_labeled.min(axis=1)
        dist_norm       = min_dist / (min_dist.max() + 1e-8)

        surp_bonus = np.zeros(len(X_pool))
        if surprise_pts:
            for x_surp, w in surprise_pts[-10:]:
                d = cdist(X_pool, x_surp.reshape(1, -1),
                          metric="euclidean").ravel()
                surp_bonus += w * np.exp(-d / surp_radius)
            surp_bonus = surp_bonus / (surp_bonus.max() + 1e-8)

        # Select n_query points sequentially with different objectives (no duplicates)
        available = list(range(len(unlabeled)))   # local indices into pool
        queries_local = []

        for q_i in range(min(n_query, len(available))):
            mask   = np.array(available)
            mu_av  = mu_pool[mask]
            sig_av = sig_pool[mask]
            dn_av  = dist_norm[mask]
            sb_av  = surp_bonus[mask]

            if strategy == "lcb":
                if q_i == 0:
                    # Point A: LCBDS (exploitation + diversity + surprise)
                    score = mu_av - beta_t*sig_av - gamma*dn_av - delta*sb_av
                    pick  = mask[int(np.argmin(score))]
                elif q_i == 1:
                    # Point B: Max sigma (fill most uncertain region)
                    pick  = mask[int(np.argmax(sig_av))]
                else:
                    # Point C: Max mu (understand highest J10 region)
                    pick  = mask[int(np.argmax(mu_av))]
            elif strategy == "greedy":
                # Greedy: always select lowest predicted mu
                pick = mask[int(np.argmin(mu_av))]
            else:
                # Random: uniform random selection
                pick = mask[int(rng.integers(0, len(mask)))]

            queries_local.append(pick)
            available = [a for a in available if a != pick]

        # Add queries to labeled set (simulate a batch of experiments)
        for q_local in queries_local:
            mu_queried = float(mu_pool[q_local])
            query_idx  = unlabeled[q_local]

            labeled.add(query_idx)
            X_tr = np.vstack([X_tr, X_oracle[query_idx]])
            y_tr = np.append(y_tr, y_oracle[query_idx])

            # Surprise detection
            residual_mV = abs(float(y_oracle[query_idx]) - mu_queried)
            surp_thresh = surp_thresh_sigma * scale
            if residual_mV > surp_thresh:
                surprise_pts.append((X_oracle[query_idx].copy(), residual_mV))
                surprise_history.append((it + 1, float(residual_mV)))
                strat_label = "LCBDS" if strategy == "lcb" else strategy.upper()
                print(f"    [SURPRISE] [{strat_label}] Iter {it+1} | "
                      f"residual={residual_mV:.1f} mV "
                      f"(predicted={mu_queried:.1f}, actual={y_oracle[query_idx]:.1f})")

        best_history.append(float(y_tr.min()))

        if (it + 1) % 10 == 0:
            strat_label = "LCBDS" if strategy == "lcb" else strategy.upper()
            n_surp = len(surprise_pts)
            surp_mark = f" ★×{n_surp}" if n_surp > 0 else ""
            print(f"    [{strat_label}] Iter {it+1:3d} | "
                  f"n={len(y_tr):3d} | best={y_tr.min():.1f} mV | "
                  f"OOB R²={oob_r2:.3f} | σ̄={sig_pool.mean():.1f} mV | "
                  f"β={beta_t:.2f} | [{mode}]{surp_mark}")

    # Final model training (for held-out evaluation)
    final_y_sc = (y_tr - mean_) / scale
    final_models, _ = bootstrap_train_oob(
        X_tr, final_y_sc, scale, mean_,
        n_models=n_bootstrap, n_epochs=300, seed_offset=seed * 99999,
    )

    # Natural test set evaluation (never-queried points, evaluated only after AL ends)
    unlabeled_final = list(set(range(N)) - labeled)
    X_test_nat = X_oracle[unlabeled_final]
    y_test_nat = y_oracle[unlabeled_final]
    mu_nat, _ = ensemble_predict(final_models, X_test_nat, scale, mean_)
    final_natural_r2 = float(r2_score(y_test_nat, mu_nat))

    strat_label = "LCBDS" if strategy == "lcb" else strategy.upper()
    print(f"    [{strat_label}] natural_R2={final_natural_r2:.4f} ({len(y_test_nat)} pts) | "
          f"best={y_tr.min():.1f} mV | surprises:{len(surprise_history)}")

    return {
        "best"              : best_history,
        "oob_r2"            : oob_r2_history,
        "sigma"             : sigma_history,
        "beta"              : beta_history,
        "surprise_history"  : surprise_history,
        "final_natural_r2"  : final_natural_r2,
        "n_labeled"         : len(labeled),
        "n_test_natural"    : len(unlabeled_final),
        # Ensemble models for analysis (5-seed x 5-bootstrap)
        "final_models"      : final_models,
        "X_tr"              : X_tr,
        "y_tr"              : y_tr,
        "mean_"             : mean_,
        "scale_"            : scale,
        "labeled_idx"       : sorted(labeled),
    }


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=str(ROOT / "config.yaml"))
    args = parser.parse_args()
    cfg_path = Path(args.config)
    cfg      = yaml.safe_load(open(cfg_path))
    cfg_name = cfg_path.stem   # e.g. "al_aggressive"
    print(f"[ActiveLearning v6] Device: {DEVICE}")
    print("[ActiveLearning v6] Three-point batch query (exploitation+exploration+contrastive) + Bootstrap OOB\n")

    df    = load_data(ROOT / cfg["data"]["path"])
    y_all = df["J10"].values.astype(np.float32)
    X_all = df[ELEM].values.astype(np.float32)

    print(f"[Data] Loaded {len(df)} samples")
    print(f"  Global minimum J10: {y_all.min():.1f} mV (unknown to the AL agent)")
    for thr in [360, 370, 380]:
        cnt = (y_all < thr).sum()
        print(f"  J10 < {thr} mV: {cnt} samples ({cnt/len(y_all)*100:.2f}%)")


    al          = cfg["active_learning"]
    N_INIT      = al["n_init"]
    N_ITER      = al["n_iter"]
    N_QUERY     = al["n_query"]
    N_SEEDS     = al["n_seeds"]
    SEED_START  = al.get("seed_start", 0)
    N_BOOTSTRAP = al["n_bootstrap"]
    N_MC        = al["n_mc"]
    THRESHOLD   = al["threshold_mV"]
    BETA        = al["beta"]
    BETA_MIN    = al["beta_min"]
    GAMMA       = al["gamma"]
    DELTA       = al["delta"]
    SURP_RADIUS = al["surp_radius"]
    SURP_SIGMA  = al["surp_thresh_sigma"]

    print(f"\n  [config] n_query={N_QUERY} | β:{BETA}→{BETA_MIN} | "
          f"γ={GAMMA} | δ={DELTA} | surp_r={SURP_RADIUS} | σ={SURP_SIGMA}")
    print(f"  {N_QUERY} queries/iter | {N_ITER} iters | total {N_INIT + N_ITER*N_QUERY} labeled | "
          f"seeds x{N_SEEDS} | bootstrap x{N_BOOTSTRAP}\n")

    strategies = ["lcb", "greedy", "random"]
    histories  = {s: [] for s in strategies}

    for seed in range(SEED_START, SEED_START + N_SEEDS):
        print(f"── Seed {seed+1}/{SEED_START + N_SEEDS} (seed={seed}) ──")
        for strat in strategies:
            strat_label = "LCBDS" if strat == "lcb" else strat.upper()
            print(f"  [{strat_label}]")
            result = run_pool_al(
                X_all, y_all,
                n_init             = N_INIT,
                n_iter             = N_ITER,
                beta               = BETA,
                beta_min           = BETA_MIN,
                gamma              = GAMMA,
                delta              = DELTA,
                surp_radius        = SURP_RADIUS,
                surp_thresh_sigma  = SURP_SIGMA,
                n_mc               = N_MC,
                n_query            = N_QUERY,
                seed               = seed,
                strategy           = strat,
                n_bootstrap        = N_BOOTSTRAP,
            )
            histories[strat].append(result)

        for strat in strategies:
            r = histories[strat][-1]
            strat_label = "LCBDS" if strat == "lcb" else strat.upper()
            n_surp = len(r.get("surprise_history", []))
            print(f"  {strat_label:<8} -> "
                  f"best={r['best'][-1]:.1f} mV | "
                  f"OOB R2={r['oob_r2'][-1]:.3f} | "
                  f"natural_R2={r['final_natural_r2']:.4f} | "
                  f"surprises={n_surp}")
        print()

    # Statistics (each iteration adds N_QUERY points)
    total_pts   = N_INIT + N_ITER * N_QUERY
    n_exps_best = np.array([N_INIT + i * N_QUERY for i in range(N_ITER + 1)])
    n_exps_r2   = n_exps_best[1:]

    def break_exp(hist_list, thr):
        """Return cumulative experiments when J10 first drops below thr."""
        results = []
        for d in hist_list:
            # best_history[k] corresponds to N_INIT + k*N_QUERY labeled samples
            hits = [i for i, v in enumerate(d["best"]) if v < thr]
            results.append(N_INIT + hits[0] * N_QUERY
                           if hits else N_INIT + N_ITER * N_QUERY)
        return np.array(results)

    print("=" * 60)
    print(f"Experiments to break {THRESHOLD:.0f} mV threshold (mean +- std):")
    for strat in strategies:
        be = break_exp(histories[strat], THRESHOLD)
        strat_label = "LCBDS" if strat == "lcb" else strat.upper()
        print(f"  {strat_label:<8}: {be.mean():.1f} +- {be.std():.1f}")

    lcb_be = break_exp(histories["lcb"],    THRESHOLD)
    rnd_be = break_exp(histories["random"], THRESHOLD)
    print(f"  Experiments saved (LCBDS vs Random): {rnd_be.mean()-lcb_be.mean():.1f}")

    print(f"\nFinal OOB R^2 (mean over seeds):")
    for strat in strategies:
        r2s = np.array([d["oob_r2"][-1] for d in histories[strat]])
        strat_label = "LCBDS" if strat == "lcb" else strat.upper()
        print(f"  {strat_label:<8}: {r2s.mean():.4f} +- {r2s.std():.4f}")

    print(f"\nNatural test set R^2 (never-queried points, evaluated after AL):")
    for strat in strategies:
        r2s = np.array([d["final_natural_r2"] for d in histories[strat]])
        n_t = histories[strat][0]["n_test_natural"]
        strat_label = "LCBDS" if strat == "lcb" else strat.upper()
        print(f"  {strat_label:<8}: {r2s.mean():.4f} +- {r2s.std():.4f}  "
              f"({n_t} test samples)")

    print(f"\nSurprise event statistics (LCBDS):")
    surp_counts = [len(d.get("surprise_history", [])) for d in histories["lcb"]]
    print(f"  Surprises per seed: {surp_counts}")
    print(f"  Mean: {np.mean(surp_counts):.1f} +- {np.std(surp_counts):.1f} per seed")
    print("=" * 60)

    # Diagnostic plots (three panels)
    colors = {"lcb": "blue", "greedy": "green", "random": "red"}
    labels = {"lcb": "LCBDS (surprise+diversity+annealing)",
              "greedy": "Greedy (pure exploitation)",
              "random": "Random (baseline)"}
    styles = {"lcb": "o-", "greedy": "s--", "random": "^:"}

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    ax1, ax2, ax3 = axes

    # Panel 1: Best J10 convergence curve
    for strat in strategies:
        arr = np.array([d["best"] for d in histories[strat]])
        mu, std = arr.mean(0), arr.std(0)
        ax1.plot(n_exps_best, mu, styles[strat], ms=3, lw=2,
                 color=colors[strat], label=labels[strat])
        ax1.fill_between(n_exps_best, mu-std, mu+std, alpha=0.15, color=colors[strat])
    ax1.axhline(THRESHOLD, color="purple", ls="--", lw=1.5,
                label=f"Target {THRESHOLD:.0f} mV", alpha=0.8)
    ax1.set_xlabel("Labeled samples", fontsize=11)
    ax1.set_ylabel("Best J10 found (mV)", fontsize=11)
    ax1.set_title("(a) Best J10 Convergence\nA=LCBDS, B=MaxSigma, C=MaxMu", fontweight="bold")
    ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3); ax1.invert_yaxis()

    # Panel 2: OOB R^2 learning curve
    for strat in strategies:
        arr = np.array([d["oob_r2"] for d in histories[strat]])
        mu, std = arr.mean(0), arr.std(0)
        ax2.plot(n_exps_r2, mu, styles[strat], ms=3, lw=2,
                 color=colors[strat], label=labels[strat])
        ax2.fill_between(n_exps_r2, mu-std, mu+std, alpha=0.15, color=colors[strat])
    ax2.set_xlabel("Labeled samples", fontsize=11)
    ax2.set_ylabel("Bootstrap OOB R^2", fontsize=11)
    ax2.set_title("(b) Surrogate Quality (OOB R^2)\nMaxMu improves landscape generalization", fontweight="bold")
    ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)

    # Panel 3: Mean uncertainty sigma
    for strat in strategies:
        arr = np.array([d["sigma"] for d in histories[strat]])
        mu, std = arr.mean(0), arr.std(0)
        ax3.plot(n_exps_r2, mu, styles[strat], ms=3, lw=2,
                 color=colors[strat], label=labels[strat])
        ax3.fill_between(n_exps_r2, (mu-std).clip(0), mu+std,
                         alpha=0.15, color=colors[strat])
    ax3.set_xlabel("Labeled samples", fontsize=11)
    ax3.set_ylabel("Mean uncertainty sigma-bar (mV)", fontsize=11)
    ax3.set_title("(c) Residual Uncertainty\nMaxSigma actively reduces uncertainty", fontweight="bold")
    ax3.legend(fontsize=8); ax3.grid(True, alpha=0.3)

    plt.suptitle(
        "Prospective AL v6: Three-point batch query\n"
        "A=LCBDS(exploitation) + B=MaxSigma(exploration) + C=MaxMu(contrastive)",
        fontsize=12, fontweight="bold")
    plt.tight_layout()

    save_path = FIG_DIR / f"active_learning_v6_{cfg_name}_curve.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[ActiveLearning v6] Figure -> {save_path}")

    # Save results to JSON
    out = {
        "version"          : f"v6 n_query={N_QUERY} beta:{BETA}->{BETA_MIN} gamma={GAMMA} delta={DELTA}",
        "design"           : "Prospective AL, Bootstrap OOB evaluation, key metrics: break_exp + best_mV",
        "improvements"     : [
            f"Three-point batch: A=LCBDS B=MaxSigma C=MaxMu (n_query={N_QUERY})",
            "MC Dropout n_mc=20",
            f"LCBDS gamma={GAMMA}mV delta={DELTA}mV",
            f"beta annealing {BETA}→{BETA_MIN}",
            f"surprise curiosity r={SURP_RADIUS} thresh={SURP_SIGMA}sigma",
        ],
        "n_init"           : N_INIT,
        "n_iter"           : N_ITER,
        "n_query"          : N_QUERY,
        "total_labeled"    : N_INIT + N_ITER * N_QUERY,
        "n_seeds"          : N_SEEDS,
        "n_bootstrap"      : N_BOOTSTRAP,
        "beta"             : BETA,
        "beta_min"         : BETA_MIN,
        "gamma_mV"         : GAMMA,
        "delta_mV"         : DELTA,
        "surp_radius"      : SURP_RADIUS,
        "surp_thresh_sigma": SURP_SIGMA,
        "n_mc"             : N_MC,
        "mean_surprise_count": float(np.mean(surp_counts)),
        "total_data"       : int(len(df)),
        "threshold_mV"     : THRESHOLD,
        "config_name"      : cfg_name,
        "lcb_break"        : {"mean": float(lcb_be.mean()), "std": float(lcb_be.std())},
        "rnd_break"        : {"mean": float(rnd_be.mean()), "std": float(rnd_be.std())},
        "greedy_break"     : {"mean": float(break_exp(histories["greedy"], THRESHOLD).mean()),
                              "std":  float(break_exp(histories["greedy"], THRESHOLD).std())},
        "experiments_saved": float(rnd_be.mean() - lcb_be.mean()),
        "final_oob_r2"     : {s: float(np.mean([d["oob_r2"][-1] for d in histories[s]]))
                              for s in strategies},
        "final_natural_r2" : {s: float(np.mean([d["final_natural_r2"] for d in histories[s]]))
                              for s in strategies},
        "final_best_mV"    : {s: float(np.mean([d["best"][-1] for d in histories[s]]))
                              for s in strategies},
        "natural_test_size": histories["lcb"][0]["n_test_natural"],
        # Per-iteration curve data (for Fig 2 plotting)
        "lcb_best_history"   : [d["best"]   for d in histories["lcb"]],
        "rnd_best_history"   : [d["best"]   for d in histories["random"]],
        "greedy_best_history": [d["best"]   for d in histories["greedy"]],
        "lcb_oob_history"    : [d["oob_r2"] for d in histories["lcb"]],
        "rnd_oob_history"    : [d["oob_r2"] for d in histories["random"]],
        "greedy_oob_history" : [d["oob_r2"] for d in histories["greedy"]],
        "lcb_sigma_history"  : [d["sigma"]  for d in histories["lcb"]],
        "rnd_sigma_history"  : [d["sigma"]  for d in histories["random"]],
        "greedy_sigma_history": [d["sigma"] for d in histories["greedy"]],
        # Per-seed best_mV tracking (to detect if 206 mV was found)
        "lcb_best_per_seed"  : [float(min(d["best"])) for d in histories["lcb"]],
        "rnd_best_per_seed"  : [float(min(d["best"])) for d in histories["random"]],
        "greedy_best_per_seed": [float(min(d["best"])) for d in histories["greedy"]],
    }
    out_path = RES_DIR / f"active_learning_v6_{cfg_name}_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[ActiveLearning v6] Saved → {out_path}")


if __name__ == "__main__":
    main()
