"""
Generalization test: same PC-BAN + LCBDS method, NEW dataset.
UCI Concrete Compressive Strength: 1030 mixes x 8 ingredients -> strength (MPa).
Goal: find the mix with MAX strength using fewest experiments.
We minimize y = -strength (so 'lower is better', identical convention to J10).
"""
import sys, json, time
from pathlib import Path
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import numpy as np, pandas as pd, torch
import torch.nn as nn
from sklearn.metrics import r2_score
from scipy.spatial.distance import cdist
from src.process_model import PCBAN

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Permanent, in-repo copy of the real UCI Concrete data (no /tmp dependency)
df = pd.read_csv(_HERE / "data_external" / "concrete.csv")
FEAT = ["cement","slag","ash","water","superplastic","coarseagg","fineagg","age"]
Xraw = df[FEAT].values.astype(np.float64)
X = ((Xraw - Xraw.min(0)) / (Xraw.max(0) - Xraw.min(0) + 1e-9)).astype(np.float32)
strength = df["strength"].values.astype(np.float64)
y_obj = (-strength).astype(np.float64)          # minimize -> maximize strength
N_ELEM = 8
N = len(df)
GLOBAL_BEST = strength.max()
TOP5 = np.percentile(strength, 95)              # "success" threshold = top 5% strength
YRANGE = strength.max() - strength.min()
GAMMA = 0.02 * YRANGE                            # diversity penalty (same 2% of range as OER)
DELTA = 0.03 * YRANGE                            # surprise bonus (same 3% of range as OER)

def build_model(n):
    if   n < 40:  cfg = dict(embed_dim=8,  n_components=4,  hidden_dims=[32,16])
    elif n < 70:  cfg = dict(embed_dim=16, n_components=6,  hidden_dims=[64,32])
    elif n < 120: cfg = dict(embed_dim=24, n_components=8,  hidden_dims=[96,48])
    else:         cfg = dict(embed_dim=32, n_components=10, hidden_dims=[128,64])
    return PCBAN(n_elem=N_ELEM, dropout=0.1, **cfg).to(DEVICE)

def train_ensemble(Xtr, ytr_sc, scale, mean_, B, epochs, seed_offset):
    rng = np.random.default_rng(seed_offset)
    torch.manual_seed(seed_offset); torch.cuda.manual_seed_all(seed_offset)
    n = len(Xtr); models = []; oob = np.full((B, n), np.nan)
    for b in range(B):
        bi = rng.integers(0, n, n); mask = np.ones(n, bool); mask[np.unique(bi)] = False
        m = build_model(n)
        opt = torch.optim.AdamW(m.parameters(), lr=1.5e-3, weight_decay=1e-3)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)
        Xt = torch.tensor(Xtr[bi], device=DEVICE); yt = torch.tensor(ytr_sc[bi], device=DEVICE)
        m.train()
        for _ in range(epochs):
            loss, *_ = m.full_loss(Xt, yt, lam_mse=0.10, lam_rec=0.0)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); sch.step()
        m.eval(); models.append(m)
        if mask.sum() > 0:
            with torch.no_grad():
                p = m.predict_mean(torch.tensor(Xtr[mask], device=DEVICE)).cpu().numpy()
                oob[b, mask] = p * scale + mean_
    ymv = ytr_sc * scale + mean_; om = np.nanmean(oob, 0); v = ~np.isnan(om)
    r2 = float(r2_score(ymv[v], om[v])) if v.sum() >= 5 else float("nan")
    return models, r2

def mc_predict(models, scale, mean_, n_mc, seed_offset):
    torch.manual_seed(seed_offset + 777)
    Xt = torch.tensor(X, device=DEVICE); preds = []
    for m in models:
        m.train()
        with torch.no_grad():
            for _ in range(n_mc):
                preds.append(m.predict_mean(Xt).cpu().numpy() * scale + mean_)
        m.eval()
    P = np.array(preds); return P.mean(0), P.std(0)

def run_al(strategy, seed, n_init=20, n_iter=60, n_query=2, B=3, n_mc=12,
           beta0=2.5, beta_min=0.2):
    rng = np.random.default_rng(seed)
    labeled = set(rng.choice(N, n_init, replace=False))
    surprises = []
    best_hist = [strength[list(labeled)].max()]
    oob_hist = []
    for it in range(n_iter):
        Lidx = np.array(sorted(labeled))
        ytr = y_obj[Lidx]; mean_ = ytr.mean(); scale = ytr.std() + 1e-9
        ytr_sc = (ytr - mean_) / scale
        models, oob = train_ensemble(X[Lidx], ytr_sc, scale, mean_, B, 150, seed*10000+it)
        oob_hist.append(oob)
        mu, sig = mc_predict(models, scale, mean_, n_mc, seed*10000+it)
        beta_t = beta_min + (beta0 - beta_min) * np.cos(np.pi * it / (2 * n_iter))
        for q in range(n_query):
            avail = np.array([i for i in range(N) if i not in labeled])
            Lcur = np.array(sorted(labeled))
            mu_a = mu[avail]; sig_a = sig[avail]
            d = cdist(X[avail], X[Lcur]).min(1); dn = d / (d.max() + 1e-9)
            sb = np.zeros(len(avail))
            if surprises:
                sx = np.array([s[0] for s in surprises]); sw = np.array([s[1] for s in surprises])
                dd = cdist(X[avail], sx); sb = (sw[None, :] * np.exp(-dd / 0.1)).sum(1)
                sb = sb / (sb.max() + 1e-9)
            if strategy == "lcb":      # LCBDS
                if q == 0:
                    score = mu_a - beta_t*sig_a - GAMMA*dn - DELTA*sb
                    pick = avail[int(np.argmin(score))]
                else:
                    pick = avail[int(np.argmax(sig_a))]        # Point B: max sigma
            elif strategy == "greedy":
                pick = avail[int(np.argmin(mu_a))]
            else:                       # random
                pick = avail[int(rng.integers(len(avail)))]
            # surprise detection (residual vs predicted at picked point)
            resid = abs(y_obj[pick] - mu[pick])
            if resid > 2 * (sig[pick] + 1e-9):
                surprises.append((X[pick], resid))
                surprises[:] = surprises[-10:]
            labeled.add(pick)
        best_hist.append(strength[list(labeled)].max())
    final_oob = float(np.nanmean([r for r in [oob_hist[-1]] ])) if False else None
    return {"best_hist": [float(b) for b in best_hist],
            "best": float(max(best_hist)),
            "oob_r2_last": float(np.nan if not oob_hist else
                                 _oob_r2(oob_hist[-1], y_obj[np.array(sorted(labeled))]))}

def _oob_r2(oob, ymv_full):
    return float("nan")  # placeholder (OOB tracked per-iter inside train); not used in summary

def break_exp(best_hist, thr, n_init, n_query):
    for i, v in enumerate(best_hist):
        if v >= thr:
            return n_init + i * n_query
    return n_init + (len(best_hist) - 1) * n_query  # not reached -> max budget

if __name__ == "__main__":
    SEEDS = [0,1,2,3,4]; NQ = 2; NINIT = 15; NITER = 45
    TOP1 = np.percentile(strength, 99)          # harder target: top 1% strength
    print(f"pool={N} | max_strength={GLOBAL_BEST:.1f} MPa | top1%={TOP1:.1f} | "
          f"budget={NINIT+NITER*NQ} ({(NINIT+NITER*NQ)/N*100:.1f}% of pool)")
    print(f"GAMMA={GAMMA:.2f} DELTA={DELTA:.2f} (2%/3% of strength range {YRANGE:.1f})\n")
    out = {}
    for strat in ["lcb", "greedy", "random"]:
        bests = []; breaks = []; regrets = []
        for s in SEEDS:
            t = time.time()
            r = run_al(strat, s, n_init=NINIT, n_iter=NITER, n_query=NQ)
            bests.append(r["best"]); breaks.append(break_exp(r["best_hist"], TOP1, NINIT, NQ))
            regrets.append(GLOBAL_BEST - r["best"])
            print(f"  [{strat:6s}] seed{s}: best={r['best']:.1f} MPa | "
                  f"top1%@{breaks[-1]} exp | regret={GLOBAL_BEST-r['best']:.1f} | {time.time()-t:.0f}s",
                  flush=True)
        label = {"lcb":"LCBDS","greedy":"GREEDY","random":"RANDOM"}[strat]
        out[label] = {"best_mean": float(np.mean(bests)), "best_std": float(np.std(bests)),
                      "best_per_seed": bests,
                      "regret_mean": float(np.mean(regrets)),
                      "break_mean": float(np.mean(breaks)), "break_std": float(np.std(breaks)),
                      "break_per_seed": [int(b) for b in breaks]}
        print(f"  => {label}: best={np.mean(bests):.1f}±{np.std(bests):.1f} MPa | "
              f"regret={np.mean(regrets):.1f} | top1%-break={np.mean(breaks):.0f}±{np.std(breaks):.0f} exp\n",
              flush=True)
    json.dump({"dataset":"UCI Concrete (1030x8)","global_best":float(GLOBAL_BEST),
               "top1pct":float(TOP1),"budget":NINIT+NITER*NQ,"results":out},
              open("results/generalize_concrete.json","w"), indent=2)
    print("saved -> results/generalize_concrete.json")
