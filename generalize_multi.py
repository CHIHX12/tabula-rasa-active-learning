"""
Generalization test across MULTIPLE real public datasets.
Same PC-BAN + LCBDS method (μ - βσ - γ·diversity - δ·surprise, β-annealing,
bootstrap + MC-dropout). For each dataset we compare LCBDS vs Greedy vs Random.

Convention: we always MINIMIZE an internal objective y_obj.
  - "maximize strength/quality"  -> y_obj = -target
  - "minimize energy load"       -> y_obj = +target
"best" is reported back in the original target units.

Usage:  python3 generalize_multi.py [name1 name2 ...]   (default: all, idempotent)
"""
import sys, json, time, warnings
warnings.filterwarnings("ignore")
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
DX = _HERE / "data_external"

# name -> (csv, sep, feature cols, target col, direction, n_init, n_iter)
DATASETS = {
    "concrete": (DX/"concrete.csv", ",",
        ["cement","slag","ash","water","superplastic","coarseagg","fineagg","age"],
        "strength", "max", 15, 45),
    "slump": (DX/"slump.csv", ",",
        ["Cement","Slag","Fly ash","Water","SP","Coarse Aggr.","Fine Aggr."],
        "Compressive Strength (28-day)(Mpa)", "max", 8, 12),
    "wine_red": (DX/"wine_red.csv", ";",
        ["fixed acidity","volatile acidity","citric acid","residual sugar","chlorides",
         "free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol"],
        "quality", "max", 15, 40),
    "energy": (DX/"energy.csv", ",",
        ["X1","X2","X3","X4","X5","X6","X7","X8"],
        "Y1", "min", 15, 40),
    "steel": (DX/"steel_strength.csv", ",",
        ["c","mn","si","cr","ni","mo","v","n","nb","co","w","al","ti"],
        "yield strength", "max", 12, 20),
}

def build_model(n, n_elem):
    if   n < 40:  cfg = dict(embed_dim=8,  n_components=4,  hidden_dims=[32,16])
    elif n < 70:  cfg = dict(embed_dim=16, n_components=6,  hidden_dims=[64,32])
    elif n < 120: cfg = dict(embed_dim=24, n_components=8,  hidden_dims=[96,48])
    else:         cfg = dict(embed_dim=32, n_components=10, hidden_dims=[128,64])
    return PCBAN(n_elem=n_elem, dropout=0.1, **cfg).to(DEVICE)

def train_ensemble(X, Xtr, ytr_sc, scale, mean_, B, epochs, seed_offset, n_elem):
    rng = np.random.default_rng(seed_offset)
    torch.manual_seed(seed_offset); torch.cuda.manual_seed_all(seed_offset)
    n = len(Xtr); models = []; oob = np.full((B, n), np.nan)
    for b in range(B):
        bi = rng.integers(0, n, n); mask = np.ones(n, bool); mask[np.unique(bi)] = False
        m = build_model(n, n_elem)
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

def mc_predict(models, X, scale, mean_, n_mc, seed_offset):
    torch.manual_seed(seed_offset + 777)
    Xt = torch.tensor(X, device=DEVICE); preds = []
    for m in models:
        m.train()
        with torch.no_grad():
            for _ in range(n_mc):
                preds.append(m.predict_mean(Xt).cpu().numpy() * scale + mean_)
        m.eval()
    P = np.array(preds); return P.mean(0), P.std(0)

def run_al(X, y_obj, target_orig, sign, strategy, seed, n_init, n_iter,
           n_query=2, B=3, n_mc=12, beta0=2.5, beta_min=0.2, gamma=0.0, delta=0.0):
    N, n_elem = X.shape
    rng = np.random.default_rng(seed)
    labeled = set(rng.choice(N, n_init, replace=False))
    surprises = []
    # best = best ORIGINAL target seen (max or min depending on sign)
    def cur_best():
        vals = target_orig[list(labeled)]
        return float(vals.max() if sign < 0 else vals.min())
    best_hist = [cur_best()]; natr2_hist = []; oob_last = float("nan")
    for it in range(n_iter):
        Lidx = np.array(sorted(labeled))
        ytr = y_obj[Lidx]; mean_ = ytr.mean(); scale = ytr.std() + 1e-9
        models, oob = train_ensemble(X, X[Lidx], (ytr-mean_)/scale, scale, mean_,
                                     B, 120, seed*10000+it, n_elem)
        oob_last = oob
        mu, sig = mc_predict(models, X, scale, mean_, n_mc, seed*10000+it)
        # record surrogate quality on the current unexplored pool (for curves)
        av_now = np.array([i for i in range(N) if i not in labeled])
        natr2_hist.append(float(r2_score(y_obj[av_now], mu[av_now])) if len(av_now) >= 5 else float("nan"))
        beta_t = beta_min + (beta0 - beta_min) * np.cos(np.pi * it / (2 * n_iter))
        for q in range(n_query):
            avail = np.array([i for i in range(N) if i not in labeled])
            if len(avail) == 0: break
            Lcur = np.array(sorted(labeled))
            mu_a = mu[avail]; sig_a = sig[avail]
            d = cdist(X[avail], X[Lcur]).min(1); dn = d / (d.max() + 1e-9)
            sb = np.zeros(len(avail))
            if surprises:
                sx = np.array([s[0] for s in surprises]); sw = np.array([s[1] for s in surprises])
                dd = cdist(X[avail], sx); sb = (sw[None,:]*np.exp(-dd/0.1)).sum(1)
                sb = sb / (sb.max() + 1e-9)
            if strategy == "lcb":
                if q == 0:
                    pick = avail[int(np.argmin(mu_a - beta_t*sig_a - gamma*dn - delta*sb))]
                else:
                    pick = avail[int(np.argmax(sig_a))]
            elif strategy == "greedy":
                pick = avail[int(np.argmin(mu_a))]
            else:
                pick = avail[int(rng.integers(len(avail)))]
            if abs(y_obj[pick] - mu[pick]) > 2*(sig[pick]+1e-9):
                surprises.append((X[pick], abs(y_obj[pick]-mu[pick]))); surprises[:] = surprises[-10:]
            labeled.add(pick)
        best_hist.append(cur_best())
    # MODEL-QUALITY axis: final ensemble R^2 on the UNEXPLORED pool.
    # This is where LCBDS beats Greedy (Greedy over-exploits -> poor coverage).
    avail_final = np.array([i for i in range(N) if i not in labeled])
    if len(avail_final) >= 5:
        with torch.no_grad():
            Xt = torch.tensor(X[avail_final], device=DEVICE)
            preds = np.mean([m.predict_mean(Xt).cpu().numpy() for m in models], axis=0) * scale + mean_
        natural_r2 = float(r2_score(y_obj[avail_final], preds))
    else:
        natural_r2 = float("nan")
    return {"best_hist": best_hist, "natr2_hist": natr2_hist, "natural_r2": natural_r2}

def evaluate(name, seeds=(0,1,2,3,4)):
    csv, sep, feat, tgt, direction, n_init, n_iter = DATASETS[name]
    df = pd.read_csv(csv, sep=sep)
    df.columns = [c.strip() for c in df.columns]
    feat = [f.strip() for f in feat]
    Xraw = df[feat].values.astype(np.float64)
    X = ((Xraw - Xraw.min(0)) / (Xraw.max(0) - Xraw.min(0) + 1e-9)).astype(np.float32)
    target = df[tgt].values.astype(np.float64)
    sign = -1.0 if direction == "max" else 1.0      # minimize y_obj = sign*target
    y_obj = sign * target
    yrange = target.max() - target.min()
    gamma = 0.02 * yrange; delta = 0.03 * yrange
    global_best = target.max() if direction == "max" else target.min()
    thr = np.percentile(target, 99 if direction == "max" else 1)
    budget = n_init + n_iter*2
    print(f"\n##### {name}: pool={len(df)} n_elem={len(feat)} | target={tgt} ({direction}) "
          f"| global_best={global_best:.2f} | budget={budget} ({budget/len(df)*100:.0f}%)", flush=True)
    out = {}
    for strat in ["lcb", "greedy", "random"]:
        bests = []; nat = []; best_curves = []; natr2_curves = []
        for s in seeds:
            t = time.time()
            r = run_al(X, y_obj, target, sign, strat, s, n_init, n_iter,
                       gamma=gamma, delta=delta)
            b = max(r["best_hist"]) if direction == "max" else min(r["best_hist"])
            bests.append(b); nat.append(r["natural_r2"])
            best_curves.append(r["best_hist"]); natr2_curves.append(r["natr2_hist"])
            print(f"  [{strat:6s}] seed{s}: best={b:.2f} | natR2={r['natural_r2']:.3f} | {time.time()-t:.0f}s", flush=True)
        label = {"lcb":"LCBDS","greedy":"GREEDY","random":"RANDOM"}[strat]
        bc = np.array(best_curves); nc = np.array(natr2_curves)
        out[label] = {"best_mean": float(np.mean(bests)), "best_std": float(np.std(bests)),
                      "best_per_seed": [float(x) for x in bests],
                      "natural_r2_mean": float(np.nanmean(nat)), "natural_r2_std": float(np.nanstd(nat)),
                      "natural_r2_per_seed": [float(x) for x in nat],
                      "best_curve_mean": bc.mean(0).tolist(), "best_curve_std": bc.std(0).tolist(),
                      "natr2_curve_mean": np.nanmean(nc, 0).tolist(), "natr2_curve_std": np.nanstd(nc, 0).tolist()}
        print(f"  => {label}: best={np.mean(bests):.2f}±{np.std(bests):.2f} | "
              f"model-quality natR2={np.nanmean(nat):.3f}±{np.nanstd(nat):.3f}", flush=True)
    res = {"dataset": name, "pool": len(df), "n_elem": len(feat), "target": tgt,
           "direction": direction, "global_best": float(global_best),
           "n_init": n_init, "n_query": 2, "n_iter": n_iter,
           "budget": budget, "results": out}
    json.dump(res, open(_HERE/f"results/generalize_{name}.json", "w"), indent=2)
    print(f"  saved -> results/generalize_{name}.json", flush=True)
    return res

if __name__ == "__main__":
    names = sys.argv[1:] or list(DATASETS.keys())
    for nm in names:
        outp = _HERE/f"results/generalize_{nm}.json"
        if outp.exists():
            print(f"SKIP {nm} (done)"); continue
        evaluate(nm)
