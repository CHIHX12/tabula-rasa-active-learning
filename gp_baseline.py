"""
Standard Gaussian-Process Bayesian Optimization (GP-BO) baseline.
Matern-5/2 kernel + Expected Improvement acquisition (the canonical BO baseline).
Run on the OER benchmark (10 seeds) and the five generalization datasets (5 seeds),
using the SAME budgets as the main framework, so GP-BO can be added as a fourth
strategy in the comparisons. Deterministic given seed (reproducible).
"""
import json, sys, time, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score
from scipy.stats import norm
from scipy.spatial.distance import cdist

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import generalize_multi as G   # reuse the 5 dataset specs

def load_xy(spec):
    csv, sep, feat, tgt, direction, n_init, n_iter = spec
    df = pd.read_csv(csv, sep=sep); df.columns = [c.strip() for c in df.columns]
    feat = [f.strip() for f in feat]
    Xr = df[feat].values.astype(np.float64)
    X = ((Xr - Xr.min(0)) / (Xr.max(0) - Xr.min(0) + 1e-9)).astype(np.float64)
    y = df[tgt].values.astype(np.float64)
    return X, y, direction, n_init, n_iter

def load_oer():
    df = pd.read_csv(HERE / "data" / "OER_database.csv")
    feat = ["Ni", "Fe", "Co", "Ce"]; Xr = df[feat].values.astype(np.float64)
    X = ((Xr - Xr.min(0)) / (Xr.max(0) - Xr.min(0) + 1e-9))
    y = df["J10"].values.astype(np.float64)
    return X, y, "min", 20, 100   # n_query fixed to 2 below

def ei(mu, sd, best, direction):
    sd = np.maximum(sd, 1e-9)
    imp = (best - mu) if direction == "min" else (mu - best)
    z = imp / sd
    return imp * norm.cdf(z) + sd * norm.pdf(z)

def run_gp(X, y, direction, n_init, n_iter, n_query, seed):
    N = len(X); D = X.shape[1]
    km = KMeans(n_clusters=n_init, random_state=seed, n_init=10).fit(X)
    labeled = set()
    for c in range(n_init):
        idx = np.where(km.labels_ == c)[0]
        d = cdist(X[idx], km.cluster_centers_[c:c+1]).ravel()
        labeled.add(int(idx[d.argmin()]))
    def cur_best():
        v = y[list(labeled)]; return float(v.min() if direction == "min" else v.max())
    best_hist = [cur_best()]; natr2_hist = []
    for it in range(n_iter):
        L = np.array(sorted(labeled)); ytr = y[L]
        ym, ys = ytr.mean(), ytr.std() + 1e-9
        kernel = ConstantKernel(1.0) * Matern(nu=2.5, length_scale=np.ones(D)) + WhiteKernel(1e-2)
        gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, n_restarts_optimizer=2, normalize_y=False)
        gp.fit(X[L], (ytr - ym) / ys)
        mu, sd = gp.predict(X, return_std=True); mu = mu * ys + ym; sd = sd * ys
        av = np.array([i for i in range(N) if i not in labeled])
        natr2_hist.append(float(r2_score(y[av], mu[av])) if len(av) >= 5 else float("nan"))
        best = cur_best()
        for q in range(n_query):
            av = np.array([i for i in range(N) if i not in labeled])
            a = ei(mu[av], sd[av], best, direction)
            labeled.add(int(av[int(a.argmax())]))
        best_hist.append(cur_best())
    return best_hist, natr2_hist

def evaluate(name, X, y, direction, n_init, n_iter, n_query, seeds):
    bests = []; nats = []; bcurves = []; ncurves = []
    gb = y.min() if direction == "min" else y.max()
    for s in seeds:
        t = time.time()
        bh, nh = run_gp(X, y, direction, n_init, n_iter, n_query, s)
        b = min(bh) if direction == "min" else max(bh)
        bests.append(b); nats.append(nh[-1]); bcurves.append(bh); ncurves.append(nh)
        print(f"  [GP-BO {name}] seed{s}: best={b:.2f} natR2={nh[-1]:.3f} {time.time()-t:.0f}s", flush=True)
    bc = np.array(bcurves); nc = np.array(ncurves)
    out = {"dataset": name, "direction": direction, "global_best": float(gb),
           "n_init": n_init, "n_query": n_query, "n_iter": n_iter,
           "best_mean": float(np.mean(bests)), "best_std": float(np.std(bests)),
           "best_per_seed": [float(x) for x in bests],
           "natural_r2_mean": float(np.nanmean(nats)), "natural_r2_per_seed": [float(x) for x in nats],
           "best_curve_mean": bc.mean(0).tolist(), "best_curve_std": bc.std(0).tolist(),
           "natr2_curve_mean": np.nanmean(nc, 0).tolist(), "natr2_curve_std": np.nanstd(nc, 0).tolist()}
    json.dump(out, open(HERE / f"results/gp_{name}.json", "w"), indent=2)
    print(f"  => GP-BO {name}: best {np.mean(bests):.2f}±{np.std(bests):.2f} | natR2 {np.nanmean(nats):.3f}\n", flush=True)

if __name__ == "__main__":
    # OER: 10 seeds (0-4,10-14), budget 220 (n_query=2)
    Xo, yo, d, ni, nit = load_oer()
    print("### GP-BO on OER (10 seeds, budget 220) ###", flush=True)
    evaluate("oer", Xo, yo, d, ni, nit, 2, [0,1,2,3,4,10,11,12,13,14])
    # 5 generalization datasets: 5 seeds
    for nm, spec in G.DATASETS.items():
        X, y, dr, ni, nit = load_xy(spec)
        print(f"### GP-BO on {nm} ###", flush=True)
        evaluate(nm, X, y, dr, ni, nit, 2, [0,1,2,3,4])
    print("GP-BO ALL DONE", flush=True)
