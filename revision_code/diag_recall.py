"""Does the surrogate carry region-level signal even though it cannot predict
the cliff at a point?

If the 17 sub-370 mV compositions are ranked highly by the model even when it
badly mis-predicts each one's value, then a policy that sweeps the model's top
list will find them, and the bottleneck is not the model.  If they are ranked
at random, the model contributes nothing to the search and only coverage does.
"""
import sys, numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from scipy.spatial.distance import cdist
from al_harness import build_surrogate, kmeans_init

df = pd.read_csv('data/OER_database.csv'); df.columns=[c.strip().lstrip('﻿') for c in df.columns]
X = df[['Ni','Fe','Co','Ce']].values.astype(np.float32)
y = df['J10'].values.astype(np.float32)
good = np.where(y < 370)[0]
opt  = int(np.argmin(y))
print(f"pool {len(X)},  {len(good)} points below 370 mV,  optimum {y[opt]:.0f} at idx {opt}")
print(f"good points J10 = {sorted(int(v) for v in y[good])}\n")

# where are they?
print("composition of the sub-370 set (mean +- sd):")
print(f"   Ni {X[good,0].mean():.3f}+-{X[good,0].std():.3f}   pool {X[:,0].mean():.3f}")
print(f"   Fe {X[good,1].mean():.3f}+-{X[good,1].std():.3f}   pool {X[:,1].mean():.3f}")
print(f"   Co {X[good,2].mean():.3f}+-{X[good,2].std():.3f}   pool {X[:,2].mean():.3f}")
print(f"   Ce {X[good,3].mean():.3f}+-{X[good,3].std():.3f}   pool {X[:,3].mean():.3f}")
D = cdist(X[good], X[good])
print(f"   pairwise distance among the good points: median {np.median(D[D>0]):.3f}")
print(f"   pool-wide median pairwise distance      : {np.median(cdist(X[::40],X[::40])):.3f}\n")

rng = np.random.default_rng(0)
for n_lab in (100, 220):
    idx = kmeans_init(X, 20, 0)
    extra = rng.choice([i for i in range(len(X)) if i not in set(idx)],
                       size=n_lab-20, replace=False)
    lab = np.concatenate([idx, extra]).astype(int)
    n_good_lab = len(set(lab) & set(good.tolist()))
    unl = np.array([i for i in range(len(X)) if i not in set(lab)])
    rem = [g for g in good if g not in set(lab)]
    print(f"--- {n_lab} random labelled points ({n_good_lab} of them good) ---")
    print(f"{'surrogate':10s} {'recall@220':>11s} {'recall@500':>11s} {'median rank of good':>21s} {'rank of 206':>12s}")
    for name in ['pcban','rf','gp']:
        s = build_surrogate(name, sigma_calib=True)
        s.fit(X[lab], y[lab], seed=0, iteration=0, X_pool=X[unl])
        mu, sd = s.predict(X[unl])
        order = unl[np.argsort(mu)]
        pos = {v: i for i, v in enumerate(order)}
        ranks = np.array([pos[g] for g in rem])
        r220 = (ranks < 220).sum() / max(1, len(rem))
        r500 = (ranks < 500).sum() / max(1, len(rem))
        ropt = pos[opt] + 1 if opt in pos else -1
        print(f"{name:10s} {r220:>10.2f} {r500:>11.2f} {int(np.median(ranks)):>16d}/{len(unl):<5d} {ropt:>12d}")
    # what a random shortlist would get
    print(f"{'random':10s} {220/len(unl):>10.2f} {500/len(unl):>11.2f}")
    print()
