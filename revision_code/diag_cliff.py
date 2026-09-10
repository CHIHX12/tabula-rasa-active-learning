"""Can the surrogate represent the cliff at all?

Label the six nearest neighbours of the global optimum (which are 379-425 mV)
plus a random background, WITHOUT labelling the optimum itself, and ask what
each surrogate predicts at the optimum.  If mu is pulled to the neighbours'
level and sigma stays small, then no acquisition built on mu and sigma can
select it, and the bottleneck is the model class rather than the acquisition.
"""
import sys, numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from scipy.spatial.distance import cdist
from al_harness import build_surrogate

df = pd.read_csv('data/OER_database.csv'); df.columns=[c.strip().lstrip('﻿') for c in df.columns]
X = df[['Ni','Fe','Co','Ce']].values.astype(np.float32)
y = df['J10'].values.astype(np.float32)
opt = int(np.argmin(y))
D = cdist(X, X); np.fill_diagonal(D, np.inf)
nbrs = np.argsort(D[opt])[:12]

print(f"optimum idx {opt}, J10 = {y[opt]:.0f}")
print(f"its 12 nearest neighbours: J10 = {[int(v) for v in y[nbrs]]}\n")

rng = np.random.default_rng(0)
for n_bg in (200, 400):
    bg = rng.choice([i for i in range(len(X)) if i != opt and i not in set(nbrs)],
                    size=n_bg, replace=False)
    lab = np.concatenate([nbrs, bg])
    print(f"--- labelled set: 12 neighbours + {n_bg} random background = {len(lab)} points ---")
    print(f"{'surrogate':16s} {'mu at optimum':>16s} {'sigma':>9s} {'true':>7s} "
          f"{'error':>8s} {'rank of optimum':>17s}")
    for name in ['pcban','rf','gp','mlp']:
        s = build_surrogate(name, sigma_calib=True)
        s.fit(X[lab], y[lab], seed=0, iteration=0, X_pool=X)
        mu, sd = s.predict(X)
        unl = np.array([i for i in range(len(X)) if i not in set(lab)])
        order = unl[np.argsort(mu[unl])]
        rank = int(np.where(order == opt)[0][0]) + 1
        print(f"{name:16s} {mu[opt]:>16.1f} {sd[opt]:>9.2f} {y[opt]:>7.0f} "
              f"{mu[opt]-y[opt]:>+8.1f} {rank:>10d}/{len(unl):<6d}")
    print()
