"""Measure the actual magnitude of each term in the LCBDS acquisition.

    a(x) = mu(x) - beta_t*sigma(x) - gamma*d_norm(x) - delta*s(x)

If the exploitation term mu spans far more than the other three combined, then
gamma, delta and the beta schedule cannot change which candidate is selected --
which would explain why every ablation of them comes out null.
"""
import sys, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from scipy.spatial.distance import cdist
from al_harness import kmeans_init, build_surrogate, load_dataset

X, y, direction, n_init, _ = load_dataset("oer")
seed = 0
BETA, BETA_MIN, GAMMA, DELTA = 2.5, 0.2, 8.0, 12.0
N_ITER = 100
rng = np.random.default_rng(seed)
idx = kmeans_init(X, 20, seed)
labeled = set(idx); X_tr, y_tr = X[idx].copy(), y[idx].copy()
sur = build_surrogate("pcban")
surprise = []
print(f"{'iter':>5} {'n':>5} | {'mu spread':>28} | {'beta*sigma':>16} "
      f"{'gamma*d':>10} {'delta*s':>10} | {'rank chg':>9}")
for it in range(N_ITER):
    unl = np.array(sorted(set(range(len(X))) - labeled))
    Xp = X[unl]
    sur.fit(X_tr, y_tr, seed=seed, iteration=it, X_pool=Xp)
    mu, sd = sur.predict(Xp)
    beta_t = BETA_MIN + (BETA - BETA_MIN) * np.cos(np.pi * it / (2 * N_ITER))
    dmin = cdist(Xp, X_tr).min(axis=1); dn = dmin / (dmin.max() + 1e-8)
    sb = np.zeros(len(Xp))
    if surprise:
        for xs, w in surprise[-10:]:
            sb += w * np.exp(-cdist(Xp, xs.reshape(1, -1)).ravel() / 0.10)
        sb = sb / (sb.max() + 1e-8)
    t_mu, t_sig, t_g, t_d = mu, beta_t*sd, GAMMA*dn, DELTA*sb
    score_full = t_mu - t_sig - t_g - t_d
    score_mu   = t_mu
    # how much do the three extra terms change the ranking of the top candidates?
    top_full = set(np.argsort(score_full)[:20]); top_mu = set(np.argsort(score_mu)[:20])
    overlap = len(top_full & top_mu)
    if it % 10 == 0 or it == N_ITER-1:
        print(f"{it:>5} {len(y_tr):>5} | mu {mu.min():7.1f}..{mu.max():7.1f} "
              f"(sd {mu.std():6.2f}) | {t_sig.mean():7.2f} (max {t_sig.max():5.2f}) "
              f"{t_g.mean():9.2f} {t_d.mean():9.2f} | {20-overlap:>4d}/20")
    # advance one iteration with the published protocol
    order = np.argsort(score_full); pickA = int(order[0])
    pickB = int(np.argsort(-sd)[0])
    if pickB == pickA: pickB = int(np.argsort(-sd)[1])
    ysc = float(y_tr.std()) + 1e-8
    for p in (pickA, pickB):
        g = int(unl[p]); labeled.add(g)
        X_tr = np.vstack([X_tr, X[g]]); y_tr = np.append(y_tr, y[g])
        if abs(float(y[g]) - float(mu[p])) > 2.0*ysc:
            surprise.append((X[g].copy(), abs(float(y[g]) - float(mu[p]))))
