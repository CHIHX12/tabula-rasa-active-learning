"""
Active learning on a CONTINUOUS compositional simplex (Referee 1, comment 2;
Editor: "all active-learning experiments are performed on finite predefined
candidate pools").

There is no fixed candidate pool here.  At every iteration a *fresh* set of
candidates is drawn from the continuous simplex, the acquisition function is
evaluated on them, and the incumbent is then refined by local perturbation, so
the query is a genuinely continuous argmin rather than a lookup in a
pre-enumerated list.  No composition is ever offered twice.

Two oracle families are used so that the conclusion does not depend on either:

  (A) 'digital twin' of the real OER landscape -- an RBF interpolant fitted to
      all 6074 measured (Ni,Fe,Co,Ce) -> J10 points.  Realistic, but the oracle
      is itself a model, and its hold-out accuracy is reported.

  (B) analytic test functions on the simplex with exact ground truth:
      'sharp'  - smooth bowl plus one narrow isolated optimum (mimics the
                 206 mV outlier), and
      'rugged' - Ackley-type multimodal landscape.

Surrogate quality is scored on a fixed hold-out sample of 5000 compositions
drawn once from the same continuous distribution and never queried -- the
continuous analogue of the out-of-pool R^2 used everywhere else.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import norm
from sklearn.metrics import r2_score

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "revision"))
from revision.surrogates import SURROGATES, MLPEnsembleSurrogate  # noqa: E402
from revision.al_harness import build_surrogate, expected_improvement  # noqa: E402

OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(exist_ok=True)


# ─────────────────────────────────────────────────────────────────
# Oracles on the continuous simplex
# ─────────────────────────────────────────────────────────────────
class DigitalTwinOracle:
    """RBF interpolant of the 6074 measured OER compositions."""

    name = "oer_twin"
    direction = "min"

    def __init__(self, smoothing=0.01, neighbors=256):
        from scipy.interpolate import RBFInterpolator
        df = pd.read_csv(ROOT / "data" / "OER_database.csv")
        df.columns = [c.strip().lstrip("﻿") for c in df.columns]
        X = df[["Ni", "Fe", "Co", "Ce"]].values.astype(np.float64)
        y = df["J10"].values.astype(np.float64)
        self.X_meas, self.y_meas = X, y
        rng = np.random.default_rng(0)
        idx = rng.permutation(len(X))
        cut = int(0.8 * len(X))
        tr, te = idx[:cut], idx[cut:]
        f_tr = RBFInterpolator(X[tr], y[tr], kernel="thin_plate_spline",
                               smoothing=smoothing, neighbors=neighbors)
        self.holdout_r2 = float(r2_score(y[te], f_tr(X[te])))
        self.f = RBFInterpolator(X, y, kernel="thin_plate_spline",
                                 smoothing=smoothing, neighbors=neighbors)

    def __call__(self, X):
        return self.f(np.atleast_2d(np.asarray(X, dtype=np.float64)))


class AnalyticOracle:
    """Exact analytic ground truth on the 3-simplex."""

    direction = "min"

    def __init__(self, kind="sharp", n_dim=4, seed=0):
        self.name = f"analytic_{kind}"
        self.kind = kind
        self.n_dim = n_dim
        rng = np.random.default_rng(seed)
        # a fixed, reproducible optimum location on the simplex
        self.x_star = rng.dirichlet(np.ones(n_dim) * 1.5)
        self.holdout_r2 = float("nan")

    def __call__(self, X):
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        d = np.linalg.norm(X - self.x_star, axis=1)
        if self.kind == "sharp":
            bowl = 400.0 + 120.0 * d
            spike = -180.0 * np.exp(-(d / 0.045) ** 2)
            ripple = 12.0 * np.sin(9.0 * X[:, 0]) * np.cos(9.0 * X[:, -1])
            return bowl + spike + ripple
        # 'rugged': Ackley-like on the simplex
        z = (X - self.x_star) * 6.0
        a = -20.0 * np.exp(-0.2 * np.sqrt(np.mean(z**2, axis=1)))
        b = -np.exp(np.mean(np.cos(2 * np.pi * z), axis=1))
        return 100.0 * (a + b + 20.0 + np.e) + 200.0


def make_oracle(name):
    if name == "oer_twin":
        return DigitalTwinOracle()
    return AnalyticOracle(kind=name.replace("analytic_", ""))


# ─────────────────────────────────────────────────────────────────
def sample_simplex(rng, n, dim=4, alpha=1.0):
    return rng.dirichlet(np.full(dim, alpha), size=n).astype(np.float32)


def run_continuous(oracle, *, protocol, surrogate="pcban", seed=0,
                   n_init=20, n_iter=100, n_cand=4000, n_refine=400,
                   beta=2.5, beta_min=0.2, gamma=8.0, delta=12.0,
                   surp_radius=0.10, surp_thresh_sigma=2.0,
                   surrogate_kwargs=None, dim=4, holdout=None):
    rules = protocol.split("+")
    rng = np.random.default_rng(seed)
    surrogate_kwargs = surrogate_kwargs or {}

    X_tr = sample_simplex(rng, n_init, dim)
    y_tr = oracle(X_tr).astype(np.float32)

    Xh, yh = holdout
    sur = build_surrogate(surrogate, **surrogate_kwargs)

    hist = dict(best=[float(y_tr.min())], holdout_r2=[], sigma=[], oob_r2=[])
    surprise_pts = []
    t0 = time.time()

    for it in range(n_iter):
        # fresh continuous candidate set -- never the same twice
        X_cand = sample_simplex(rng, n_cand, dim)
        sur.fit(X_tr, y_tr, seed=seed, iteration=it, X_pool=X_cand)
        mu_h, _ = sur.predict(Xh)
        hist["holdout_r2"].append(float(r2_score(yh, mu_h)))
        hist["oob_r2"].append(float(sur.oob_r2))

        beta_t = beta_min + (beta - beta_min) * np.cos(np.pi * it / (2 * n_iter))

        mu_c, sd_c = sur.predict(X_cand)
        hist["sigma"].append(float(sd_c.mean()))

        picks = []
        for rule in rules:
            X_c, mu, sd = X_cand, mu_c, sd_c
            if picks:                      # keep the batch from collapsing
                far = cdist(X_cand, np.asarray(picks)).min(axis=1) > 1e-3
                if far.sum() > 10:
                    X_c, mu, sd = X_cand[far], mu_c[far], sd_c[far]

            def acq(Xq, mu_q, sd_q, rule=rule, beta_t=beta_t):
                dmin = cdist(Xq, X_tr).min(axis=1)
                dn = dmin / (dmin.max() + 1e-8)
                sb = np.zeros(len(Xq))
                if surprise_pts:
                    for xs, w in surprise_pts[-10:]:
                        sb += w * np.exp(-cdist(Xq, xs.reshape(1, -1)).ravel() / surp_radius)
                    sb = sb / (sb.max() + 1e-8)
                if rule == "lcbds":
                    return mu_q - beta_t * sd_q - gamma * dn - delta * sb
                if rule == "lcb":
                    return mu_q - beta_t * sd_q
                if rule == "greedy":
                    return mu_q
                if rule == "ei":
                    return -expected_improvement(mu_q, sd_q, float(y_tr.min()), "min")
                if rule == "maxsigma":
                    return -sd_q
                if rule == "random":
                    return rng.random(len(Xq))
                raise ValueError(rule)

            s = acq(X_c, mu, sd)
            best_i = int(np.argmin(s))
            x_best = X_c[best_i].copy()

            # continuous local refinement around the coarse argmin
            if rule != "random" and n_refine > 0:
                for scale in (0.05, 0.015):
                    pert = x_best[None] + rng.normal(0, scale, size=(n_refine, dim))
                    pert = np.clip(pert, 0.0, None)
                    pert = (pert / pert.sum(1, keepdims=True)).astype(np.float32)
                    mu_p, sd_p = sur.predict(pert)
                    s_p = acq(pert, mu_p, sd_p)
                    if s_p.min() < s[best_i]:
                        j = int(np.argmin(s_p))
                        x_best = pert[j].copy()
                        s = s_p
                        best_i = j
            picks.append(x_best)

        y_scale = float(y_tr.std()) + 1e-8
        mu_pick, _ = sur.predict(np.asarray(picks, dtype=np.float32))
        y_new = oracle(np.asarray(picks, dtype=np.float64)).astype(np.float32)
        for k, xq in enumerate(picks):
            X_tr = np.vstack([X_tr, xq.astype(np.float32)])
            y_tr = np.append(y_tr, y_new[k])
            if abs(float(y_new[k]) - float(mu_pick[k])) > surp_thresh_sigma * y_scale:
                surprise_pts.append((xq.copy(), abs(float(y_new[k]) - float(mu_pick[k]))))
        hist["best"].append(float(y_tr.min()))

        if (it + 1) % 25 == 0:
            print(f"    it{it+1:4d} n={len(y_tr):4d} best={hist['best'][-1]:.2f} "
                  f"holdoutR2={hist['holdout_r2'][-1]:.3f}", flush=True)

    X_c = sample_simplex(rng, n_cand, dim)
    sur.fit(X_tr, y_tr, seed=seed, iteration=n_iter, X_pool=X_c)
    mu_h, _ = sur.predict(Xh)
    return dict(seed=int(seed), protocol=protocol, surrogate=surrogate,
                n_labeled=int(len(y_tr)),
                best_overall=float(y_tr.min()),
                final_holdout_r2=float(r2_score(yh, mu_h)),
                seconds=float(time.time() - t0), history=hist)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oracle", default="oer_twin",
                    choices=["oer_twin", "analytic_sharp", "analytic_rugged"])
    ap.add_argument("--protocol", required=True)
    ap.add_argument("--surrogate", default="pcban")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--n-iter", type=int, default=100)
    ap.add_argument("--n-init", type=int, default=20)
    ap.add_argument("--gamma", type=float, default=8.0)
    ap.add_argument("--delta", type=float, default=12.0)
    args = ap.parse_args()

    out_path = OUT / f"{args.tag}.json"
    if out_path.exists():
        print(f"SKIP {out_path}")
        return

    oracle = make_oracle(args.oracle)
    rng_h = np.random.default_rng(12345)
    Xh = sample_simplex(rng_h, 5000, 4)
    yh = oracle(Xh).astype(np.float32)
    print(f"[{args.tag}] oracle={oracle.name} holdout_r2_of_oracle={oracle.holdout_r2} "
          f"y range on holdout: {yh.min():.1f}..{yh.max():.1f}", flush=True)

    runs = []
    for s in [int(x) for x in args.seeds.split(",")]:
        r = run_continuous(oracle, protocol=args.protocol, surrogate=args.surrogate,
                           seed=s, n_init=args.n_init, n_iter=args.n_iter,
                           gamma=args.gamma, delta=args.delta,
                           holdout=(Xh, yh))
        runs.append(r)
        print(f"  seed {s}: best={r['best_overall']:.2f} "
              f"holdoutR2={r['final_holdout_r2']:.3f} ({r['seconds']:.0f}s)", flush=True)

    json.dump(dict(tag=args.tag, oracle=oracle.name,
                   oracle_holdout_r2=oracle.holdout_r2,
                   protocol=args.protocol, surrogate=args.surrogate,
                   n_init=args.n_init, n_iter=args.n_iter,
                   budget=args.n_init + args.n_iter * len(args.protocol.split("+")),
                   runs=runs), open(out_path, "w"))
    b = [r["best_overall"] for r in runs]
    h = [r["final_holdout_r2"] for r in runs]
    print(f"[{args.tag}] DONE best={np.mean(b):.2f}+-{np.std(b):.2f} "
          f"holdoutR2={np.mean(h):.3f}+-{np.std(h):.3f}", flush=True)


if __name__ == "__main__":
    main()
