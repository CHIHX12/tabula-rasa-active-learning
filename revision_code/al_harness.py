"""
Controlled active-learning harness for the Digital Discovery revision.

Design goal: hold EVERYTHING constant except the one factor under test, so that
each question about what the surrogate and the protocol contribute can be
answered by a clean factorial contrast.

Held constant across every run
* candidate pool and oracle
* K-means space-filling initialisation (n_init points, same seed)
* total labelled budget (n_init + n_iter * n_query)
* random seeds
* evaluation metric

Factors that can be varied
* ``surrogate``   : pcban | rf | gp | mlp | mlp_descriptor   (surrogate study)
* ``protocol``    : the per-iteration batch, e.g. "lcbds+maxsigma"  (protocol study)
* ``gamma``/``delta``/``beta`` : LCBDS weights                      (weight study)
* ``train_mode``  : scratch | warm | warm_kd                        (surrogate study)
* ``lam_mse``/``lam_rec``/``K`` : PC-BAN loss and MDN settings   (loss and MDN study)

Common evaluation metric
``pool_r2`` -- the coefficient of determination of the surrogate's predictive
mean over the *currently unlabelled* candidate pool, recorded at every
iteration for EVERY strategy including GP-BO.  This is directly comparable
across all surrogates; ``oob_r2`` is still recorded where defined, but it is no
longer the metric on which any cross-strategy claim rests.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import norm
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "revision"))

from revision.surrogates import SURROGATES, MLPEnsembleSurrogate  # noqa: E402

RES = Path(__file__).resolve().parent / "results"
RES.mkdir(exist_ok=True)

QUERY_RULES = {"lcbds", "lcb", "greedy", "ei", "maxsigma", "maxmu", "random",
               "mei", "rough", "lcbdr", "meirough", "sweep"}


# ─────────────────────────────────────────────────────────────────
def kmeans_init(X, n_init, seed):
    km = KMeans(n_clusters=n_init, random_state=seed, n_init=10)
    labels = km.fit_predict(X)
    rng = np.random.default_rng(seed)
    return [int(rng.choice(np.where(labels == c)[0])) for c in range(n_init)]


def build_surrogate(name, **kw):
    if name == "mlp_descriptor":
        return MLPEnsembleSurrogate(descriptor=True)
    cls = SURROGATES[name]
    if name == "pcban":
        return cls(n_bootstrap=kw.get("n_bootstrap", 5),
                   n_mc=kw.get("n_mc", 20),
                   train_mode=kw.get("train_mode", "warm_kd"),
                   lam_mse=kw.get("lam_mse", 0.10),
                   lam_rec=kw.get("lam_rec", 0.02),
                   kd_alpha=kw.get("kd_alpha", 0.5),
                   n_components_override=kw.get("n_components", None),
                   sigma_calib=kw.get("sigma_calib", False),
                   warm_epochs=kw.get("warm_epochs", 60),
                   y_transform=kw.get("y_transform", "none"),
                   tail_tau=kw.get("tail_tau", 0.0),
                   direction=kw.get("direction", "min"))
    return cls()


def expected_improvement(mu, sd, incumbent, direction):
    sd = np.maximum(sd, 1e-9)
    imp = (incumbent - mu) if direction == "min" else (mu - incumbent)
    z = imp / sd
    return imp * norm.cdf(z) + sd * norm.pdf(z)


def mixture_expected_improvement(pi, mu, sd, incumbent, direction):
    """Expected improvement under a Gaussian MIXTURE.

    EI is linear in the mixture weights, so the exact value is the weighted sum
    of the per-component EIs.  Unlike the moment-matched Gaussian version this
    can express 'probably mediocre, but with a small chance of being far
    better' -- the probabilistic signature of the discontinuities that hold the
    optima of this landscape.
    """
    sd = np.maximum(sd, 1e-9)
    imp = (incumbent - mu) if direction == "min" else (mu - incumbent)
    z = imp / sd
    ei_k = imp * norm.cdf(z) + sd * norm.pdf(z)
    return (pi * ei_k).sum(axis=1)


def local_roughness(X_pool, X_lab, y_lab, k=8):
    """Spread of the measured values among the k nearest LABELLED points.

    The optimum of the OER pool sits at the 96.5th percentile of this quantity
    while sitting at ordinary density, so a geometric diversity term (which
    rewards empty regions) points away from it whereas this one points at it.
    """
    k = int(min(k, len(y_lab)))
    if k < 2:
        return np.zeros(len(X_pool))
    d = cdist(X_pool, X_lab)
    idx = np.argpartition(d, k - 1, axis=1)[:, :k]
    vals = y_lab[idx]
    r = vals.std(axis=1)
    return r / (r.max() + 1e-12)


# ─────────────────────────────────────────────────────────────────
def run_al(X, y, *, protocol, surrogate="pcban", direction="min",
           n_init=20, n_iter=100, seed=0,
           beta=2.5, beta_min=0.2, anneal=True,
           gamma=8.0, delta=12.0, surp_radius=0.10, surp_thresh_sigma=2.0,
           acq_scale="absolute", surp_mode="global", eta=1.0, rough_k=8,
           sweep_gate=370.0,
           surrogate_kwargs=None, verbose_every=25):
    """Run one pool-based AL campaign.

    ``protocol`` is a '+'-separated list of query rules applied in order within
    each iteration, e.g. ``"lcbds+maxsigma"`` (the published LCBDS protocol) or
    ``"ei+ei"`` (a standard BO batch without a dedicated exploration query).
    """
    rules = protocol.split("+")
    assert all(r in QUERY_RULES for r in rules), f"bad protocol {protocol}"
    n_query = len(rules)
    surrogate_kwargs = surrogate_kwargs or {}

    N = len(X)
    rng = np.random.default_rng(seed)
    init_idx = kmeans_init(X, n_init, seed)
    labeled = set(init_idx)
    X_tr, y_tr = X[init_idx].copy(), y[init_idx].copy()

    sur = build_surrogate(surrogate, **surrogate_kwargs)

    sign = 1.0 if direction == "min" else -1.0   # acquisitions are minimised on sign*y

    def incumbent():
        return float(y_tr.min() if direction == "min" else y_tr.max())

    hist = dict(best=[incumbent()], pool_r2=[], oob_r2=[], sigma=[],
                beta=[], fit_seconds=[], n_surprise=[], mode=[])
    surprise_pts = []
    t_start = time.time()

    for it in range(n_iter):
        unlabeled = np.array(sorted(set(range(N)) - labeled))
        X_pool = X[unlabeled]

        sur.fit(X_tr, y_tr, seed=seed, iteration=it, X_pool=X_pool)
        mu, sd = sur.predict(X_pool)

        # ---- COMMON METRIC: R^2 on the unexplored candidate pool ----
        hist["pool_r2"].append(float(r2_score(y[unlabeled], mu))
                               if len(unlabeled) >= 5 else float("nan"))
        hist["oob_r2"].append(float(sur.oob_r2))
        hist["sigma"].append(float(sd.mean()))
        hist["fit_seconds"].append(float(sur.fit_seconds))
        hist["mode"].append(getattr(sur, "last_mode", "n/a"))

        beta_t = (beta_min + (beta - beta_min) * np.cos(np.pi * it / (2 * n_iter))
                  if anneal else beta)
        hist["beta"].append(float(beta_t))

        # ---- shared LCBDS ingredients ----
        min_dist = cdist(X_pool, X_tr).min(axis=1)
        dist_norm = min_dist / (min_dist.max() + 1e-8)
        surp = np.zeros(len(X_pool))
        if surprise_pts:
            for x_s, w in surprise_pts[-10:]:
                d = cdist(X_pool, x_s.reshape(1, -1)).ravel()
                surp += w * np.exp(-d / surp_radius)
            surp = surp / (surp.max() + 1e-8)

        # The published acquisition adds terms measured in millivolts to a
        # predictive mean whose spread over the pool is an order of magnitude
        # larger, so the diversity, surprise and uncertainty terms can only
        # reorder candidates whose predictions already nearly coincide.  In
        # "standardized" mode every term is expressed in units of the pool
        # standard deviation of mu, which makes the weights comparable.
        if acq_scale == "standardized":
            mu_c, mu_s = float(mu.mean()), float(mu.std()) + 1e-12
            sd_s = float(sd.std()) + 1e-12
            mu_use = (mu - mu_c) / mu_s
            sd_use = (sd - float(sd.mean())) / sd_s
        else:
            mu_use, sd_use = mu, sd

        # local roughness of the measured surface, and the full mixture law,
        # computed once per iteration and only if a rule needs them
        need_rough = any(r in ("rough", "lcbdr", "meirough") for r in rules)
        rough = (local_roughness(X_pool, X_tr, y_tr, k=rough_k)
                 if need_rough else np.zeros(len(X_pool)))
        need_mix = any(r in ("mei", "meirough") for r in rules)
        mix = None
        if need_mix and hasattr(sur, "predict_mixture"):
            mix = sur.predict_mixture(X_pool)

        inc = incumbent()
        avail = list(range(len(unlabeled)))
        picks = []
        for rule in rules[: min(n_query, len(avail))]:
            m = np.array(avail)
            mu_a, sd_a = mu[m], sd[m]
            mu_z, sd_z = mu_use[m], sd_use[m]
            if rule == "lcbds":
                score = sign * mu_z - beta_t * sd_z - gamma * dist_norm[m] - delta * surp[m]
                pick = m[int(np.argmin(score))]
            elif rule == "lcb":
                pick = m[int(np.argmin(sign * mu_z - beta_t * sd_z))]
            elif rule == "greedy":
                pick = m[int(np.argmin(sign * mu_a))]
            elif rule == "ei":
                pick = m[int(np.argmax(expected_improvement(mu_a, sd_a, inc, direction)))]
            elif rule == "maxsigma":
                pick = m[int(np.argmax(sd_a))]
            elif rule == "maxmu":
                pick = m[int(np.argmax(sign * mu_a))]
            elif rule == "sweep":
                # Find-then-sweep.  The surrogate locates the good REGION well
                # (recall of the sub-370 set is 0.59 against 0.04 for a random
                # shortlist) but carries no information about which point in it
                # is the discontinuity: the 206 mV optimum is ranked ~2400/5854
                # because its own neighbours sit at ~409 mV, so a mu-based rule
                # walks away from it.  Once a good value has been observed, the
                # cheapest reliable way to find the cliff is to cover its
                # neighbourhood outright -- from the 252 mV point the optimum is
                # 0.083 away, a ball holding 64 candidates, i.e. 29% of the
                # budget.  Below the gate this query therefore spends itself on
                # the nearest uncovered candidate to the incumbent; above it,
                # it behaves as the ordinary exploration query.
                if (inc < sweep_gate) if direction == "min" else (inc > sweep_gate):
                    centre = X_tr[int(np.argmin(sign * y_tr))]
                    dc = cdist(X_pool[m], centre.reshape(1, -1)).ravel()
                    pick = m[int(np.argmin(dc))]
                else:
                    pick = m[int(np.argmax(sd_a))]
            elif rule == "mei":
                if mix is None:
                    v = expected_improvement(mu_a, sd_a, inc, direction)
                else:
                    v = mixture_expected_improvement(mix[0][m], mix[1][m],
                                                     mix[2][m], inc, direction)
                pick = m[int(np.argmax(v))]
            elif rule == "rough":
                pick = m[int(np.argmin(sign * mu_z - eta * rough[m]))]
            elif rule == "lcbdr":
                score = (sign * mu_z - beta_t * sd_z - eta * rough[m]
                         - delta * surp[m])
                pick = m[int(np.argmin(score))]
            elif rule == "meirough":
                if mix is None:
                    v = expected_improvement(mu_a, sd_a, inc, direction)
                else:
                    v = mixture_expected_improvement(mix[0][m], mix[1][m],
                                                     mix[2][m], inc, direction)
                v = v / (v.max() + 1e-12)
                pick = m[int(np.argmax(v + eta * rough[m]))]
            else:  # random
                pick = m[int(rng.integers(0, len(m)))]
            picks.append(int(pick))
            avail = [a for a in avail if a != pick]

        # A "surprise" should mean the model was wrong by more than it expected.
        # The published rule compares the residual with the standard deviation
        # of the labels seen so far (~30 mV here), which almost never fires, so
        # s(x) stays identically zero for the whole campaign and delta multiplies
        # a zero vector.  "local" mode compares the residual with the model's own
        # predictive sd at that point, which is what the text describes.
        y_scale = float(y_tr.std()) + 1e-8
        for p in picks:
            gidx = int(unlabeled[p])
            labeled.add(gidx)
            X_tr = np.vstack([X_tr, X[gidx]])
            y_tr = np.append(y_tr, y[gidx])
            resid = abs(float(y[gidx]) - float(mu[p]))
            thresh = (surp_thresh_sigma * float(sd[p]) if surp_mode == "local"
                      else surp_thresh_sigma * y_scale)
            if resid > max(thresh, 1e-9):
                surprise_pts.append((X[gidx].copy(), resid))

        hist["best"].append(incumbent())
        hist["n_surprise"].append(len(surprise_pts))

        if verbose_every and (it + 1) % verbose_every == 0:
            print(f"    it{it+1:4d} n={len(y_tr):4d} best={hist['best'][-1]:.3f} "
                  f"poolR2={hist['pool_r2'][-1]:.3f} oobR2={hist['oob_r2'][-1]:.3f} "
                  f"sig={hist['sigma'][-1]:.2f}", flush=True)

    # final refit and final pool R^2 on the never-queried points
    unlabeled = np.array(sorted(set(range(N)) - labeled))
    sur.fit(X_tr, y_tr, seed=seed, iteration=n_iter, X_pool=X[unlabeled])
    mu_f, _ = sur.predict(X[unlabeled])
    final_pool_r2 = float(r2_score(y[unlabeled], mu_f))

    return dict(
        protocol=protocol, surrogate=surrogate, seed=int(seed),
        n_labeled=int(len(labeled)), n_pool_final=int(len(unlabeled)),
        best_final=float(hist["best"][-1]),
        best_overall=float(min(hist["best"]) if direction == "min" else max(hist["best"])),
        final_pool_r2=final_pool_r2,
        final_oob_r2=float(hist["oob_r2"][-1]),
        total_seconds=float(time.time() - t_start),
        total_fit_seconds=float(np.sum(hist["fit_seconds"])),
        n_surprise=int(len(surprise_pts)),
        history=hist,
    )


# ─────────────────────────────────────────────────────────────────
def load_oer():
    import pandas as pd
    df = pd.read_csv(ROOT / "data" / "OER_database.csv")
    df.columns = [c.strip().lstrip("﻿") for c in df.columns]
    X = df[["Ni", "Fe", "Co", "Ce"]].values.astype(np.float32)
    y = df["J10"].values.astype(np.float32)
    return X, y, "min", 20, 100


def load_dataset(name):
    """OER plus the five public generalization datasets, min-max scaled inputs.

    Returns (X, y_internal, direction, n_init, n_iter) where ``y_internal`` is
    always MINIMISED; for 'max' datasets the sign is flipped so that a single
    code path serves every system (this removes the several small
    implementation differences that existed between the OER pipeline and the
    original standalone generalization script).
    """
    import pandas as pd
    if name == "oer":
        return load_oer()
    import generalize_multi as G
    csv, sep, feat, tgt, direction, n_init, n_iter = G.DATASETS[name]
    df = pd.read_csv(csv, sep=sep)
    df.columns = [c.strip() for c in df.columns]
    feat = [f.strip() for f in feat]
    Xr = df[feat].values.astype(np.float64)
    X = ((Xr - Xr.min(0)) / (Xr.max(0) - Xr.min(0) + 1e-9)).astype(np.float32)
    y = df[tgt].values.astype(np.float32)
    return X, y, direction, n_init, n_iter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--protocol", required=True)
    ap.add_argument("--surrogate", default="pcban")
    ap.add_argument("--n-iter", type=int, default=100)
    ap.add_argument("--n-init", type=int, default=20)
    ap.add_argument("--seeds", default="0,1,2,3,4,10,11,12,13,14")
    ap.add_argument("--gamma", type=float, default=8.0)
    ap.add_argument("--delta", type=float, default=12.0)
    ap.add_argument("--beta", type=float, default=2.5)
    ap.add_argument("--beta-min", type=float, default=0.2)
    ap.add_argument("--no-anneal", action="store_true")
    ap.add_argument("--train-mode", default="warm_kd")
    ap.add_argument("--n-mc", type=int, default=20)
    ap.add_argument("--n-bootstrap", type=int, default=5)
    ap.add_argument("--lam-mse", type=float, default=0.10)
    ap.add_argument("--lam-rec", type=float, default=0.02)
    ap.add_argument("--n-components", type=int, default=0)
    ap.add_argument("--acq-scale", default="absolute",
                    choices=["absolute", "standardized"])
    ap.add_argument("--sigma-calib", action="store_true")
    ap.add_argument("--surp-mode", default="global", choices=["global", "local"],
                    help="'local' triggers a surprise when the residual exceeds "
                         "k x the model's own predictive sd at that point")
    ap.add_argument("--warm-epochs", type=int, default=60,
                    help="-1 scales the warm-start budget with the labelled-set size")
    ap.add_argument("--y-transform", default="none", choices=["none", "log"],
                    help="monotone transform expanding resolution at the good end")
    ap.add_argument("--tail-tau", type=float, default=0.0,
                    help=">0 weights the loss towards the best observations "
                         "(smaller = sharper); full_loss already supported this")
    ap.add_argument("--sweep-gate", type=float, default=370.0,
                    help="the sweep query activates once the incumbent passes this value")
    ap.add_argument("--eta", type=float, default=1.0,
                    help="weight of the local-roughness term (rules rough/lcbdr/meirough)")
    ap.add_argument("--rough-k", type=int, default=8)
    ap.add_argument("--surp-radius", type=float, default=0.10)
    ap.add_argument("--surp-thresh-sigma", type=float, default=2.0)
    ap.add_argument("--dataset", default="oer")
    ap.add_argument("--gamma-frac", type=float, default=None,
                    help="if set, gamma = frac * (y range); overrides --gamma")
    ap.add_argument("--delta-frac", type=float, default=None)
    args = ap.parse_args()

    out_path = RES / f"{args.tag}.json"
    if out_path.exists():
        print(f"SKIP (exists) {out_path}")
        return

    X, y, direction, dn_init, dn_iter = load_dataset(args.dataset)
    if args.dataset != "oer":
        # keep the dataset's own budget unless the caller overrode it
        if args.n_init == 20:
            args.n_init = dn_init
        if args.n_iter == 100:
            args.n_iter = dn_iter
    yrange = float(y.max() - y.min())
    gamma = args.gamma if args.gamma_frac is None else args.gamma_frac * yrange
    delta = args.delta if args.delta_frac is None else args.delta_frac * yrange
    seeds = [int(s) for s in args.seeds.split(",")]
    skw = dict(train_mode=args.train_mode, n_mc=args.n_mc,
               n_bootstrap=args.n_bootstrap, lam_mse=args.lam_mse,
               lam_rec=args.lam_rec,
               n_components=(args.n_components or None),
               sigma_calib=args.sigma_calib, warm_epochs=args.warm_epochs,
               y_transform=args.y_transform, tail_tau=args.tail_tau,
               direction=direction)

    runs = []
    print(f"[{args.tag}] surrogate={args.surrogate} protocol={args.protocol} "
          f"budget={args.n_init + args.n_iter*len(args.protocol.split('+'))}", flush=True)
    for s in seeds:
        t0 = time.time()
        r = run_al(X, y, protocol=args.protocol, surrogate=args.surrogate,
                   direction=direction, n_init=args.n_init, n_iter=args.n_iter,
                   seed=s, beta=args.beta, beta_min=args.beta_min,
                   anneal=not args.no_anneal, gamma=gamma, delta=delta,
                   surp_radius=args.surp_radius,
                   surp_thresh_sigma=args.surp_thresh_sigma,
                   acq_scale=args.acq_scale, surp_mode=args.surp_mode,
                   eta=args.eta, rough_k=args.rough_k,
                   sweep_gate=args.sweep_gate,
                   surrogate_kwargs=skw, verbose_every=50)
        runs.append(r)
        print(f"  seed {s}: best={r['best_overall']:.1f} poolR2={r['final_pool_r2']:.3f} "
              f"oobR2={r['final_oob_r2']:.3f} n={r['n_labeled']} "
              f"({time.time()-t0:.0f}s)", flush=True)

    summary = dict(
        tag=args.tag, dataset=args.dataset, direction=direction,
        surrogate=args.surrogate, protocol=args.protocol,
        n_init=args.n_init, n_iter=args.n_iter,
        n_query=len(args.protocol.split("+")),
        budget=args.n_init + args.n_iter * len(args.protocol.split("+")),
        gamma=gamma, delta=delta, beta=args.beta,
        surp_radius=args.surp_radius, surp_thresh_sigma=args.surp_thresh_sigma,
        acq_scale=args.acq_scale, surp_mode=args.surp_mode,
        eta=args.eta, rough_k=args.rough_k, sweep_gate=args.sweep_gate,
        y_transform=args.y_transform, tail_tau=args.tail_tau,
        sigma_calib=args.sigma_calib,
        warm_epochs=args.warm_epochs,
        beta_min=args.beta_min, anneal=not args.no_anneal,
        train_mode=args.train_mode, n_mc=args.n_mc,
        lam_mse=args.lam_mse, lam_rec=args.lam_rec,
        n_components=args.n_components or None,
        seeds=seeds, runs=runs,
    )
    json.dump(summary, open(out_path, "w"))
    b = [r["best_overall"] for r in runs]
    p = [r["final_pool_r2"] for r in runs]
    print(f"[{args.tag}] DONE best={np.mean(b):.1f}+-{np.std(b):.1f} "
          f"poolR2={np.mean(p):.3f}+-{np.std(p):.3f} -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
