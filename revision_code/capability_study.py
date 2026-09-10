"""
Where, if anywhere, does the descriptor-free PC-BAN surrogate actually beat a
simple composition-only baseline?

The active-learning benchmarks answer "does the whole pipeline work"; they do
not isolate the surrogate, and on the 4-component OER pool a plain random forest
matches PC-BAN on every axis.  This study therefore tests the four places where
an interaction-explicit neural surrogate *could* differ, each chosen because it
corresponds to something a discovery campaign actually needs:

  R1  random split          -- baseline sanity check + low-n learning curve.
                               "How much labelled data does each model need?"

  R2  leave-one-region-out  -- the input space is k-means partitioned and one
                               whole region is held out.  This is the situation
                               active learning is in on day one: predict into
                               chemistry you have never sampled.  Tree ensembles
                               are piecewise constant and cannot extrapolate;
                               a smooth network can.

  R3  composition tail      -- (OER only) train on Ce <= median, test on the
                               Ce-rich half that contains the 206 mV optimum.
                               "Can a model that has never seen Ce-rich oxides
                               point you towards them?"

  R4  dimensionality        -- the same comparison at N = 4, 7, 8, 8, 11, 13.
                               PC-BAN's BAN module generates 15 interaction
                               features at N=4 but 378 at N=13, so any benefit
                               of the architecture should grow with N.

Metrics go beyond R^2, because a discovery surrogate is used to *rank*
candidates and to *quantify its own ignorance*, not to minimise MSE:

  r2, rmse                  -- point accuracy
  rank_spearman             -- can it order the unexplored candidates at all?
  recall@k                  -- of the true top-20 in the held-out set, how many
                               are inside the model's predicted top-k?
  enrichment                -- recall relative to random selection
  sigma_err_spearman        -- does the uncertainty the acquisition function
                               consumes actually track the error?
  coverage95, nll           -- calibration
  fit_seconds               -- cost
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "revision"))

from revision.al_harness import build_surrogate, load_dataset  # noqa: E402

OUT = ROOT / "revision" / "results"
OUT.mkdir(exist_ok=True)

TOP_M = 20          # size of the "true best" set we try to recover
TOP_K = 100         # size of the model's shortlist


# ─────────────────────────────────────────────────────────────────
def evaluate(y_true, mu, sd, direction):
    """Accuracy, ranking, retrieval and uncertainty quality on a held-out set."""
    y_true = np.asarray(y_true, float)
    mu = np.asarray(mu, float)
    sd = np.maximum(np.asarray(sd, float), 1e-9)
    err = y_true - mu
    n = len(y_true)

    sign = 1.0 if direction == "min" else -1.0
    m = min(TOP_M, max(1, n // 20))
    k = min(TOP_K, max(m, n // 5))
    true_top = set(np.argsort(sign * y_true)[:m].tolist())
    pred_top = set(np.argsort(sign * mu)[:k].tolist())
    recall = len(true_top & pred_top) / len(true_top)
    baseline = k / n                      # recall a random shortlist would get
    enrichment = recall / baseline if baseline > 0 else float("nan")

    # What a chemist would actually get: synthesise the model's top-k shortlist,
    # how good is the best thing in it, and how far is that from the true optimum?
    pred_top_idx = np.argsort(sign * mu)[:k]
    best_in_topk = float(y_true[pred_top_idx].min() if direction == "min"
                         else y_true[pred_top_idx].max())
    best_possible = float(y_true.min() if direction == "min" else y_true.max())
    worst_possible = float(y_true.max() if direction == "min" else y_true.min())
    span = abs(best_possible - worst_possible) + 1e-12
    regret = abs(best_in_topk - best_possible)
    # 0 = the shortlist contains the true optimum; 1 = as bad as the pool allows
    norm_regret = float(regret / span)
    # a retrieval target that does not saturate at zero: the true top 5 %
    m5 = max(1, int(round(0.05 * n)))
    true_top5 = set(np.argsort(sign * y_true)[:m5].tolist())
    recall_top5pct = len(true_top5 & set(pred_top_idx.tolist())) / min(len(true_top5), k)

    # A degenerate (near-constant) prediction makes every ranking metric an
    # artefact of argsort tie-breaking, so mark those cells explicitly.
    degenerate = float(np.std(mu)) < 1e-8 * (abs(float(np.mean(mu))) + 1.0)
    rank_rho = float("nan") if degenerate else spearmanr(y_true, mu).correlation
    sig_rho = spearmanr(np.abs(err), sd).correlation
    if degenerate:
        recall_top5pct = float("nan")
        norm_regret = float("nan")

    return dict(
        best_in_topk=best_in_topk, best_possible=best_possible,
        regret=float(regret), norm_regret=norm_regret,
        recall_top5pct=float(recall_top5pct), degenerate=bool(degenerate),
        n_test=int(n),
        r2=float(r2_score(y_true, mu)),
        rmse=float(np.sqrt(np.mean(err**2))),
        rank_spearman=float(rank_rho) if rank_rho == rank_rho else float("nan"),
        recall_at_k=float(recall), top_m=int(m), top_k=int(k),
        enrichment=float(enrichment),
        sigma_err_spearman=float(sig_rho) if sig_rho == sig_rho else float("nan"),
        coverage95=float(np.mean(np.abs(err) <= 1.959964 * sd)),
        nll=float(np.mean(0.5 * np.log(2 * np.pi * sd**2) + 0.5 * (err / sd) ** 2)),
        mean_sigma=float(sd.mean()),
    )


def fit_and_score(surrogate, X, y, tr, te, direction, seed):
    sur = build_surrogate(surrogate)
    t0 = time.time()
    sur.fit(X[tr], y[tr], seed=seed, iteration=0, X_pool=X[te])
    mu, sd = sur.predict(X[te])
    rec = evaluate(y[te], mu, sd, direction)
    rec["fit_seconds"] = float(time.time() - t0)
    rec["oob_r2"] = float(sur.oob_r2)
    rec["n_train"] = int(len(tr))
    return rec


# ─────────────────────────────────────────────────────────────────
def splits_random(X, y, n_train, seed):
    rng = np.random.default_rng(1000 + seed)
    idx = rng.permutation(len(X))
    return [(idx[:n_train], idx[n_train:])]


def splits_region(X, y, n_clusters, seed):
    """Leave-one-region-out: whole contiguous chunks of design space held out."""
    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10).fit(X)
    out = []
    for c in range(n_clusters):
        te = np.where(km.labels_ == c)[0]
        tr = np.where(km.labels_ != c)[0]
        frac = len(te) / len(X)
        if 0.03 <= frac <= 0.5 and len(te) >= 30 and len(tr) >= 30:
            out.append((tr, te))
    return out


def splits_tail(X, y, col, seed):
    """OER-specific: train on the Ce-poor half, predict the Ce-rich half."""
    thr = np.median(X[:, col])
    tr = np.where(X[:, col] <= thr)[0]
    te = np.where(X[:, col] > thr)[0]
    return [(tr, te)]


# ─────────────────────────────────────────────────────────────────
DATASETS = ["oer", "slump", "concrete", "energy", "wine_red", "steel"]
SURROGATES = ["pcban", "rf", "gp", "mlp"]
SIZES = [20, 60, 120, 220]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default=",".join(DATASETS))
    ap.add_argument("--surrogates", default=",".join(SURROGATES))
    ap.add_argument("--regimes", default="random,region,tail")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--region-seeds", default="0,1")
    ap.add_argument("--n-clusters", type=int, default=6)
    ap.add_argument("--extrap-sizes", default="220",
                    help="training-set sizes for the region/tail extrapolation "
                         "regimes; matched to the active-learning budget")
    ap.add_argument("--out", default="capability_study.json")
    args = ap.parse_args()

    out_path = OUT / args.out
    done = {}
    if out_path.exists():
        for r in json.load(open(out_path)):
            done[(r["dataset"], r["surrogate"], r["regime"],
                  r["n_train_nominal"], r["seed"], r["fold"])] = r
        print(f"resuming: {len(done)} cells already computed", flush=True)

    records = list(done.values())
    datasets = args.datasets.split(",")
    surrogates = args.surrogates.split(",")
    regimes = args.regimes.split(",")
    seeds = [int(s) for s in args.seeds.split(",")]
    rseeds = [int(s) for s in args.region_seeds.split(",")]

    for ds in datasets:
        X, y, direction, _, _ = load_dataset(ds)
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        N, D = X.shape
        surs = list(surrogates)
        if ds == "oer" and "mlp" in surs:
            surs.append("mlp_descriptor")     # descriptor control, OER only
        print(f"\n########## {ds}  N={D} pool={N} direction={direction}", flush=True)

        jobs = []
        if "random" in regimes:
            for n_train in SIZES:
                if n_train > 0.6 * N:
                    continue
                for s in seeds:
                    for tr, te in splits_random(X, y, n_train, s):
                        jobs.append(("random", n_train, s, 0, tr, te))
        # For the extrapolation regimes the training side is subsampled to the
        # size of a real active-learning campaign: the question is "you have
        # measured n compositions in one part of the space -- what can you say
        # about the part you have never touched?", not "you have thousands".
        extrap_sizes = [int(v) for v in args.extrap_sizes.split(",")]
        if "region" in regimes:
            for s in rseeds:
                for f, (tr, te) in enumerate(splits_region(X, y, args.n_clusters, s)):
                    for nt in extrap_sizes:
                        if nt > len(tr):
                            continue
                        rng = np.random.default_rng(50000 + 97 * s + f)
                        tr_s = rng.choice(tr, size=nt, replace=False)
                        jobs.append(("region", nt, s, f, tr_s, te))
        if "tail" in regimes and ds == "oer":
            for f, (tr, te) in enumerate(splits_tail(X, y, 3, 0)):   # col 3 = Ce
                for nt in extrap_sizes:
                    for s in seeds:
                        if nt > len(tr):
                            continue
                        rng = np.random.default_rng(60000 + 97 * s)
                        tr_s = rng.choice(tr, size=nt, replace=False)
                        jobs.append(("tail", nt, s, f, tr_s, te))

        for regime, n_nom, s, fold, tr, te in jobs:
            for sur in surs:
                key = (ds, sur, regime, n_nom, s, fold)
                if key in done:
                    continue
                try:
                    rec = fit_and_score(sur, X, y, tr, te, direction, s)
                except Exception as e:
                    print(f"  FAIL {ds}/{sur}/{regime}/n={n_nom}/seed{s}/f{fold}: {e}",
                          flush=True)
                    continue
                rec.update(dataset=ds, n_dim=int(D), pool=int(N),
                           direction=direction, surrogate=sur, regime=regime,
                           n_train_nominal=n_nom, seed=s, fold=fold)
                records.append(rec)
                done[key] = rec
                print(f"  {ds:9s} {sur:15s} {regime:6s} n={rec['n_train']:4d} "
                      f"f{fold} s{s} | R2={rec['r2']:+.3f} rank={rec['rank_spearman']:+.3f} "
                      f"rec5%={rec['recall_top5pct']:.2f} "
                      f"regret={rec['norm_regret']:.3f} sig_rho={rec['sigma_err_spearman']:+.3f} "
                      f"cov={rec['coverage95']:.2f} {rec['fit_seconds']:.0f}s", flush=True)
                json.dump(records, open(out_path, "w"))

    json.dump(records, open(out_path, "w"))
    print(f"\nDONE: {len(records)} cells -> {out_path}")


if __name__ == "__main__":
    main()
