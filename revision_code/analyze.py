"""
Aggregate every controlled run into one table on ONE common metric.

Primary surrogate-quality metric (design requirement):
    pool_R2_terminal = mean of the out-of-pool R^2 over the last 10 AL
    iterations, computed identically for every surrogate (PC-BAN, GP, RF, MLP)
    on the compositions that strategy has never queried.

We also carry the single-shot final-refit value and, where it exists, the
bootstrap OOB R^2, so that the gap between the two can be quantified.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

RES = Path(__file__).resolve().parent / "results"
THRESH = 370.0


def load(tag):
    p = RES / f"{tag}.json"
    return json.load(open(p)) if p.exists() else None


def per_seed(d, tail=10):
    """Return dict of per-seed metric arrays for one arm."""
    out = {"seed": [], "best": [], "pool_r2_terminal": [], "pool_r2_final_refit": [],
           "oob_r2_final": [], "break_exp": [], "hit206": [], "fit_seconds": [],
           "total_seconds": []}
    nq = d["n_query"]
    n_init = d["n_init"]
    direction = d.get("direction", "min")
    for r in d["runs"]:
        h = r["history"]
        out["seed"].append(r["seed"])
        out["best"].append(r["best_overall"])
        pr = np.asarray(h["pool_r2"], dtype=float)
        out["pool_r2_terminal"].append(float(np.nanmean(pr[-tail:])))
        out["pool_r2_final_refit"].append(r["final_pool_r2"])
        out["oob_r2_final"].append(r["final_oob_r2"])
        out["fit_seconds"].append(r["total_fit_seconds"])
        out["total_seconds"].append(r["total_seconds"])
        if direction == "min":
            hits = [i for i, v in enumerate(h["best"]) if v < THRESH]
            out["break_exp"].append(n_init + hits[0] * nq if hits
                                    else n_init + d["n_iter"] * nq)
            out["hit206"].append(1.0 if r["best_overall"] <= 210 else 0.0)
        else:
            out["break_exp"].append(np.nan)
            out["hit206"].append(np.nan)
    return {k: np.asarray(v, dtype=float) if k != "seed" else np.asarray(v)
            for k, v in out.items()}


def ms(a):
    a = np.asarray(a, dtype=float)
    return f"{np.nanmean(a):.3f} ± {np.nanstd(a):.3f}"


def paired_test(a, b, seeds_a=None, seeds_b=None):
    """Paired test aligned on seed id, so arms run with different seed sets
    (e.g. the 5-seed distillation arms vs the 10-seed main arms) are compared
    only on the seeds they share."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if seeds_a is not None and seeds_b is not None:
        common = [s for s in seeds_a if s in set(seeds_b)]
        if len(common) < 5:
            return float("nan")
        ia = [list(seeds_a).index(s) for s in common]
        ib = [list(seeds_b).index(s) for s in common]
        a, b = a[ia], b[ib]
    elif len(a) != len(b):
        return float("nan")
    m = ~(np.isnan(a) | np.isnan(b))
    if m.sum() < 5 or np.allclose(a[m], b[m]):
        return float("nan")
    try:
        return float(wilcoxon(a[m], b[m]).pvalue)
    except Exception:
        return float("nan")


def table(tags, title, ref=None):
    print(f"\n### {title}\n")
    hdr = (f"| {'arm':<24} | {'surrogate':<14} | {'protocol':<16} | budget | "
           f"{'best J10 (mV)':<16} | {'break_exp':<14} | "
           f"{'pool R2 (common)':<18} | {'pool R2 refit':<16} | {'OOB R2':<16} |")
    print(hdr)
    print("|" + "|".join(["-" * (len(c)) for c in hdr.split("|")[1:-1]]) + "|")
    store = {}
    for t in tags:
        d = load(t)
        if d is None:
            print(f"| {t:<24} | (not finished) |")
            continue
        s = per_seed(d)
        store[t] = (d, s)
        print(f"| {t:<24} | {d['surrogate']:<14} | {d['protocol']:<16} | "
              f"{d['budget']:>6} | "
              f"{np.mean(s['best']):>6.1f} ± {np.std(s['best']):<7.1f} | "
              f"{np.nanmean(s['break_exp']):>5.0f} ± {np.nanstd(s['break_exp']):<6.0f} | "
              f"{ms(s['pool_r2_terminal']):<18} | {ms(s['pool_r2_final_refit']):<16} | "
              f"{ms(s['oob_r2_final']):<16} |")
    if ref and ref in store:
        print(f"\nPaired Wilcoxon vs `{ref}` (same 10 seeds):\n")
        print(f"| {'arm':<24} | p(best) | p(pool R2 common) | p(break_exp) | n paired |")
        print(f"|{'-'*26}|---------|-------------------|--------------|----------|")
        for t in tags:
            if t == ref or t not in store:
                continue
            a, b = store[ref][1], store[t][1]
            sa, sb = a["seed"], b["seed"]
            n_common = len(set(sa) & set(sb))
            print(f"| {t:<24} | {paired_test(a['best'], b['best'], sa, sb):.4f} | "
                  f"{paired_test(a['pool_r2_terminal'], b['pool_r2_terminal'], sa, sb):.4f} | "
                  f"{paired_test(a['break_exp'], b['break_exp'], sa, sb):.4f} | n={n_common} |")
    return store


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"

    if which in ("all", "e2"):
        table(["E2_gp_ei", "E2_gp_ei_maxsigma", "E2_gp_lcb", "E2_gp_lcb_maxsigma",
               "E2_gp_random", "E2_pcban_ei", "E2_pcban_ei_maxsigma",
               "E1_full", "E1_greedy", "E1_greedy_maxsigma", "E1_random"],
              "exploration protocol held constant",
              ref="E1_full")

    if which in ("all", "e1"):
        table(["E1_full", "E1_nodiv", "E1_nosurp", "E1_nodivsurp",
               "E1_lcbds_rand2", "E1_lcbds_x2",
               "E1_Aonly_matched", "E1_pureLCB_matched",
               "E1_greedy", "E1_random"],
              "budget-matched component ablation",
              ref="E1_full")

    if which in ("all", "e3"):
        table(["E1_full", "E3_rf", "E3_gp_lcbds", "E3_mlp", "E3_mlpdesc"],
              "surrogate held-out under one identical protocol",
              ref="E1_full")

    if which in ("all", "e4"):
        table(["E1_full", "E4_warm", "E4_scratch"],
              "retraining protocol (warm start / distillation)",
              ref="E1_full")
