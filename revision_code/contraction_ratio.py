# -*- coding: utf-8 -*-
"""The uncertainty-contraction ratio: a held-out-free check on an active-learning surrogate.

Motivation
----------
In data-scarce active learning the usual way to check whether a campaign has
left a usable surrogate behind is the bootstrap out-of-bag score, computed on
resamples of the labelled set.  On the runs in this archive that score
overstates true out-of-pool accuracy by up to 0.26, and it does so unevenly:
across twelve budget-matched arms the overstatement ranges from 0.02 to 0.26, a
factor of thirteen.  It therefore cannot be used to rank sampling strategies
against one another, which is exactly what it tends to be used for.

The quantity
------------
For a campaign that records, at each iteration, the mean predictive standard
deviation over the *unlabelled* candidates:

    contraction ratio = mean(sigma over the last  tenth of iterations)
                        ------------------------------------------
                        mean(sigma over the first tenth of iterations)

with a floor of three iterations at each end.

Nothing is held out.  The numbers are the ones the acquisition function already
evaluates in order to choose the next query.

What it does
------------
It ranks configurations.  On the 6074-composition oxygen-evolution pool, over
fourteen budget-matched configurations sharing one surrogate, every
configuration with a ratio of 0.65 or less ends with an out-of-pool R2 between
0.557 and 0.633, and every configuration with a ratio of 0.75 or above ends
between 0.305 and 0.495.  The two groups do not overlap.  It was then tested on
seven systems that played no part in choosing it (five further datasets and two
continuous-simplex oracles with no candidate pool at all) and orders the
strategies correctly on all seven.

What it does not do
-------------------
It does not predict which *seed* of a fixed configuration will end better
(|Spearman| <= 0.49, p >= 0.15), and its threshold is specific to a model
class: for a random forest the tree-ensemble spread does not contract in the
same way (ratio 1.08) even though its out-of-pool R2 is 0.582.  Calibrate the
threshold within one model class before relying on the numbers above.

Usage
-----
    python3 contraction_ratio.py                 # reproduce the validation
    from contraction_ratio import ratio          # use it on your own history
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "revision_results"


def ratio(sigma_history, frac=10, floor=3):
    """Contraction ratio of one campaign.

    Parameters
    ----------
    sigma_history : sequence of float
        Mean predictive standard deviation over the unlabelled candidates, one
        value per active-learning iteration.
    frac, floor : int
        Average over ``len // frac`` iterations at each end, never fewer than
        ``floor``.  The defaults give the last ten and first ten iterations of
        a 100-iteration campaign.

    Returns
    -------
    float, or nan when the history is too short or the initial spread is zero.
    """
    s = np.asarray(sigma_history, dtype=float)
    n = max(floor, s.size // frac)
    s0 = np.nanmean(s[:n])
    s1 = np.nanmean(s[-n:])
    if not np.isfinite(s0) or s0 <= 0:
        return float("nan")
    return float(s1 / s0)


def verdict(r, low=0.65, high=0.75):
    """Read a ratio against the thresholds calibrated for a neural ensemble."""
    if not np.isfinite(r):
        return "unknown"
    if r <= low:
        return "the surrogate improved globally"
    if r >= high:
        return "the surrogate did not improve globally"
    return "inconclusive; this is the empty band on the reference benchmark"


def _load(tag):
    f = RESULTS / f"{tag}.json"
    return json.loads(f.read_text()) if f.exists() else None


def _summarise(tags, r2_key="pool_r2"):
    ras, r2s = [], []
    for t in tags:
        d = _load(t)
        if d is None:
            continue
        for run in d["runs"]:
            h = run["history"]
            n = max(3, len(h["sigma"]) // 10)
            ras.append(ratio(h["sigma"]))
            key = r2_key if r2_key in h else "holdout_r2"
            r2s.append(float(np.nanmean(np.asarray(h[key], float)[-n:])))
    if not ras:
        return None
    return float(np.nanmean(ras)), float(np.nanmean(r2s))


def main():
    from scipy.stats import spearmanr

    print("=" * 74)
    print("Calibration: oxygen-evolution pool, fourteen budget-matched configurations")
    print("=" * 74)
    arms = ["E1_full", "E1_greedy", "E1_random", "E1_greedy_maxsigma", "E1_nodiv",
            "E1_nosurp", "E1_nodivsurp", "E1_lcbds_rand2", "E1_lcbds_x2",
            "E1_Aonly_matched", "E1_pureLCB_matched", "E1_noanneal",
            "E2_pcban_ei", "E2_pcban_ei_maxsigma"]
    rows = []
    for a in arms:
        s = _summarise([a])
        if s:
            rows.append((s[0], s[1], a))
    for ra, r2, a in sorted(rows):
        print(f"  {a:24s} ratio {ra:.2f}   out-of-pool R2 {r2:+.3f}   {verdict(ra)}")
    lo = [r for r in rows if r[0] <= 0.65]
    hi = [r for r in rows if r[0] >= 0.75]
    if lo and hi:
        print(f"\n  ratio <= 0.65 : {len(lo):2d} configurations, R2 "
              f"{min(r[1] for r in lo):+.3f} to {max(r[1] for r in lo):+.3f}")
        print(f"  ratio >= 0.75 : {len(hi):2d} configurations, R2 "
              f"{min(r[1] for r in hi):+.3f} to {max(r[1] for r in hi):+.3f}")
        print(f"  groups overlap: {max(r[1] for r in hi) >= min(r[1] for r in lo)}")

    print("\n" + "=" * 74)
    print("Validation on systems that played no part in choosing the quantity")
    print("=" * 74)
    pooled_x, pooled_y = [], []
    for ds in ("slump", "concrete", "energy", "wine_red", "steel"):
        pts = []
        for a in ("lcbds", "grd_ms", "greedy", "random"):
            s = _summarise([f"G_{ds}_{a}", f"GP_{ds}_{a}"])
            if s:
                pts.append(s)
        if len(pts) < 3:
            continue
        pooled_x += [p[0] for p in pts]; pooled_y += [p[1] for p in pts]
        rho = spearmanr([p[0] for p in pts], [p[1] for p in pts])[0]
        print(f"  {ds:10s} {len(pts)} strategies   Spearman(ratio, R2) = {rho:+.2f}")
    if pooled_x:
        rho, pv = spearmanr(pooled_x, pooled_y)
        print(f"  {'pooled':10s} {len(pooled_x)} configurations  Spearman = {rho:+.3f}  p = {pv:.3f}")

    for oracle in ("oer_twin", "analytic_sharp"):
        pts = []
        for a in ("lcbds", "greedy", "gpei", "random"):
            ras, r2s = [], []
            for prefix in ("C_", "CP_"):
                d = _load(f"{prefix}{oracle}_{a}")
                if d is None:
                    continue
                for run in d["runs"]:
                    h = run["history"]
                    n = max(3, len(h["sigma"]) // 10)
                    ras.append(ratio(h["sigma"]))
                    r2s.append(float(run["final_holdout_r2"]))
            if ras:
                pts.append((float(np.nanmean(ras)), float(np.nanmean(r2s))))
        if len(pts) >= 3:
            rho = spearmanr([p[0] for p in pts], [p[1] for p in pts])[0]
            print(f"  {oracle:10s} {len(pts)} strategies, no candidate pool   "
                  f"Spearman = {rho:+.2f}")


if __name__ == "__main__":
    main()
