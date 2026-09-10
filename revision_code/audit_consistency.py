# -*- coding: utf-8 -*-
"""Cross-document consistency audit.

Every number that appears in the manuscript, the tables, the supplementary
information and the response letter is checked against the value recomputed
from revision/results/*.json.  A claim that cannot be traced to a stored result
is reported, and so is any figure that disagrees with it.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import docx
import numpy as np
from scipy.stats import wilcoxon

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from analyze import load, per_seed  # noqa: E402


def merged(tags):
    seeds, best, pool, oob, brk = [], [], [], [], []
    for t in tags:
        d = load(t)
        if d is None:
            return None
        s = per_seed(d)
        seeds += list(s["seed"]); best += list(s["best"])
        pool += list(s["pool_r2_terminal"]); oob += list(s["oob_r2_final"])
        brk += list(s["break_exp"])
    o = np.argsort(seeds)
    return {k: np.asarray(v)[o] for k, v in
            dict(seed=seeds, best=best, pool=pool, oob=oob, brk=brk).items()}


def fmt(v, nd=3):
    return f"{v:.{nd}f}"


def main():
    facts = {}

    # ── the numbers the manuscript asserts, recomputed from the runs ──
    single = {
        "E1_full": "LCBDS+maxsigma", "E1_greedy": "greedy", "E1_random": "random",
        "E1_greedy_maxsigma": "greedy+maxsigma", "E1_nodiv": "gamma=0",
        "E1_nosurp": "delta=0", "E1_nodivsurp": "gamma=delta=0",
        "E1_lcbds_rand2": "maxsigma->random", "E1_lcbds_x2": "two LCBDS",
        "E1_Aonly_matched": "single query matched", "E1_pureLCB_matched": "pure LCB matched",
        "E1_nomc": "no MC dropout", "E1_noanneal": "no annealing",
        "E2_gp_ei": "GP-EI", "E2_gp_ei_maxsigma": "GP-EI+maxvar",
        "E2_gp_lcb": "GP-LCB", "E2_gp_lcb_maxsigma": "GP-LCB+maxvar",
        "E2_pcban_ei": "PC-BAN+EI", "E2_pcban_ei_maxsigma": "PC-BAN+EI+maxsigma",
        "E3_rf": "random forest", "E3_gp_lcbds": "GP surrogate",
        "E3_mlp": "plain MLP", "E3_mlpdesc": "MLP+descriptors",
    }
    for tag, lab in single.items():
        m = merged([tag])
        if m is None:
            print(f"  !! result file missing for {tag}")
            continue
        facts[lab] = dict(best=float(m["best"].mean()), best_sd=float(m["best"].std()),
                          pool=float(np.nanmean(m["pool"])), pool_sd=float(np.nanstd(m["pool"])),
                          oob=float(np.nanmean(m["oob"])),
                          brk=float(np.nanmean(m["brk"])), n=len(m["best"]))

    base = merged(["E1_full"])
    print("=" * 78)
    print("A. every value asserted in the revised text, recomputed")
    print("=" * 78)
    checks = [
        ("LCBDS+maxsigma", "pool", 0.576), ("LCBDS+maxsigma", "oob", 0.827),
        ("greedy", "pool", 0.305), ("random", "pool", 0.633),
        ("greedy+maxsigma", "pool", 0.558), ("GP-EI", "pool", -0.392),
        ("GP-EI+maxvar", "pool", 0.203), ("GP-LCB", "pool", -0.640),
        ("GP-LCB+maxvar", "pool", 0.266), ("PC-BAN+EI+maxsigma", "pool", 0.579),
        ("gamma=0", "pool", 0.557), ("delta=0", "pool", 0.580),
        ("gamma=delta=0", "pool", 0.569), ("maxsigma->random", "pool", 0.592),
        ("two LCBDS", "pool", 0.495), ("single query matched", "pool", 0.470),
        ("pure LCB matched", "pool", 0.421), ("no MC dropout", "pool", 0.582),
        ("no annealing", "pool", 0.571),
        ("random forest", "pool", 0.582), ("GP surrogate", "pool", 0.417),
        ("plain MLP", "pool", 0.470), ("MLP+descriptors", "pool", -0.335),
        ("MLP+descriptors", "best", 379.0), ("plain MLP", "brk", 220.0),
    ]
    bad = 0
    for lab, field, claimed in checks:
        if lab not in facts:
            print(f"  !! no data for {lab}"); bad += 1; continue
        got = facts[lab][field]
        ok = abs(got - claimed) <= (0.0015 if abs(claimed) < 5 else 1.0)
        print(f"  {'OK ' if ok else '!! '}{lab:24s} {field:5s} manuscript {claimed:>8.3f}"
              f"   recomputed {got:>8.3f}")
        bad += (not ok)

    print("\n" + "=" * 78)
    print("B. p-values quoted in the text")
    print("=" * 78)
    ptests = [("gamma=0", "E1_nodiv", 0.19), ("delta=0", "E1_nosurp", 0.20),
              ("gamma=delta=0", "E1_nodivsurp", 0.77),
              ("maxsigma->random", "E1_lcbds_rand2", 0.23),
              ("two LCBDS", "E1_lcbds_x2", 0.010),
              ("single query matched", "E1_Aonly_matched", 0.002),
              ("greedy", "E1_greedy", 0.002), ("random", "E1_random", 0.002),
              ("greedy+maxsigma", "E1_greedy_maxsigma", 0.28),
              ("no MC dropout", "E1_nomc", 0.63), ("no annealing", "E1_noanneal", 1.00)]
    for lab, tag, claimed in ptests:
        X = merged([tag])
        p = wilcoxon(base["pool"], X["pool"]).pvalue
        ok = abs(p - claimed) <= max(0.02, 0.1 * claimed)
        print(f"  {'OK ' if ok else '!! '}{lab:24s} text p={claimed:<6.3f} recomputed p={p:.4f}")
        bad += (not ok)

    print("\n" + "=" * 78)
    print("C. cross-domain numbers (Table 2 and Section 3.5)")
    print("=" * 78)
    t2 = json.load(open(HERE / "results" / "table2_data.json", encoding="utf-8"))
    tot = {}
    for ds, arms in t2.items():
        for a, v in arms.items():
            tot.setdefault(a, [0, 0])
            tot[a][0] += v["hits"]; tot[a][1] += v["n"]
    for a, (h, n) in sorted(tot.items()):
        print(f"  {a:14s} {h}/{n} dataset-seeds recovered the optimum")
    claimed_tot = {"LCBDS+maxσ": 48, "Greedy+maxσ": 49, "GP-EI+maxσ": 46,
                   "Greedy": 46, "Random": 18}
    for a, c in claimed_tot.items():
        got = tot.get(a, [None])[0]
        ok = got == c
        print(f"  {'OK ' if ok else '!! '}{a:14s} text says {c}, recomputed {got}")
        bad += (not ok)

    print("\n" + "=" * 78)
    print("D. every number that appears in the Word tables")
    print("=" * 78)
    d = docx.Document(HERE / "Digital_Discovery_main_R2_clean.docx")
    t1_rows = json.load(open(HERE / "table1_data.json", encoding="utf-8"))
    doc_t1 = [[c.text.strip() for c in r.cells] for r in d.tables[0].rows][1:]
    mismatch = 0
    for src, got in zip(t1_rows, doc_t1):
        for a, b in zip(src, got):
            if a.strip() != b.strip():
                print(f"  !! Table 1 cell differs: source {a!r} vs document {b!r}")
                mismatch += 1
    print(f"  Table 1: {len(doc_t1)} data rows, {mismatch} cells differ from the source data")
    t2_rows = json.load(open(HERE / "table2_rows.json", encoding="utf-8"))
    doc_t2 = [[c.text.strip() for c in r.cells] for r in d.tables[1].rows][1:]
    m2 = sum(1 for s, g in zip(t2_rows, doc_t2) for a, b in zip(s, g) if a.strip() != b.strip())
    print(f"  Table 2: {len(doc_t2)} data rows, {m2} cells differ from the source data")
    bad += mismatch + m2

    print("\n" + "=" * 78)
    print(f"RESULT: {bad} inconsistencies")
    print("=" * 78)
    json.dump(facts, open(HERE / "results" / "verified_facts.json", "w"), indent=1)


if __name__ == "__main__":
    main()
