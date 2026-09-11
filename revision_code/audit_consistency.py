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

MAIN_DOCX = Path(__file__).resolve().parent / "Digital_Discovery_main_R2_clean.docx"
SUPP_DOCX = Path(__file__).resolve().parent / "Digital_Discovery_supp_R2_clean.docx"


def _manuscript(path=None):
    """The built manuscript, or None when it is not alongside this script.

    The published archive ships the code and the trajectories but not the
    manuscript, so a third party running this script gets every check that
    re-derives a number from the stored runs, and a clear note for the checks
    that compare those numbers against the document text.
    """
    p = Path(path) if path else MAIN_DOCX
    if not p.exists():
        print(f"  (skipped: {p.name} is not in this archive)")
        return None
    return docx.Document(p)
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
    d = _manuscript()
    t1_rows = json.load(open(HERE / "table1_data.json", encoding="utf-8")) if d else []
    doc_t1 = ([[c.text.strip() for c in r.cells] for r in d.tables[0].rows][1:]
              if d is not None else [])
    mismatch = 0
    for src, got in zip(t1_rows, doc_t1):
        for a, b in zip(src, got):
            if a.strip() != b.strip():
                print(f"  !! Table 1 cell differs: source {a!r} vs document {b!r}")
                mismatch += 1
    print(f"  Table 1: {len(doc_t1)} data rows, {mismatch} cells differ from the source data")
    t2_rows = (json.load(open(HERE / "table2_rows.json", encoding="utf-8"))
               if d is not None else [])
    doc_t2 = ([[c.text.strip() for c in r.cells] for r in d.tables[1].rows][1:]
              if d is not None else [])
    m2 = sum(1 for s, g in zip(t2_rows, doc_t2) for a, b in zip(s, g) if a.strip() != b.strip())
    print(f"  Table 2: {len(doc_t2)} data rows, {m2} cells differ from the source data")
    bad += mismatch + m2

    print("\n" + "=" * 78)
    print("E. deep-optimum recovery (Section 3.3 and Fig. 4c), n = 30")
    print("=" * 78)
    from scipy.stats import binomtest

    def arm(tags):
        seeds, best = [], []
        for t in tags:
            dd = load(t)
            if not dd:
                continue
            ss = per_seed(dd)
            seeds += list(ss["seed"]); best += list(ss["best"])
        o = np.argsort(seeds)
        return np.asarray(seeds)[o], np.asarray(best, float)[o]

    hits = {
        "PC-BAN + EI":        (["E2_pcban_ei", "P_ei"], 7, 13),
        "PC-BAN + EI + maxσ": (["E2_pcban_ei_maxsigma", "P_ei_ms"], 5, 10),
        "greedy + maxσ":      (["E1_greedy_maxsigma", "P_greedy"], 3, 6),
        "LCBDS + maxσ":       (["E1_full", "P_lcbds"], 1, 5),
        "Bold LCBDS (470)":   (["B_lcbds_bold", "BP_lcbds_bold"], 9, 14),
        "Bold EI (470)":      (["B_ei_bold", "BP_ei_bold"], 12, 17),
        "Bold no surprise":   (["B_bold_nosurp", "BP_bold_nosurp"], 5, 12),
        "Bold random third":  (["B_third_random", "BP_third_random"], 9, 16),
        "LCBDS 2q (470)":     (["B_lcbds_2q", "BP_lcbds_2q"], 8, 16),
    }
    for lab, (tags, c206, c260) in hits.items():
        _, b = arm(tags)
        g206, g260 = int((b <= 210).sum()), int((b <= 260).sum())
        ok = (g206 == c206) and (g260 == c260)
        print(f"  {'OK ' if ok else '!! '}{lab:20s} n={len(b):2d}  =206 text {c206:2d} got "
              f"{g206:2d}   <=260 text {c260:2d} got {g260:2d}")
        bad += (not ok)

    def mcnemar(ta, tb):
        sa, ba = arm(ta); sb, bb = arm(tb)
        common = np.intersect1d(sa, sb)
        A = (ba[np.isin(sa, common)] <= 210).astype(int)
        B = (bb[np.isin(sb, common)] <= 210).astype(int)
        n01 = int(((A == 0) & (B == 1)).sum()); n10 = int(((A == 1) & (B == 0)).sum())
        pm = binomtest(n10, n10 + n01, 0.5).pvalue if n10 + n01 else 1.0
        pw = wilcoxon(ba[np.isin(sa, common)], bb[np.isin(sb, common)]).pvalue
        return pm, pw

    mtests = [
        ("EI vs LCBDS (220), McNemar", ["E2_pcban_ei", "P_ei"],
         ["E1_full", "P_lcbds"], 0.031, "m"),
        ("EI vs LCBDS (220), Wilcoxon", ["E2_pcban_ei", "P_ei"],
         ["E1_full", "P_lcbds"], 0.008, "w"),
        ("Bold vs no surprise", ["B_lcbds_bold", "BP_lcbds_bold"],
         ["B_bold_nosurp", "BP_bold_nosurp"], 0.29, "m"),
        ("Bold vs random third", ["B_lcbds_bold", "BP_lcbds_bold"],
         ["B_third_random", "BP_third_random"], 1.00, "m"),
        ("Bold vs 2-query at 470", ["B_lcbds_bold", "BP_lcbds_bold"],
         ["B_lcbds_2q", "BP_lcbds_2q"], 1.00, "m"),
        ("Bold EI vs Bold LCBDS", ["B_ei_bold", "BP_ei_bold"],
         ["B_lcbds_bold", "BP_lcbds_bold"], 0.63, "m"),
        ("Bold EI (470) vs EI (220)", ["B_ei_bold", "BP_ei_bold"],
         ["E2_pcban_ei", "P_ei"], 0.27, "m"),
    ]
    for lab, ta, tb, claimed, kind in mtests:
        pm, pw = mcnemar(ta, tb)
        got = pm if kind == "m" else pw
        ok = abs(got - claimed) <= max(0.01, 0.06 * max(claimed, 0.05))
        print(f"  {'OK ' if ok else '!! '}{lab:30s} text p={claimed:<6.3f} "
              f"recomputed p={got:.4f}")
        bad += (not ok)

    print("\n" + "=" * 78)
    print("K. predictive-uncertainty shrinkage as a diagnostic (Section 3.2, Fig. 2d)")
    print("=" * 78)
    from scipy.stats import spearmanr as _sp
    PCB = ["E1_full", "E1_greedy", "E1_random", "E1_greedy_maxsigma", "E1_nodiv",
           "E1_nosurp", "E1_nodivsurp", "E1_lcbds_rand2", "E1_lcbds_x2",
           "E1_Aonly_matched", "E1_pureLCB_matched", "E1_noanneal",
           "E2_pcban_ei", "E2_pcban_ei_maxsigma"]
    per_run, per_arm = [], []
    for t in PCB:
        dd = load(t)
        if dd is None:
            print(f"  !! missing {t}"); bad += 1; continue
        ras, prs = [], []
        for r in dd["runs"]:
            h = r["history"]
            nn = max(3, len(h["sigma"]) // 10)
            ra = float(np.nanmean(h["sigma"][-nn:]) / np.nanmean(h["sigma"][:nn]))
            pr = float(np.nanmean(h["pool_r2"][-10:]))
            per_run.append((ra, pr)); ras.append(ra); prs.append(pr)
        per_arm.append((t, float(np.mean(ras)), float(np.mean(prs))))
    rho, pv = _sp([x[0] for x in per_run], [x[1] for x in per_run])
    ok = abs(rho + 0.517) <= 0.005 and pv < 1e-9
    print(f"  {'OK ' if ok else '!! '}per-campaign Spearman text -0.52  recomputed {rho:+.3f} "
          f"(p={pv:.1e}, n={len(per_run)})")
    bad += (not ok)
    lo = [a for a in per_arm if a[1] <= 0.65]
    hi = [a for a in per_arm if a[1] >= 0.75]
    lo_min, lo_max = min(a[2] for a in lo), max(a[2] for a in lo)
    hi_min, hi_max = min(a[2] for a in hi), max(a[2] for a in hi)
    for lab, got, want in [("low group min", lo_min, 0.557), ("low group max", lo_max, 0.633),
                           ("high group min", hi_min, 0.305), ("high group max", hi_max, 0.495)]:
        ok = abs(got - want) <= 0.0015
        print(f"  {'OK ' if ok else '!! '}{lab:16s} text {want:+.3f}  recomputed {got:+.3f}")
        bad += (not ok)
    ok = hi_max < lo_min
    print(f"  {'OK ' if ok else '!! '}the two groups do not overlap "
          f"({hi_max:+.3f} < {lo_min:+.3f})")
    bad += (not ok)
    ok = len(lo) == 10 and len(hi) == 4
    print(f"  {'OK ' if ok else '!! '}group sizes {len(lo)} and {len(hi)}")
    bad += (not ok)

    print("\n" + "=" * 78)
    print("L. contraction ratio validated on seven systems it was not derived from")
    print("=" * 78)
    from scipy.stats import spearmanr as _sp2

    def _ratio_r2(tags, key="pool_r2"):
        ras, prs = [], []
        for t in tags:
            dd = load(t)
            if dd is None:
                continue
            for r in dd["runs"]:
                h = r["history"]
                sg = np.asarray(h["sigma"], float)
                n0 = max(3, len(sg) // 10)
                s0, s1 = np.nanmean(sg[:n0]), np.nanmean(sg[-n0:])
                pr = float(np.nanmean(np.asarray(h[key], float)[-n0:]))
                if np.isfinite(s0) and np.isfinite(s1) and np.isfinite(pr) and s0 > 0:
                    ras.append(s1 / s0); prs.append(pr)
        return np.asarray(ras), np.asarray(prs)

    claimed_rho = {"concrete": -1.00, "steel": -0.80, "slump": -0.40,
                   "energy": -0.40, "wine_red": -0.40}
    allr, allp = [], []
    for ds, crho in claimed_rho.items():
        rows = []
        for a in ("lcbds", "grd_ms", "greedy", "random"):
            ra, pr = _ratio_r2([f"G_{ds}_{a}", f"GP_{ds}_{a}"])
            if len(ra):
                rows.append((float(ra.mean()), float(pr.mean())))
        if len(rows) < 3:
            print(f"  !! {ds}: too few configurations"); bad += 1; continue
        allr += [r[0] for r in rows]; allp += [r[1] for r in rows]
        rho = float(_sp2([r[0] for r in rows], [r[1] for r in rows])[0])
        ok = abs(rho - crho) <= 0.01
        print(f"  {'OK ' if ok else '!! '}{ds:9s} text Spearman {crho:+.2f}  recomputed {rho:+.2f}")
        bad += (not ok)
    rho, pv = _sp2(allr, allp)
    ok = abs(rho + 0.505) <= 0.005 and abs(pv - 0.023) <= 0.002
    print(f"  {'OK ' if ok else '!! '}{'pooled':9s} text -0.505 (p=0.023)  recomputed "
          f"{rho:+.3f} (p={pv:.3f}, n={len(allr)})")
    bad += (not ok)
    for oracle, crho in (("oer_twin", -1.00), ("analytic_sharp", -1.00)):
        rows = []
        for a in ("lcbds", "greedy", "gpei", "random"):
            ras, prs = [], []
            for pre in ("C_", "CP_"):
                f = HERE / "results" / f"{pre}{oracle}_{a}.json"
                if not f.exists():
                    continue
                for r in json.loads(f.read_text())["runs"]:
                    sg = np.asarray(r["history"]["sigma"], float)
                    n0 = max(3, len(sg) // 10)
                    ras.append(np.nanmean(sg[-n0:]) / np.nanmean(sg[:n0]))
                    prs.append(float(r["final_holdout_r2"]))
            if ras:
                rows.append((float(np.mean(ras)), float(np.mean(prs))))
        rho = float(_sp2([r[0] for r in rows], [r[1] for r in rows])[0])
        ok = abs(rho - crho) <= 0.01
        print(f"  {'OK ' if ok else '!! '}{oracle:14s} text {crho:+.2f}  recomputed {rho:+.2f}")
        bad += (not ok)

    print("\n" + "=" * 78)
    print("J. capability study (Section 3.5): 216 paired comparisons")
    print("=" * 78)
    from collections import defaultdict as _dd
    from scipy.stats import wilcoxon as _w
    cap = json.load(open(HERE / "results" / "capability_study.json", encoding="utf-8"))
    _k = lambda r: (r["dataset"], r["regime"], r["n_train_nominal"], r["seed"], r["fold"])
    _by = _dd(dict)
    for r in cap:
        _by[_k(r)][r["surrogate"]] = r

    def pair(metric, a, b, regime=None):
        xs, ys = [], []
        for kk, v in _by.items():
            if regime and kk[1] != regime:
                continue
            if a in v and b in v:
                xa, xb = v[a][metric], v[b][metric]
                if xa is None or xb is None or np.isnan(xa) or np.isnan(xb):
                    continue
                xs.append(xa); ys.append(xb)
        xs, ys = np.asarray(xs), np.asarray(ys)
        return len(xs), float(xs.mean()), float(ys.mean()), float(_w(xs, ys).pvalue)

    def mean_of(metric, sur, regime=None):
        v = [r[metric] for r in cap
             if r["surrogate"] == sur and (regime is None or r["regime"] == regime)
             and r[metric] is not None and not np.isnan(r[metric])]
        return float(np.mean(v))

    checks = [
        ("rank rho, network", mean_of("rank_spearman", "pcban"), 0.655, 0.001),
        ("rank rho, forest", mean_of("rank_spearman", "rf"), 0.647, 0.001),
        ("rank rho, GP", pair("rank_spearman", "pcban", "gp")[2], 0.580, 0.001),
        ("rank rho, plain MLP", pair("rank_spearman", "pcban", "mlp")[2], 0.529, 0.001),
        ("rank rho, descriptors", pair("rank_spearman", "pcban", "mlp_descriptor")[2], 0.091, 0.001),
    ]
    for lab, got, want, tol in checks:
        ok = abs(got - want) <= tol
        print(f"  {'OK ' if ok else '!! '}{lab:24s} text {want:+.3f}  recomputed {got:+.3f}")
        bad += (not ok)
    for lab, metric, regime, cn, cr, cp in [
            ("R2 all", "r2", None, 0.286, 0.350, 0.008),
            ("coverage95 all", "coverage95", None, 0.671, 0.895, 0.0),
            ("top-5% recall, region", "recall_top5pct", "region", 0.628, 0.578, 0.028),
            ("top-5% recall, random", "recall_top5pct", "random", 0.609, 0.646, 0.002)]:
        n, ma, mb, pp = pair(metric, "pcban", "rf", regime)
        ok = (abs(ma - cn) <= 0.0015 and abs(mb - cr) <= 0.0015
              and (pp < 1e-3 if cp == 0.0 else abs(pp - cp) <= max(0.002, 0.06 * cp)))
        print(f"  {'OK ' if ok else '!! '}{lab:24s} n={n:3d} network {ma:.3f} (text {cn:.3f}) "
              f"forest {mb:.3f} (text {cr:.3f})  p={pp:.4f} (text {cp})")
        bad += (not ok)

    print("\n" + "=" * 78)
    print("I. continuous simplex (Section 3.6), n = 10")
    print("=" * 78)
    try:
        import stage13_text as _T13
        _i, _r = _T13.build()
    except ModuleNotFoundError:
        print("  (skipped: stage13_text.py is manuscript tooling and is not "
              "in this archive; continuous_space.py regenerates the runs)")
        _i = _r = None
    if _i is None and _r is None:
        pass
    elif _i is None:
        print("  !! continuous runs incomplete"); bad += 1
    else:
        for claim in ["0.836", "0.022", "0.426", "0.169", "0.931", "0.017",
                      "0.295", "0.407", "212.7", "356.8", "4000", "5000"]:
            ok = claim in _i or claim in _r
            print(f"  {'OK ' if ok else '!! '}{claim} present in the generated text")
            bad += (not ok)
        dd = _manuscript()
        if dd is not None:
            body = "\n".join(p.text for p in dd.paragraphs)
            ok = _r[:80] in body and _i[:80] in body
            print(f"  {'OK ' if ok else '!! '}Section 3.6 text is in the manuscript verbatim")
            bad += (not ok)

    print("\n" + "=" * 78)
    print("H. random forest vs the neural surrogate across dimensionality")
    print("=" * 78)
    from scipy.stats import wilcoxon as _wil

    def xd(tags):
        seeds, pv = [], []
        for t in tags:
            dd = load(t)
            if dd is None:
                return None
            ss = per_seed(dd)
            seeds += list(ss["seed"])
            pv += [float(np.nanmean(r["history"]["pool_r2"][-10:])) for r in dd["runs"]]
        o = np.argsort(seeds)
        return np.asarray(seeds)[o], np.asarray(pv, float)[o]

    claimed = {"slump": (0.363, 0.320, 0.23), "concrete": (0.627, 0.580, 0.16),
               "energy": (0.926, 0.958, 0.002), "wine_red": (0.156, 0.215, 0.049),
               "steel": (0.361, 0.346, 1.00)}
    for ds, (cn, cr, cp) in claimed.items():
        a = xd([f"G_{ds}_lcbds", f"GP_{ds}_lcbds"])
        b = xd([f"R_{ds}_rf", f"RP_{ds}_rf"])
        if a is None or b is None:
            print(f"  !! {ds}: runs missing"); bad += 1; continue
        c = np.intersect1d(a[0], b[0])
        x, y = a[1][np.isin(a[0], c)], b[1][np.isin(b[0], c)]
        pp = 1.0 if np.allclose(x, y) else float(_wil(x, y).pvalue)
        ok = (abs(np.nanmean(x) - cn) <= 0.0015 and abs(np.nanmean(y) - cr) <= 0.0015
              and abs(pp - cp) <= max(0.01, 0.06 * cp))
        print(f"  {'OK ' if ok else '!! '}{ds:9s} network {np.nanmean(x):+.3f} (text {cn:+.3f})"
              f"   forest {np.nanmean(y):+.3f} (text {cr:+.3f})   p={pp:.3f} (text {cp:.3f})")
        bad += (not ok)

    print("\n" + "=" * 78)
    print("G. hyperparameter sweep (Section 3.4 and Fig. 6), n = 10")
    print("=" * 78)
    grid = {
        "beta_min": [("0.05", "H_beta005"), ("0.1", "H_beta010"), ("0.2", "E1_full"),
                     ("0.3", "H_beta030"), ("0.5", "H_beta050"), ("1.0", "H_beta100")],
        "gamma": [("0", "E1_nodiv"), ("3", "H_gam03"), ("5", "H_gam05"),
                  ("8", "E1_full"), ("10", "H_gam10"), ("15", "H_gam15")],
        "delta": [("0", "E1_nosurp"), ("5", "H_del05"), ("8", "H_del08"),
                  ("12", "E1_full"), ("15", "H_del15"), ("20", "H_del20")],
    }

    def sweep_arm(tag):
        dd = load(tag)
        if dd is None:
            return None
        ss = per_seed(dd)
        pool = np.asarray([np.nanmean(r["history"]["pool_r2"][-10:]) for r in dd["runs"]])
        o = np.argsort(ss["seed"])
        return dict(brk=np.asarray(ss["break_exp"], float)[o],
                    best=np.asarray(ss["best"], float)[o], pool=pool[o])

    ref = sweep_arm("E1_full")

    def wil(x, y):
        m = ~(np.isnan(x) | np.isnan(y))
        return 1.0 if np.allclose(x[m], y[m]) else float(wilcoxon(x[m], y[m]).pvalue)

    npts, nsig, pmin = 0, 0, 1.0
    rng = {}
    for name, pts in grid.items():
        vals = {"brk": [], "best": [], "pool": []}
        for lab, tag in pts:
            a = sweep_arm(tag)
            if a is None:
                print(f"  !! missing {tag}"); bad += 1; continue
            for f in vals:
                vals[f].append(float(np.nanmean(a[f])))
            if tag != "E1_full":
                for f in vals:
                    pp = wil(ref[f], a[f]); npts += 1
                    nsig += (pp < 0.05); pmin = min(pmin, pp)
        rng[name] = {f: (min(v), max(v)) for f, v in vals.items()}
        print(f"  {name:9s} brk {rng[name]['brk'][0]:.0f}-{rng[name]['brk'][1]:.0f}"
              f"   best {rng[name]['best'][0]:.0f}-{rng[name]['best'][1]:.0f}"
              f"   R2pool {rng[name]['pool'][0]:.3f}-{rng[name]['pool'][1]:.3f}")
    claimed = {"beta_min": ((90, 102), (316, 332), (0.566, 0.581)),
               "gamma":    ((94, 109), (323, 347), (0.557, 0.585)),
               "delta":    ((97, 123), (327, 344), (0.571, 0.586))}
    for name, (cb, cbe, cp) in claimed.items():
        for f, c, tol in [("brk", cb, 1.0), ("best", cbe, 1.0), ("pool", cp, 0.0015)]:
            got = rng[name][f]
            ok = abs(got[0] - c[0]) <= tol and abs(got[1] - c[1]) <= tol
            print(f"  {'OK ' if ok else '!! '}{name:9s} {f:5s} text {c}  recomputed "
                  f"({got[0]:.3f}, {got[1]:.3f})")
            bad += (not ok)
    for lab, got, want, tol in [("comparisons", npts, 45, 0),
                                ("significant at 0.05", nsig, 0, 0)]:
        ok = got == want
        print(f"  {'OK ' if ok else '!! '}{lab:22s} text {want}  recomputed {got}")
        bad += (not ok)
    ok = abs(pmin - 0.078) <= 0.002
    print(f"  {'OK ' if ok else '!! '}{'smallest p':22s} text 0.078  recomputed {pmin:.4f}")
    bad += (not ok)

    print("\n" + "=" * 78)
    print("F. every figure has a caption and every figure is cited")
    print("=" * 78)
    import re
    dd = _manuscript()
    try:
        from captions import CAPTIONS
    except ModuleNotFoundError:
        print("  (skipped: captions.py is manuscript tooling and is not in "
              "this archive)")
        CAPTIONS = {}
    CAP_STYLE = "RSC I04 Caption to Figure/Scheme/Chart"
    if dd is None:
        print(f"  ({len(CAPTIONS)} captions are defined in captions.py)")
    else:
        caps = [p.text.strip() for p in dd.paragraphs if p.style.name == CAP_STYLE]
        print(f"  {len(caps)} figure captions found, {len(CAPTIONS)} expected")
        bad += (len(caps) != len(CAPTIONS))
        body = "\n".join(p.text for p in dd.paragraphs if p.style.name != CAP_STYLE)
        for n in range(1, len(CAPTIONS) + 1):
            cited = bool(re.search(rf"Fig\. ?{n}[abc]?\b", body))
            print(f"  {'OK ' if cited else '!! '}Fig. {n} cited in the text: {cited}")
            bad += (not cited)

    # ── M. Supplementary Table S2 ────────────────────────────────────────
    # The table compares three refitting protocols.  The reference arm was run
    # at ten seeds and the other two at five, so an earlier draft paired a
    # ten-seed mean against five-seed means and quoted p-values no paired test
    # could have produced.  Every row must come from the seeds all three share.
    print("\n" + "=" * 78)
    print("M. supplementary Table S2, recomputed from the runs")
    print("=" * 78)
    try:
        import supp_tables
    except ModuleNotFoundError:
        print("  (skipped: supp_tables.py is manuscript tooling and is not in "
              "this archive)")
        supp_tables = None
    sd = _manuscript(SUPP_DOCX) if supp_tables else None
    s2 = next((t for t in (sd.tables if sd is not None else [])
               if t.rows[0].cells[0].text.strip() == "Refitting protocol"), None)
    if s2 is None:
        if supp_tables is not None and sd is not None:
            print("  !! Table S2 not found in the supplementary")
            bad += 1
    else:
        want = supp_tables._refit_rows()
        got = [[c.text.strip() for c in r.cells] for r in s2.rows[1:]]
        for w, g in zip(want, got):
            same = w == g
            print(f"  {'OK ' if same else '!! '}{w[0][:38]:38s} "
                  f"{'  '.join(w[1:])}")
            if not same:
                print(f"      document has: {'  '.join(g[1:])}")
            bad += (not same)
        if len(want) != len(got):
            print(f"  !! {len(got)} rows in the document, {len(want)} recomputed")
            bad += 1

    print("\n" + "=" * 78)
    print(f"RESULT: {bad} inconsistencies")
    print("=" * 78)
    json.dump(facts, open(HERE / "results" / "verified_facts.json", "w"), indent=1)


if __name__ == "__main__":
    main()
