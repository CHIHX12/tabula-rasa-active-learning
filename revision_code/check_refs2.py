# -*- coding: utf-8 -*-
"""Second pass: for the flagged references, print the top CrossRef candidates
(and, where a DOI is known, resolve it directly) so each can be adjudicated.
"""
from __future__ import annotations

import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from refs import REFS  # noqa: E402

UA = "ref-audit/1.0 (mailto:cycheng29721@gmail.com)"

# references to re-examine, with an optional exact-title override
TARGETS = {
    1: "Leveraging data mining, active learning, and domain adaptation for efficient discovery of advanced oxygen evolution electrocatalysts",
    5: "Machine learning for molecular and materials science",
    6: "Deep learning",
    17: "Sustainable Hydrogen Production",
    20: None,
    21: "Bilinear Attention Networks",
    23: "Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning",
    24: "Mixture Density Networks",
    25: "Adam: A Method for Stochastic Optimization",
    26: "Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles",
    27: "Active Learning",
    29: "Bayesian Optimization for Materials Design",
    30: "Gaussian Process Optimization in the Bandit Setting: No Regret and Experimental Design",
    31: "Diversified Sampling for Batched Bayesian Optimization with Determinantal Point Processes",
    32: "SGDR: Stochastic Gradient Descent with Warm Restarts",
    40: "Electrolysis of water on oxide surfaces",
    45: "Water adsorption and O-defect formation on Fe2O3(0001) surfaces",
    49: "Curiosity-driven Exploration by Self-supervised Prediction",
    50: "Exploration by Random Network Distillation",
    51: None,
    54: "Self-driving laboratory for accelerated discovery of thin-film materials",
    56: "An autonomous laboratory for the accelerated synthesis of novel materials",
    58: "High-entropy alloys",
    59: "Microstructural development in equiatomic multicomponent alloys",
    62: None,
}

DOIS = {
    20: "10.1039/D6RA02168A",
    51: "10.1039/D5DD00119F",
    1: "10.1126/sciadv.adr9038",
    5: "10.1038/s41586-018-0337-2",
    6: "10.1038/nature14539",
    17: "10.1126/science.1103197",
    54: "10.1126/sciadv.aaz8867",
    56: "10.1038/s41586-023-06734-w",
    58: "10.1038/s41578-019-0121-4",
    40: "10.1016/j.jelechem.2006.11.008",
    45: "10.1039/C6CP05313K",
}


def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def show(it, prefix="   "):
    t = (it.get("title") or [""])[0]
    c = (it.get("container-title") or [""])[0]
    a = [x.get("family", "") for x in (it.get("author") or [])]
    y = it.get("issued", {}).get("date-parts", [[None]])[0][0]
    print(f"{prefix}{t[:78]}")
    print(f"{prefix}  {c[:50]} | vol {it.get('volume')} | p {it.get('page')} | {y} "
          f"| {it.get('DOI')}")
    print(f"{prefix}  authors[{len(a)}]: {', '.join(a[:9])}")


def main():
    byn = {r[0]: r for r in REFS}
    for n in sorted(TARGETS):
        r = byn[n]
        print(f"\n===== [{n}] printed: {r[1]}")
        print(f"      \"{r[2]}\"")
        print(f"      {r[3]} {r[4]}, {r[5]} ({r[6]})")
        if n in DOIS:
            try:
                it = get(f"https://api.crossref.org/works/{urllib.parse.quote(DOIS[n])}")["message"]
                print("  -- DOI lookup:")
                show(it, "     ")
            except Exception as e:
                print(f"  -- DOI lookup FAILED {DOIS[n]}: {e}")
            time.sleep(0.5)
        q = TARGETS[n] or r[2]
        try:
            items = get("https://api.crossref.org/works?" + urllib.parse.urlencode(
                {"query.title": q, "rows": 3,
                 "select": "title,author,container-title,volume,page,issued,DOI"}))["message"]["items"]
            print("  -- title search:")
            for it in items:
                show(it, "     ")
        except Exception as e:
            print(f"  -- search FAILED: {e}")
        time.sleep(0.7)


if __name__ == "__main__":
    main()
