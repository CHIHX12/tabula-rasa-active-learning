# -*- coding: utf-8 -*-
"""Verify every reference against CrossRef.

For each entry we query CrossRef by title, take the best title match, and
compare container, volume, pages, year and the FIRST AUTHOR SURNAME.
Anything that disagrees is printed as a flag for manual follow-up.
"""
from __future__ import annotations

import difflib
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from refs import REFS  # noqa: E402

UA = "ref-audit/1.0 (mailto:cycheng29721@gmail.com)"
OUT = Path(__file__).resolve().parent / "results" / "ref_audit.json"


def norm(s):
    s = (s or "").lower()
    s = s.replace("–", "-").replace("—", "-").replace("’", "'")
    return re.sub(r"[^a-z0-9 ]+", " ", s).strip()


def sim(a, b):
    return difflib.SequenceMatcher(None, norm(a), norm(b)).ratio()


def crossref(title, rows=5):
    q = urllib.parse.urlencode({
        "query.bibliographic": title, "rows": rows,
        "select": "title,author,container-title,volume,page,issued,DOI,type",
    })
    url = f"https://api.crossref.org/works?{q}"
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.load(r)["message"]["items"]
        except Exception as e:
            if attempt == 2:
                return [{"_error": str(e)}]
            time.sleep(3)
    return []


def first_surname(printed):
    m = re.match(r"\s*([A-Za-zÀ-ÿ'\-]+)", printed)
    return m.group(1).lower() if m else ""


def page_first(p):
    m = re.match(r"\s*(\w+)", (p or "").replace("–", "-"))
    return m.group(1).lower() if m else ""


def main():
    report = []
    for (num, authors, title, container, vol, pages, year, kind) in REFS:
        items = crossref(title)
        best, bs = None, 0.0
        for it in items:
            if "_error" in it:
                best, bs = it, 0.0
                break
            t = (it.get("title") or [""])[0]
            s = sim(title, t)
            if s > bs:
                best, bs = it, s
        rec = dict(num=num, printed_authors=authors, printed_title=title,
                   printed_container=container, printed_volume=vol,
                   printed_pages=pages, printed_year=year, kind=kind,
                   title_sim=round(bs, 3), flags=[])
        if best is None or "_error" in (best or {}):
            rec["flags"].append("NO_CROSSREF_RESULT")
            report.append(rec)
            print(f"[{num:2d}] ??? no crossref result", flush=True)
            continue

        cr_title = (best.get("title") or [""])[0]
        cr_cont = (best.get("container-title") or [""])[0]
        cr_vol = best.get("volume", "")
        cr_page = best.get("page", "")
        cr_year = (best.get("issued", {}).get("date-parts", [[None]])[0][0])
        cr_auth = best.get("author", []) or []
        cr_first = (cr_auth[0].get("family", "") if cr_auth else "").lower()
        rec.update(cr_title=cr_title, cr_container=cr_cont, cr_volume=cr_vol,
                   cr_pages=cr_page, cr_year=cr_year, cr_doi=best.get("DOI"),
                   cr_first_author=cr_first,
                   cr_authors=[f"{a.get('family','')}" for a in cr_auth][:12],
                   cr_n_authors=len(cr_auth))

        if bs < 0.85:
            rec["flags"].append(f"TITLE_MISMATCH({bs:.2f})")
        else:
            if vol and cr_vol and str(vol) != str(cr_vol):
                rec["flags"].append(f"VOLUME printed={vol} crossref={cr_vol}")
            if cr_year and year and abs(int(cr_year) - int(year)) > 1:
                rec["flags"].append(f"YEAR printed={year} crossref={cr_year}")
            pf, cf = page_first(pages), page_first(cr_page)
            if pf and cf and pf != cf and not pages.startswith("arXiv"):
                rec["flags"].append(f"PAGES printed={pages} crossref={cr_page}")
            fs = first_surname(authors)
            if fs and cr_first and fs != cr_first and fs not in cr_first:
                rec["flags"].append(
                    f"FIRST_AUTHOR printed={fs} crossref={cr_first}")
            if container and cr_cont and sim(container, cr_cont) < 0.5:
                # abbreviations differ a lot; only a soft note
                rec["flags"].append(f"CONTAINER? printed={container} crossref={cr_cont}")
        report.append(rec)
        st = "OK " if not rec["flags"] else "!! "
        print(f"[{num:2d}] {st}sim={bs:.2f} | {cr_cont[:34]:34s} v{str(cr_vol):>5s} "
              f"p{str(cr_page)[:14]:14s} {cr_year} | {', '.join(rec['flags'])[:110]}",
              flush=True)
        time.sleep(0.6)

    OUT.parent.mkdir(exist_ok=True)
    json.dump(report, open(OUT, "w"), indent=1, ensure_ascii=False)
    bad = [r for r in report if r["flags"]]
    print(f"\n{len(bad)}/{len(report)} entries flagged -> {OUT}")


if __name__ == "__main__":
    main()
