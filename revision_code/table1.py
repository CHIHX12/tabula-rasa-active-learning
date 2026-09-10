# -*- coding: utf-8 -*-
"""Rebuild Table 1 (the component ablation) from the controlled re-runs.

The published table had nine arms, four columns, and reported the bootstrap
out-of-bag R^2; three of its arms silently used a 120-label budget instead of
220.  The rebuilt table has twelve budget-matched arms, reports the out-of-pool
R^2 as the primary metric with the out-of-bag value in parentheses, and adds a
column of paired significance tests against the full protocol.

Rows are added and the new column is created with Word revision marks so the
change is visible in the tracked and highlighted copies.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

from docx.oxml import OxmlElement
from docx.oxml.ns import qn

M_NS = "{http://schemas.openxmlformats.org/officeDocument/2006/math}"

HEADERS = ["Method", "break_exp ↓", "Best J10 (mV) ↓",
           "R²pool ↑ (OOB R²)", "p vs full"]
# twips; the original four columns summed to 4979 and we keep that total
WIDTHS = [1300, 850, 900, 1229, 700]


def _clear_cell(tc):
    """Strip every paragraph but the first, and empty that one."""
    ps = tc.findall(qn("w:p"))
    for extra in ps[1:]:
        tc.remove(extra)
    p = ps[0] if ps else None
    if p is None:
        p = OxmlElement("w:p")
        tc.append(p)
    for child in list(p):
        if child.tag in (qn("w:r"), M_NS + "oMath", M_NS + "oMathPara"):
            p.remove(child)
    return p


def _cell_text(tc):
    out = []
    for p in tc.findall(qn("w:p")):
        for r in p.findall(qn("w:r")):
            out.append("".join(t.text or "" for t in r.findall(qn("w:t"))))
    return "".join(out)


def _set_cell(tc, text, tr, template_run=None):
    """Replace the cell's content with `text`, marked as a tracked insertion."""
    old = _cell_text(tc)
    p = _clear_cell(tc)

    if old:
        dele = tr._mark("del")
        r_del = OxmlElement("w:r")
        if template_run is not None:
            rPr = template_run.find(qn("w:rPr"))
            if rPr is not None:
                r_del.append(copy.deepcopy(rPr))
        dt = OxmlElement("w:delText")
        dt.text = old
        dt.set(qn("xml:space"), "preserve")
        r_del.append(dt)
        dele.append(r_del)
        p.append(dele)

    ins = tr._mark("ins")
    r = OxmlElement("w:r")
    if template_run is not None:
        rPr = template_run.find(qn("w:rPr"))
        if rPr is not None:
            r.append(copy.deepcopy(rPr))
    t = OxmlElement("w:t")
    t.text = text
    t.set(qn("xml:space"), "preserve")
    r.append(t)
    ins.append(r)
    p.append(ins)


def _first_run(tc):
    for p in tc.findall(qn("w:p")):
        r = p.find(qn("w:r"))
        if r is not None:
            return r
    return None


def rebuild(table, tr, data_path="revision/table1_data.json"):
    rows = json.load(open(Path(data_path), encoding="utf-8"))
    tbl = table._tbl
    n_data_needed = len(rows)

    # ── 1. widen the grid by one column ───────────────────────────────
    grid = tbl.find(qn("w:tblGrid"))
    cols = grid.findall(qn("w:gridCol"))
    for gc, w in zip(cols, WIDTHS):
        gc.set(qn("w:w"), str(w))
    new_gc = OxmlElement("w:gridCol")
    new_gc.set(qn("w:w"), str(WIDTHS[-1]))
    grid.append(new_gc)

    # ── 2. add rows so that there are exactly len(rows) data rows ─────
    trs = tbl.findall(qn("w:tr"))
    header_tr, data_trs = trs[0], trs[1:]
    template_tr = data_trs[-1]
    while len(data_trs) < n_data_needed:
        new_tr = copy.deepcopy(template_tr)
        # mark the whole row as inserted
        trPr = new_tr.find(qn("w:trPr"))
        if trPr is None:
            trPr = OxmlElement("w:trPr")
            new_tr.insert(0, trPr)
        trPr.append(tr._mark("ins"))
        for tc in new_tr.findall(qn("w:tc")):
            _clear_cell(tc)
        data_trs[-1].addnext(new_tr)
        data_trs.append(new_tr)

    # ── 3. give every row a fifth cell ────────────────────────────────
    for row_el in [header_tr] + data_trs:
        tcs = row_el.findall(qn("w:tc"))
        for tc, w in zip(tcs, WIDTHS):
            tcW = tc.find(qn("w:tcPr") + "/" + qn("w:tcW")) if False else None
            tcPr = tc.find(qn("w:tcPr"))
            if tcPr is not None:
                el = tcPr.find(qn("w:tcW"))
                if el is not None:
                    el.set(qn("w:w"), str(w))
        new_tc = copy.deepcopy(tcs[-1])
        tcPr = new_tc.find(qn("w:tcPr"))
        if tcPr is not None:
            el = tcPr.find(qn("w:tcW"))
            if el is not None:
                el.set(qn("w:w"), str(WIDTHS[-1]))
        _clear_cell(new_tc)
        tcs[-1].addnext(new_tc)

    # ── 4. fill everything ────────────────────────────────────────────
    tmpl_run = _first_run(header_tr.findall(qn("w:tc"))[0])
    for tc, text in zip(header_tr.findall(qn("w:tc")), HEADERS):
        _set_cell(tc, text, tr, tmpl_run)

    for row_el, values in zip(data_trs, rows):
        tcs = row_el.findall(qn("w:tc"))
        run_tmpl = _first_run(tcs[0]) or tmpl_run
        for tc, text in zip(tcs, values):
            _set_cell(tc, text, tr, run_tmpl)

    return len(data_trs), len(HEADERS)
