# -*- coding: utf-8 -*-
"""Rebuild Table 2 from the unified-code-path cross-domain re-run."""
from __future__ import annotations
import copy, json
from pathlib import Path
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
import table1 as T1

HEADERS = ["Dataset (N, pool)",
           "global optimum recovered\nLCBDS+maxσ / Greedy+maxσ / GP-EI+maxσ / Greedy / Random",
           "out-of-pool R²\nsame order",
           "p  LCBDS vs Greedy"]
WIDTHS = [1500, 1500, 1400, 700]


def rebuild(table, tr, data_path="revision/table2_rows.json"):
    rows = json.load(open(Path(data_path), encoding="utf-8"))
    tbl = table._tbl
    grid = tbl.find(qn("w:tblGrid"))
    cols = grid.findall(qn("w:gridCol"))
    # the source table has 5 columns; we need 4 -> drop one and rescale
    while len(grid.findall(qn("w:gridCol"))) > len(WIDTHS):
        grid.remove(grid.findall(qn("w:gridCol"))[-1])
    for gc, w in zip(grid.findall(qn("w:gridCol")), WIDTHS):
        gc.set(qn("w:w"), str(w))

    trs = tbl.findall(qn("w:tr"))
    header_tr, data_trs = trs[0], trs[1:]
    template = data_trs[-1]
    while len(data_trs) < len(rows):
        new = copy.deepcopy(template)
        trPr = new.find(qn("w:trPr")) or OxmlElement("w:trPr")
        if new.find(qn("w:trPr")) is None:
            new.insert(0, trPr)
        trPr.append(tr._mark("ins"))
        for tc in new.findall(qn("w:tc")):
            T1._clear_cell(tc)
        data_trs[-1].addnext(new)
        data_trs.append(new)
    while len(data_trs) > len(rows):
        tbl.remove(data_trs.pop())

    for row_el in [header_tr] + data_trs:
        tcs = row_el.findall(qn("w:tc"))
        while len(tcs) > len(WIDTHS):
            row_el.remove(tcs.pop())
        for tc, w in zip(tcs, WIDTHS):
            tcPr = tc.find(qn("w:tcPr"))
            if tcPr is not None:
                el = tcPr.find(qn("w:tcW"))
                if el is not None:
                    el.set(qn("w:w"), str(w))

    tmpl = T1._first_run(header_tr.findall(qn("w:tc"))[0])
    for tc, text in zip(header_tr.findall(qn("w:tc")), HEADERS):
        T1._set_cell(tc, text.replace("\n", "  "), tr, tmpl)
    for row_el, vals in zip(data_trs, rows):
        tcs = row_el.findall(qn("w:tc"))
        rt = T1._first_run(tcs[0]) or tmpl
        for tc, text in zip(tcs, vals):
            T1._set_cell(tc, text, tr, rt)
    return len(data_trs), len(WIDTHS)
