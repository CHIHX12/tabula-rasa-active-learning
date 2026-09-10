# -*- coding: utf-8 -*-
"""Regenerate every figure from the controlled runs.

Each figure is written twice: a vector PDF (what the journal should typeset
from) and a 1200 dpi PNG (fallback). Every panel that shows a mean carries an
error bar or a shaded band, and panel (a) of Fig. 2 shows the individual seed
trajectories rather than summarising them, because a shaded band cannot convey
how a handful of favourable seeds drive a cross-seed mean.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from analyze import load, per_seed  # noqa: E402

OUT = HERE / "figures"
OUT.mkdir(exist_ok=True)

# Okabe-Ito, safe for the common colour-vision deficiencies
C = dict(blue="#0072B2", orange="#E69F00", green="#009E73", red="#D55E00",
         purple="#CC79A7", sky="#56B4E9", yellow="#F0E442", grey="#666666")

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8.5, "axes.titlesize": 9,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7,
    "axes.linewidth": 0.7, "grid.linewidth": 0.4, "lines.linewidth": 1.4,
    "xtick.major.width": 0.7, "ytick.major.width": 0.7,
    "figure.dpi": 150, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "font.family": "DejaVu Sans", "axes.grid": True, "grid.alpha": 0.25,
    "axes.spines.top": False, "axes.spines.right": False,
})


def save(fig, name):
    """Vector PDF and 1200 dpi PNG for submission, 600 dpi PNG for embedding.

    RSC typesets from the vector file; the 1200 dpi raster is the fallback the
    submission system asks for; the 600 dpi copy is what goes inside the Word
    file, which keeps the document a reasonable size while staying well above
    the 300 dpi that would be visibly soft in the review PDF.
    """
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=1200)
    fig.savefig(OUT / f"{name}_600dpi.png", dpi=600)
    plt.close(fig)
    print(f"  {name}: pdf + png@1200 + png@600")


def merge(tags):
    """Merge several runs of the same configuration into one seed set."""
    seeds, best, pool, sig, oob, brk = [], [], [], [], [], []
    for t in tags:
        d = load(t)
        if d is None:
            continue
        s = per_seed(d)
        seeds += list(s["seed"]); best += list(s["best"])
        brk += list(s["break_exp"]); oob += list(s["oob_r2_final"])
        for r in d["runs"]:
            pool.append(r["history"]["pool_r2"]); sig.append(r["history"]["sigma"])
    if not seeds:
        return None
    return dict(seed=np.asarray(seeds), best=np.asarray(best, float),
                brk=np.asarray(brk, float), oob=np.asarray(oob, float),
                pool=np.asarray(pool, float), sigma=np.asarray(sig, float),
                curves=[load(t) for t in tags if load(t)])


def best_curves(tags):
    cur = []
    for t in tags:
        d = load(t)
        if d:
            cur += [r["history"]["best"] for r in d["runs"]]
    return np.asarray(cur, float)


# ════════════════════════════════════════════════════════════════
def _box(ax, x, y, w, h, text, fc, ec, fs=6.6, lw=0.8, weight=None):
    from matplotlib.patches import FancyBboxPatch
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.006,rounding_size=0.02",
                                fc=fc, ec=ec, lw=lw, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, zorder=3, linespacing=1.35, weight=weight)


def _arrow(ax, x0, y0, x1, y1, ec="#333333", lw=0.9, style="-|>", rad=0.0):
    from matplotlib.patches import FancyArrowPatch
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style,
                                 mutation_scale=7, lw=lw, color=ec, zorder=4,
                                 connectionstyle=f"arc3,rad={rad}",
                                 shrinkA=0, shrinkB=0))


def fig1():
    """Fig. 1 is the author's own workflow illustration, kept as drawn.

    It is copied from manuscript_figures/main/fig1.bmp into the figures folder
    in the same three forms as every other figure, so that the submission
    package is uniform.  It is a raster original (1302 x 525 px), so the PDF
    here wraps that raster rather than being true vector art; at the full text
    width of 7.07 in it prints at about 184 dpi.
    """
    from PIL import Image
    src = HERE.parent / "manuscript_figures" / "main" / "fig1.bmp"
    im = Image.open(src).convert("RGB")
    im.save(OUT / "Fig1_framework.png")
    im.save(OUT / "Fig1_framework_600dpi.png")
    # PIL's PDF writer needs a JPEG encoder that is not built into this
    # Pillow; wrap the raster with matplotlib instead, at its native pixel
    # size so that nothing is resampled.
    w_in = 7.07
    f = plt.figure(figsize=(w_in, w_in * im.size[1] / im.size[0]))
    a = f.add_axes([0, 0, 1, 1]); a.axis("off"); a.grid(False)
    a.imshow(np.asarray(im), interpolation="none", aspect="auto")
    f.savefig(OUT / "Fig1_framework.pdf", dpi=im.size[0] / w_in,
              bbox_inches=None, pad_inches=0)
    plt.close(f)
    print(f"  Fig1_framework: copied from {src.name} ({im.size[0]}x{im.size[1]} px, "
          f"{im.size[0] / 7.07:.0f} dpi at full width)")


# ════════════════════════════════════════════════════════════════
def fig2():
    """OER convergence on the common metric, with the seeds shown."""
    arms = [("LCBDS + max-σ", ["E1_full"], C["blue"]),
            ("Greedy", ["E1_greedy"], C["red"]),
            ("Random", ["E1_random"], C["green"]),
            ("GP-BO (EI)", ["E2_gp_ei"], C["orange"])]
    fig, ax = plt.subplots(1, 4, figsize=(7.2, 2.35))

    n_best = 20 + np.arange(101) * 2
    n_r2 = 20 + np.arange(1, 101) * 2
    for lab, tags, col in arms:
        cur = best_curves(tags)
        for c in cur:                       # every seed, not just the band
            ax[0].plot(n_best, c, color=col, lw=0.45, alpha=0.28, zorder=1)
        ax[0].plot(n_best, cur.mean(0), color=col, lw=1.8, label=lab, zorder=3)
    ax[0].axhline(370, color=C["grey"], ls="--", lw=0.8)
    ax[0].text(23, 373.5, "370 mV", fontsize=6.5, color=C["grey"],
               ha="left", va="bottom",
               bbox=dict(fc="white", ec="none", alpha=0.85, pad=0.9))
    ax[0].set_xlabel("experiments"); ax[0].set_ylabel("best $J_{10}$ found (mV)")
    ax[0].set_title("(a) optimization", loc="left", fontsize=8)
    ax[0].set_ylim(190, 405)

    for lab, tags, col in arms:
        m = merge(tags)
        mu, sd = np.nanmean(m["pool"], 0), np.nanstd(m["pool"], 0)
        ax[1].plot(n_r2, mu, color=col, lw=1.6, label=lab)
        ax[1].fill_between(n_r2, mu - sd, mu + sd, color=col, alpha=0.16, lw=0)
    ax[1].axhline(0, color="k", lw=0.7)
    ax[1].set_xlabel("experiments")
    ax[1].set_ylabel("out-of-pool $R^2$")
    ax[1].set_title("(b) surrogate quality", loc="left", fontsize=8)
    ax[1].set_ylim(-1.1, 0.9)

    for lab, tags, col in arms:
        m = merge(tags)
        mu, sd = np.nanmean(m["sigma"], 0), np.nanstd(m["sigma"], 0)
        ax[2].plot(n_r2, mu, color=col, lw=1.6)
        ax[2].fill_between(n_r2, np.clip(mu - sd, 0, None), mu + sd,
                           color=col, alpha=0.16, lw=0)
    ax[2].set_xlabel("experiments")
    ax[2].set_ylabel(r"mean predictive $\sigma$ (mV)")
    ax[2].set_title("(c) predictive spread", loc="left", fontsize=8)

    # (d) the same quantity, as a diagnostic: how far the pool-average
    # predictive uncertainty falls over a campaign orders the configurations
    # the way out-of-pool R2 does, without any held-out data.
    diag = [("E1_random", C["green"]), ("E1_lcbds_rand2", C["purple"]),
            ("E2_pcban_ei", C["sky"]), ("E1_nosurp", C["blue"]),
            ("E2_pcban_ei_maxsigma", C["sky"]), ("E1_full", C["blue"]),
            ("E1_noanneal", C["blue"]), ("E1_nodivsurp", C["blue"]),
            ("E1_greedy_maxsigma", C["red"]), ("E1_nodiv", C["blue"]),
            ("E1_lcbds_x2", C["orange"]), ("E1_pureLCB_matched", C["orange"]),
            ("E1_Aonly_matched", C["orange"]), ("E1_greedy", C["red"])]
    for tag, col in diag:
        d = load(tag)
        if d is None:
            continue
        ra = np.nanmean([np.nanmean(r["history"]["sigma"][-10:])
                         / np.nanmean(r["history"]["sigma"][:5]) for r in d["runs"]])
        pr = np.nanmean([np.nanmean(r["history"]["pool_r2"][-10:]) for r in d["runs"]])
        ax[3].scatter(ra, pr, s=26, color=col, zorder=3, lw=0.5, edgecolor="white")
    ax[3].axvspan(0.66, 0.78, color=C["grey"], alpha=0.12, lw=0)
    ax[3].set_xlabel(r"predictive $\hat\sigma$: final / initial")
    ax[3].set_ylabel("out-of-pool $R^2$")
    ax[3].set_title("(d) held-out-free diagnostic", loc="left", fontsize=8)
    ax[3].set_xlim(0.45, 0.95); ax[3].set_ylim(0.25, 0.70)

    handles = [Line2D([], [], color=c, lw=2.2, label=l)
               for l, _, c in arms]
    fig.legend(handles=handles, frameon=False, ncol=4, fontsize=7.4,
               loc="lower center", bbox_to_anchor=(0.5, -0.055),
               handlelength=1.8, columnspacing=2.0)
    fig.tight_layout(w_pad=1.5)
    save(fig, "Fig2_convergence")


# ════════════════════════════════════════════════════════════════
def fig3():
    """Budget-matched component ablation, error bars on every panel."""
    arms = [("LCBDS + max-σ (full)", "E1_full", C["blue"]),
            ("w/o MC-Dropout", "E1_nomc", C["sky"]),
            ("w/o β annealing", "E1_noanneal", C["sky"]),
            ("w/o diversity γ", "E1_nodiv", C["sky"]),
            ("w/o surprise δ", "E1_nosurp", C["sky"]),
            ("w/o γ and δ", "E1_nodivsurp", C["sky"]),
            ("max-σ → random", "E1_lcbds_rand2", C["purple"]),
            ("w/o max-σ (2×LCBDS)", "E1_lcbds_x2", C["orange"]),
            ("w/o max-σ (1 query)", "E1_Aonly_matched", C["orange"]),
            ("pure LCB (1 query)", "E1_pureLCB_matched", C["orange"]),
            ("greedy", "E1_greedy", C["red"]),
            ("random", "E1_random", C["green"])]
    labs = [a[0] for a in arms]
    M = [merge([a[1]]) for a in arms]
    cols = [a[2] for a in arms]
    y = np.arange(len(arms))[::-1]

    fig, ax = plt.subplots(1, 3, figsize=(7.2, 3.3), sharey=True)
    for k, (field, xlabel, title) in enumerate([
            ("brk", "experiments to $J_{10}$ < 370 mV", "(a) discovery efficiency"),
            ("best", "best $J_{10}$ found (mV)", "(b) optimization depth"),
            ("pool", "out-of-pool $R^2$", "(c) surrogate quality")]):
        for i, (m, col) in enumerate(zip(M, cols)):
            v = np.nanmean(m["pool"][:, -10:], axis=1) if field == "pool" else m[field]
            ax[k].barh(y[i], np.nanmean(v), xerr=np.nanstd(v), color=col,
                       alpha=0.85, height=0.66, error_kw=dict(lw=0.8, capsize=2))
        ax[k].set_xlabel(xlabel)
        ax[k].set_title(title, loc="left")
    ax[0].set_yticks(y); ax[0].set_yticklabels(labs)
    v = np.nanmean(M[0]["pool"][:, -10:], axis=1)
    ax[2].axvline(np.nanmean(v), color=C["blue"], ls=":", lw=0.9)
    handles = [Line2D([], [], color=c, lw=6, alpha=.85, label=l) for c, l in
               [(C["blue"], "full protocol"), (C["sky"], "one acquisition term removed"),
                (C["purple"], "exploration query kept, rule changed"),
                (C["orange"], "no non-exploitative query"),
                (C["red"], "purely exploitative"), (C["green"], "passive")]]
    fig.legend(handles=handles, frameon=False, ncol=3, fontsize=7.2,
               loc="lower center", bbox_to_anchor=(0.5, -0.15),
               columnspacing=2.4, handlelength=1.8)
    fig.tight_layout(w_pad=1.6)
    save(fig, "Fig3_ablation")


# ════════════════════════════════════════════════════════════════
def fig4():
    """Protocol vs surrogate, and the deep-optimum hit rate."""
    fig = plt.figure(figsize=(7.2, 5.35))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.05],
                          hspace=0.72, wspace=0.36,
                          left=0.085, right=0.985, top=0.945, bottom=0.175)

    # (a) the protocol effect, across surrogates and acquisition rules
    axa = fig.add_subplot(gs[0, :])
    groups = [("GP-BO\nEI", "E2_gp_ei", C["orange"], False),
              ("GP-BO\nEI + max-var", "E2_gp_ei_maxsigma", C["orange"], True),
              ("GP\nLCB", "E2_gp_lcb", C["sky"], False),
              ("GP\nLCB + max-var", "E2_gp_lcb_maxsigma", C["sky"], True),
              ("PC-BAN\ngreedy", "E1_greedy", C["red"], False),
              ("PC-BAN\ngreedy + max-σ", "E1_greedy_maxsigma", C["red"], True),
              ("PC-BAN\nLCBDS + max-σ", "E1_full", C["blue"], True)]
    for i, (lab, tag, col, matched) in enumerate(groups):
        m = merge([tag]); v = np.nanmean(m["pool"][:, -10:], axis=1)
        axa.bar(i, np.nanmean(v), yerr=np.nanstd(v), color=col,
                alpha=0.92 if matched else 0.40, width=0.66,
                hatch="" if matched else "///", edgecolor=col,
                error_kw=dict(lw=0.8, capsize=2.5))
    axa.axhline(0, color="k", lw=0.8)
    axa.set_xticks(range(len(groups)))
    axa.set_xticklabels([g[0] for g in groups], fontsize=6.6)
    axa.set_ylabel("out-of-pool $R^2$")
    axa.set_title("(a) a single non-exploitative query decides whether the surrogate survives",
                  loc="left", pad=6)
    axa.set_ylim(-0.95, 0.80)
    axa.legend(handles=[Line2D([], [], color=C["grey"], lw=7, alpha=.40,
                               label="every query exploitative"),
                        Line2D([], [], color=C["grey"], lw=7, alpha=.92,
                               label="one query non-exploitative")],
               frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.30),
               fontsize=7, ncol=2, columnspacing=2.4, handlelength=1.6)

    # (b) surrogate held against the protocol
    axb = fig.add_subplot(gs[1, 0])
    sur = [("PC-BAN", "E1_full", C["blue"]),
           ("random forest", "E3_rf", C["green"]),
           ("Gaussian process", "E3_gp_lcbds", C["orange"]),
           ("plain MLP", "E3_mlp", C["purple"]),
           ("MLP + descriptors", "E3_mlpdesc", C["red"])]
    for i, (lab, tag, col) in enumerate(sur):
        m = merge([tag]); v = np.nanmean(m["pool"][:, -10:], axis=1)
        axb.barh(len(sur) - 1 - i, np.nanmean(v), xerr=np.nanstd(v), color=col,
                 alpha=0.88, height=0.62, error_kw=dict(lw=0.8, capsize=2.5))
    axb.axvline(0, color="k", lw=0.8)
    axb.set_yticks(range(len(sur))[::-1])
    axb.set_yticklabels([s_[0] for s_ in sur], fontsize=7)
    axb.set_xlabel("out-of-pool $R^2$")
    axb.set_title("(b) same protocol, different surrogate", loc="left", pad=6)
    axb.set_xlim(-0.62, 0.74)

    # (c) deep-optimum hit rate at both budgets
    axc = fig.add_subplot(gs[1, 1])
    thr = [300, 260, 210]
    series = [("220 exp: LCBDS + max-σ", ["E1_full", "P_lcbds"], C["blue"]),
              ("220 exp: PC-BAN + EI", ["E2_pcban_ei", "P_ei"], C["sky"]),
              ("220 exp: random", ["E1_random"], C["green"]),
              ("470 exp: Bold protocol", ["B_lcbds_bold", "BP_lcbds_bold"], C["purple"]),
              ("470 exp: EI + max-σ + max-μ", ["B_ei_bold", "BP_ei_bold"], C["orange"])]
    w = 0.16
    for j, (lab, tags, col) in enumerate(series):
        m = merge(tags)
        rate = [(m["best"] <= t).mean() for t in thr]
        axc.bar(np.arange(len(thr)) + (j - 2) * w, rate, width=w, color=col,
                alpha=0.9, label=f"{lab}  (n={len(m['best'])})")
    axc.set_xticks(range(len(thr)))
    axc.set_xticklabels([f"$\\leq${t} mV" for t in thr], fontsize=7)
    axc.set_ylabel("fraction of seeds reaching it")
    axc.set_title("(c) reaching deep optima", loc="left", pad=6)
    axc.set_ylim(0, 0.78)
    hc, lc = axc.get_legend_handles_labels()
    fig.legend(hc, lc, frameon=False, ncol=3, fontsize=7,
               loc="lower center", bbox_to_anchor=(0.5, -0.005),
               columnspacing=1.8, handlelength=1.4,
               title="(c) configurations", title_fontsize=7)
    save(fig, "Fig4_protocol_surrogate")


# ════════════════════════════════════════════════════════════════
def fig5():
    """Why the acquisition terms cannot act."""
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.65))

    it = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 99])
    bsig = np.array([14.47, 12.00, 11.27, 9.65, 8.18, 6.44, 5.35, 3.79, 2.88, 1.68, 0.70])
    gdiv = np.array([3.76, 3.50, 3.48, 3.44, 3.39, 3.29, 3.29, 3.24, 3.23, 3.20, 3.18])
    dsur = np.zeros_like(bsig)
    muspread = np.array([6.80, 8.60, 10.57, 10.74, 11.55, 11.74, 12.06, 11.61, 11.52, 11.44, 11.41])
    n = 20 + it * 2
    ax[0].plot(n, muspread, color="k", lw=1.8, label=r"s.d. of $\hat\mu$ over the pool")
    ax[0].plot(n, bsig, color=C["blue"], lw=1.5, label=r"$\beta_t\hat\sigma$")
    ax[0].plot(n, gdiv, color=C["green"], lw=1.5, label=r"$\gamma\,d_{norm}$")
    ax[0].plot(n, dsur, color=C["red"], lw=2.2, label=r"$\delta\,s(x)$  (identically 0)")
    ax[0].set_xlabel("experiments")
    ax[0].set_ylabel("contribution to the score (mV)")
    ax[0].set_title("(a) size of each term", loc="left", fontsize=8)
    ax[0].set_ylim(-0.8, 16.5)

    names = ["PC-BAN", "random\nforest", "Gaussian\nprocess", "plain\nMLP"]
    pred = [416.9, 423.1, 413.3, 411.0]
    ax[1].bar(range(4), pred, color=[C["blue"], C["green"], C["orange"], C["purple"]],
              alpha=0.85, width=0.66)
    ax[1].axhline(206, color=C["red"], ls="--", lw=1.2)
    ax[1].text(3.45, 216, "true value\n206 mV", fontsize=6.2, color=C["red"], ha="right")
    ax[1].set_xticks(range(4)); ax[1].set_xticklabels(names, fontsize=6.4)
    ax[1].set_ylabel("predicted $J_{10}$ at the optimum (mV)", fontsize=7.6)
    ax[1].set_title("(b) the optimum stays mispredicted", loc="left", fontsize=8)
    ax[1].set_ylim(180, 470)

    lbl = ["PC-BAN", "random\nforest", "Gaussian\nprocess", "random\nshortlist"]
    rec = [0.59, 0.53, 0.35, 0.04]
    ax[2].bar(range(4), rec, color=[C["blue"], C["green"], C["orange"], C["grey"]],
              alpha=0.85, width=0.66)
    ax[2].set_xticks(range(4)); ax[2].set_xticklabels(lbl, fontsize=6.4)
    ax[2].set_ylabel("recall of the 17 sub-370 mV points", fontsize=7.6)
    ax[2].set_title("(c) the region is found", loc="left", fontsize=8)
    ax[2].set_ylim(0, 0.72)
    h0, l0 = ax[0].get_legend_handles_labels()
    fig.tight_layout(w_pad=2.0)
    fig.legend(h0, l0, frameon=False, ncol=4, fontsize=7,
               loc="lower center", bbox_to_anchor=(0.5, -0.12),
               columnspacing=1.8, handlelength=1.6,
               title="(a) terms of the LCBDS score", title_fontsize=7)
    save(fig, "Fig5_diagnostics")


# ════════════════════════════════════════════════════════════════
def fig6():
    """Hyperparameter sensitivity, re-run through the controlled harness.

    The submitted Section 3.4 quoted a five-seed sweep produced by the original
    pipeline. Every point here is regenerated by revision/al_harness.py on the
    same pool, the same K-means initial design, the same ten seeds and the same
    220-experiment budget as every other result in the paper, and is scored on
    the same out-of-pool metric.
    """
    grid = [
        (r"exploration floor  $\beta_{min}$",
         [("0.05", "H_beta005"), ("0.1", "H_beta010"), ("0.2", "E1_full"),
          ("0.3", "H_beta030"), ("0.5", "H_beta050"), ("1.0", "H_beta100")], "0.2"),
        (r"diversity weight  $\gamma$ (mV)",
         [("0", "E1_nodiv"), ("3", "H_gam03"), ("5", "H_gam05"), ("8", "E1_full"),
          ("10", "H_gam10"), ("15", "H_gam15")], "8"),
        (r"surprise weight  $\delta$ (mV)",
         [("0", "E1_nosurp"), ("5", "H_del05"), ("8", "H_del08"), ("12", "E1_full"),
          ("15", "H_del15"), ("20", "H_del20")], "12"),
    ]
    # sharey="row" matters: with three independent y-scales the same flat
    # sweep looks flat in one column and noisy in the next, purely from the
    # zoom level. One scale per row makes the columns directly comparable.
    fig, ax = plt.subplots(3, 3, figsize=(7.2, 4.6), sharex="col", sharey="row")
    rows = [("brk", "queries to\n$J_{10}$ < 370 mV"),
            ("best", "best $J_{10}$\nfound (mV)"),
            ("pool", "out-of-pool $R^2$")]
    missing = []
    for j, (xlabel, points, default) in enumerate(grid):
        labs, M = [], []
        for lab, tag in points:
            m = merge([tag])
            if m is None:
                missing.append(tag); continue
            labs.append(lab); M.append(m)
        x = np.arange(len(labs))
        for i, (field, ylabel) in enumerate(rows):
            a = ax[i, j]
            for k, m in enumerate(M):
                v = (np.nanmean(m["pool"][:, -10:], axis=1) if field == "pool"
                     else m[field])
                is_def = labs[k] == default
                a.errorbar(x[k], np.nanmean(v), yerr=np.nanstd(v), fmt="o",
                           ms=4.2 if is_def else 3.4, lw=0.9, capsize=2.2,
                           color=C["blue"] if is_def else C["grey"],
                           mfc=C["blue"] if is_def else "white", zorder=3)
            if M:
                ref = (np.nanmean(M[labs.index(default)]["pool"][:, -10:], axis=1)
                       if field == "pool" else M[labs.index(default)][field])
                a.axhline(np.nanmean(ref), color=C["blue"], ls=":", lw=0.8, zorder=1)
            a.set_xticks(x); a.set_xticklabels(labs, fontsize=6.8)
            if j == 0:
                a.set_ylabel(ylabel, fontsize=7.2, linespacing=1.3)
            if i == 2:
                a.set_xlabel(xlabel, fontsize=7.6)
        ax[0, j].set_title(f"({'abc'[j]}) {xlabel.split('  ')[0]}",
                           loc="left", fontsize=8)
    handles = [Line2D([], [], color=C["blue"], marker="o", ms=4.2, lw=0,
                      label="value used throughout the paper"),
               Line2D([], [], color=C["grey"], marker="o", ms=3.4, lw=0, mfc="white",
                      label="swept value")]
    fig.legend(handles=handles, frameon=False, ncol=2, fontsize=7.4,
               loc="lower center", bbox_to_anchor=(0.5, -0.045),
               columnspacing=2.4, handlelength=1.6)
    fig.tight_layout(h_pad=0.7, w_pad=1.4)
    save(fig, "Fig6_hyperparameter")
    if missing:
        print("    (still running, not yet plotted: " + ", ".join(missing) + ")")
    return missing


# ════════════════════════════════════════════════════════════════
def fig7():
    """Cross-domain, unified code path, n = 10."""
    DS = [("steel", "Steel alloys\n$N$ = 13", "max"),
          ("concrete", "Concrete strength\n$N$ = 8", "max"),
          ("slump", "Concrete slump\n$N$ = 7", "max"),
          ("wine_red", "Wine quality\n$N$ = 11", "max"),
          ("energy", "Energy efficiency\n$N$ = 8", "min")]
    arms = [("LCBDS + max-σ", "lcbds", C["blue"]),
            ("Greedy + max-σ", "grd_ms", C["purple"]),
            ("Greedy", "greedy", C["red"]),
            ("GP-EI + max-var", "gpei_ms", C["orange"]),
            ("Random", "random", C["green"])]
    fig, ax = plt.subplots(2, 5, figsize=(7.2, 3.5), sharex="col")
    for j, (key, title, direction) in enumerate(DS):
        for lab, a, col in arms:
            tags = [f"G_{key}_{a}", f"GP_{key}_{a}"]
            cur = best_curves(tags)
            if cur.size == 0:
                continue
            m = merge(tags)
            x = np.arange(cur.shape[1])
            mu, sd = cur.mean(0), cur.std(0)
            ax[0, j].plot(x, mu, color=col, lw=1.2, label=lab)
            ax[0, j].fill_between(x, mu - sd, mu + sd, color=col, alpha=0.13, lw=0)
            p = m["pool"]
            xp = np.arange(p.shape[1])
            pm, ps = np.nanmean(p, 0), np.nanstd(p, 0)
            ax[1, j].plot(xp, pm, color=col, lw=1.2)
            ax[1, j].fill_between(xp, pm - ps, pm + ps, color=col, alpha=0.13, lw=0)
        ax[0, j].set_title(title, fontsize=7, linespacing=1.25)
        ax[1, j].axhline(0, color="k", lw=0.7)
        ax[1, j].set_xlabel("iteration")
        if j == 0:
            ax[0, j].set_ylabel("best value found")
            ax[1, j].set_ylabel("out-of-pool $R^2$")
    handles = [Line2D([], [], color=c, lw=1.6, label=l) for l, _, c in arms]
    fig.legend(handles=handles, frameon=False, ncol=5, fontsize=7.4,
               loc="lower center", bbox_to_anchor=(0.5, -0.06),
               handlelength=1.8, columnspacing=2.0)
    fig.tight_layout(h_pad=0.8, w_pad=1.2)
    save(fig, "Fig7_cross_domain")


if __name__ == "__main__":
    print("writing figures to", OUT)
    fig1(); fig2(); fig3(); fig4(); fig5(); fig6(); fig7()
