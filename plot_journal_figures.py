"""
Journal-quality figures — all English, no text overlap, proper proportions.
Output: figures/
"""
import json
import warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

warnings.filterwarnings("ignore")

ROOT    = Path(__file__).parent
# Use fresh results if available, otherwise fall back to precomputed results
_results_dir      = ROOT / "results"
_precomputed_dir  = ROOT / "precomputed_results"
RES_DIR = _results_dir if _results_dir.exists() and any(_results_dir.glob("*.json")) else _precomputed_dir
OUT_DIR = ROOT / "figures"
OUT_DIR.mkdir(exist_ok=True)

# ── Journal style (Nature/RSC/JPCB compatible)
plt.rcParams.update({
    "font.family"        : "Liberation Sans",
    "font.size"          : 9,
    "axes.titlesize"     : 9,
    "axes.titleweight"   : "bold",
    "axes.titlepad"      : 7,
    "axes.labelsize"     : 9,
    "axes.labelpad"      : 4,
    "xtick.labelsize"    : 8,
    "ytick.labelsize"    : 8,
    "xtick.major.pad"    : 3,
    "ytick.major.pad"    : 3,
    "legend.fontsize"    : 7.5,
    "legend.framealpha"  : 0.9,
    "legend.edgecolor"   : "#cccccc",
    "legend.handlelength": 1.5,
    "figure.dpi"         : 300,
    "savefig.dpi"        : 300,
    "axes.spines.top"    : False,
    "axes.spines.right"  : False,
    "axes.grid"          : True,
    "grid.alpha"         : 0.2,
    "grid.linewidth"     : 0.4,
    "lines.linewidth"    : 1.6,
    "patch.linewidth"    : 0.8,
})

# Color palette (color-blind friendly)
C = {
    "lcb"    : "#2166ac",   # blue
    "greedy" : "#d73027",   # red
    "random" : "#4dac26",   # green
    "bold"   : "#762a83",   # purple
    "base"   : "#f4a582",   # salmon (baseline highlight)
    "ablated": "#4393c3",   # light blue
    "gray"   : "#888888",
}

def load(path):
    with open(path) as f:
        return json.load(f)

def save(fig, name):
    path = OUT_DIR / name
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved: {name}")


# ══════════════════════════════════════════════════════════
# Fig 2  AL Learning Curves
# (Aggressive vs Bold_206, 10 seeds each, 3 panels)
# ══════════════════════════════════════════════════════════
def fig2_al_curves():
    agg_s0  = load(RES_DIR / "active_learning_v6_al_aggressive_results.json")
    agg_s10 = load(RES_DIR / "active_learning_v6_al_aggressive_s10_results.json")
    bld_s0  = load(RES_DIR / "active_learning_v6_al_bold_206_results.json")
    bld_s10 = load(RES_DIR / "active_learning_v6_al_bold_206_s10_results.json")

    def merge_hist(d1, d2, key):
        h1 = d1.get(key, [])
        h2 = d2.get(key, [])
        return np.array(h1 + h2) if h1 and h2 else None

    n_init  = 20
    n_query_agg  = agg_s0["n_query"]   # 2
    n_query_bold = bld_s0["n_query"]   # 3
    threshold    = 370.0

    datasets = {
        "LCBDS (220 samples, $n_q$=2)": {
            "best" : merge_hist(agg_s0, agg_s10, "lcb_best_history"),
            "oob"  : merge_hist(agg_s0, agg_s10, "lcb_oob_history"),
            "sigma": merge_hist(agg_s0, agg_s10, "lcb_sigma_history"),
            "nq"   : n_query_agg,
            "color": C["lcb"], "ls": "-",
        },
        "LCBDS+MaxMu (470 samples, $n_q$=3)": {
            "best" : merge_hist(bld_s0, bld_s10, "lcb_best_history"),
            "oob"  : merge_hist(bld_s0, bld_s10, "lcb_oob_history"),
            "sigma": merge_hist(bld_s0, bld_s10, "lcb_sigma_history"),
            "nq"   : n_query_bold,
            "color": C["bold"], "ls": "-",
        },
        "Greedy (220 samples)": {
            "best" : merge_hist(agg_s0, agg_s10, "greedy_best_history"),
            "oob"  : merge_hist(agg_s0, agg_s10, "greedy_oob_history"),
            "sigma": merge_hist(agg_s0, agg_s10, "greedy_sigma_history"),
            "nq"   : n_query_agg,
            "color": C["greedy"], "ls": "--",
        },
        "Random (220 samples)": {
            "best" : merge_hist(agg_s0, agg_s10, "rnd_best_history"),
            "oob"  : merge_hist(agg_s0, agg_s10, "rnd_oob_history"),
            "sigma": merge_hist(agg_s0, agg_s10, "rnd_sigma_history"),
            "nq"   : n_query_agg,
            "color": C["random"], "ls": ":",
        },
    }

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.9))
    fig.subplots_adjust(wspace=0.42, left=0.09, right=0.98, top=0.88, bottom=0.30)

    panel_labels = ["(a)", "(b)", "(c)"]
    panel_titles = [
        "Best $J_{10}$ Found",
        "Surrogate Quality (OOB $R^2$)",
        "Residual Uncertainty $\\bar{\\sigma}$",
    ]
    ylabels = [
        "Best $J_{10}$ (mV)",
        "Bootstrap OOB $R^2$",
        "Mean uncertainty $\\bar{\\sigma}$ (mV)",
    ]

    for idx, (label, ds) in enumerate(datasets.items()):
        arr_best  = ds["best"]
        arr_oob   = ds["oob"]
        arr_sigma = ds["sigma"]
        nq        = ds["nq"]
        color     = ds["color"]
        ls        = ds["ls"]

        if arr_best is None:
            continue

        n_steps_best  = arr_best.shape[1]
        x_best = np.arange(n_steps_best) * nq + n_init

        for arr, ax, x_arr in zip(
            [arr_best, arr_oob, arr_sigma],
            axes,
            [x_best,
             np.arange(arr_oob.shape[1]) * nq + n_init if arr_oob is not None else x_best,
             np.arange(arr_sigma.shape[1]) * nq + n_init if arr_sigma is not None else x_best],
        ):
            if arr is None:
                continue
            mu  = arr.mean(0)
            std = arr.std(0)
            # align x to mu length
            x_use = x_arr[:len(mu)]
            ax.plot(x_use, mu, color=color, ls=ls, lw=1.6, label=label)
            ax.fill_between(x_use, mu - std, mu + std,
                            alpha=0.12, color=color, linewidth=0)

    # Panel (a): best J10
    axes[0].axhline(threshold, color=C["gray"], ls="--", lw=1.0,
                    label=f"Target ({threshold:.0f} mV)")
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Labeled samples")
    axes[0].set_ylabel(ylabels[0])

    # Panel (b): OOB R2
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].set_xlabel("Labeled samples")
    axes[1].set_ylabel(ylabels[1])

    # Panel (c): sigma
    axes[2].set_xlabel("Labeled samples")
    axes[2].set_ylabel(ylabels[2])

    for i, ax in enumerate(axes):
        ax.set_title(f"{panel_labels[i]} {panel_titles[i]}", loc="left", pad=5)

    # Shared legend below (anchor within bottom margin, bottom=0.30 gives space)
    handles, labels_leg = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_leg,
               loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, 0.01),
               fontsize=7, framealpha=0.9,
               columnspacing=1.0, handlelength=1.5)

    save(fig, "fig2_al_curves.png")


# ══════════════════════════════════════════════════════════
# Fig 3  Component Ablation (3 panels)
# ══════════════════════════════════════════════════════════
def fig3_ablation():
    def load_abl(name):
        return load(RES_DIR / f"active_learning_v6_{name}_results.json")

    configs = [
        ("al_aggressive",   "Baseline"),
        ("al_no_mc",        "w/o MC"),
        ("al_no_annealing", "w/o $\\beta$"),
        ("al_no_diversity", "w/o $\\gamma$"),
        ("al_no_surprise",  "w/o $\\delta$"),
        ("al_single_query", "w/o Max$\\Sigma$"),
        ("al_pure_lcbds",   "Pure LCBDS"),
        ("al_no_extras",    "No Extras"),
    ]
    names  = [c[0] for c in configs]
    labels = [c[1] for c in configs]
    data   = {n: load_abl(n) for n in names}

    def get(d, k, sub=None):
        v = d.get(k, {})
        return v.get(sub, float("nan")) if sub else v.get("mean", float("nan"))
    def getstd(d, k):
        return d.get(k, {}).get("std", 0.0)

    lcb_break = np.array([get(data[n], "lcb_break")           for n in names])
    lcb_std   = np.array([getstd(data[n], "lcb_break")         for n in names])
    lcb_best  = np.array([get(data[n], "final_best_mV", "lcb") for n in names])
    lcb_oob   = np.array([get(data[n], "final_oob_r2",  "lcb") for n in names])
    rnd_break = get(data["al_aggressive"], "rnd_break")
    rnd_best  = get(data["al_aggressive"], "final_best_mV", "random")
    rnd_oob   = get(data["al_aggressive"], "final_oob_r2",  "random")

    x      = np.arange(len(configs))
    colors = [C["base"] if n == "al_aggressive" else C["ablated"] for n in names]
    ekw    = dict(elinewidth=0.8, capsize=2.5, ecolor="#555")

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.6))
    fig.subplots_adjust(wspace=0.48, left=0.10, right=0.98,
                        top=0.88, bottom=0.32)

    # (a) break_exp
    ax = axes[0]
    ax.bar(x, lcb_break, yerr=lcb_std, color=colors, alpha=0.85,
           width=0.62, error_kw=ekw, zorder=3)
    ax.axhline(rnd_break, color=C["random"], ls="--", lw=1.2,
               label=f"Random ({rnd_break:.0f})", zorder=4)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8, ha="right", rotation=40)
    ax.set_ylabel("Experiments to break\n370 mV threshold")
    ax.set_title("(a) Discovery Efficiency", loc="left")
    ax.set_ylim(0, 245)
    ax.legend(loc="upper left", fontsize=7)

    # (b) best_mV
    ax = axes[1]
    y_lo = min(lcb_best.min(), rnd_best) - 15
    y_hi = max(lcb_best.max(), rnd_best) + 18
    ax.bar(x, lcb_best, color=colors, alpha=0.85, width=0.62, zorder=3)
    ax.axhline(rnd_best, color=C["random"], ls="--", lw=1.2,
               label=f"Random ({rnd_best:.0f} mV)", zorder=4)
    for xi, val in zip(x, lcb_best):
        ax.text(xi, val + 1.2, f"{val:.0f}",
                ha="center", va="bottom", fontsize=6.5, color="#222")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8, ha="right", rotation=40)
    ax.set_ylim(y_lo, y_hi)
    ax.set_ylabel("Best $J_{10}$ found (mV)\n[lower is better]")
    ax.set_title("(b) Optimization Depth", loc="left")
    ax.legend(loc="upper left", fontsize=7)

    # (c) OOB R2
    ax = axes[2]
    ax.bar(x, lcb_oob, color=colors, alpha=0.85, width=0.62, zorder=3)
    ax.axhline(rnd_oob, color=C["random"], ls="--", lw=1.2,
               label=f"Random ({rnd_oob:.3f})", zorder=4)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8, ha="right", rotation=40)
    ax.set_ylabel("Bootstrap OOB $R^2$")
    ax.set_ylim(0, 1.08)
    ax.set_title("(c) Model Quality", loc="left")
    ax.legend(loc="lower right", fontsize=7)

    patch_b = mpatches.Patch(color=C["base"],    label="Baseline (all components)")
    patch_a = mpatches.Patch(color=C["ablated"], label="Ablated variant")
    fig.legend(handles=[patch_b, patch_a],
               loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, 0.0),
               fontsize=7.5, framealpha=0.9)

    save(fig, "fig3_ablation_components.png")


# ══════════════════════════════════════════════════════════
# Fig 4  Aggressive vs Bold_206 Comparison
# (Point C contribution — the paper's key argument)
# ══════════════════════════════════════════════════════════
def fig4_strategy_comparison():
    agg_s0  = load(RES_DIR / "active_learning_v6_al_aggressive_results.json")
    agg_s10 = load(RES_DIR / "active_learning_v6_al_aggressive_s10_results.json")
    bld_s0  = load(RES_DIR / "active_learning_v6_al_bold_206_results.json")
    bld_s10 = load(RES_DIR / "active_learning_v6_al_bold_206_s10_results.json")

    def per_seed(d1, d2, key):
        return d1.get(key, []) + d2.get(key, [])

    agg_lcb  = np.array(per_seed(agg_s0, agg_s10, "lcb_best_per_seed"))
    agg_grdy = np.array(per_seed(agg_s0, agg_s10, "greedy_best_per_seed"))
    agg_rnd  = np.array(per_seed(agg_s0, agg_s10, "rnd_best_per_seed"))
    bld_lcb  = np.array(per_seed(bld_s0, bld_s10, "lcb_best_per_seed"))
    bld_grdy = np.array(per_seed(bld_s0, bld_s10, "greedy_best_per_seed"))
    bld_rnd  = np.array(per_seed(bld_s0, bld_s10, "rnd_best_per_seed"))

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
    fig.subplots_adjust(wspace=0.38, left=0.10, right=0.98,
                        top=0.88, bottom=0.18)

    # (a) Strip plot — per-seed best_mV distribution
    ax = axes[0]
    datasets = [
        ("Agg. LCBDS\n($n_q$=2)", agg_lcb,  C["lcb"],    0),
        ("Bold LCBDS\n($n_q$=3)", bld_lcb,  C["bold"],   1.4),
        ("Agg.\nGreedy",          agg_grdy, C["greedy"], 2.8),
        ("Agg.\nRandom",          agg_rnd,  C["random"], 4.0),
    ]
    positions = [d[3] for d in datasets]
    for label, vals, color, pos in datasets:
        jitter = np.random.default_rng(42).uniform(-0.12, 0.12, len(vals))
        ax.scatter(np.full(len(vals), pos) + jitter, vals,
                   color=color, alpha=0.75, s=28, zorder=4, edgecolors="white", lw=0.4)
        ax.plot([pos - 0.22, pos + 0.22], [np.mean(vals)] * 2,
                color=color, lw=2.0, zorder=5)

    ax.axhline(206, color="#999", ls=":", lw=0.9, alpha=0.8)
    ax.text(4.5, 206, "206 mV\n(global min)", va="center",
            fontsize=6.5, color="#666")
    ax.axhline(370, color=C["random"], ls="--", lw=0.9, alpha=0.6)
    ax.text(4.5, 370, "370 mV\n(target)", va="center",
            fontsize=6.5, color="#4dac26")

    ax.set_xticks(positions)
    ax.set_xticklabels([d[0] for d in datasets], fontsize=7.5, ha="center")
    ax.set_ylabel("Best $J_{10}$ found (mV) [lower = better]")
    ax.set_title("(a) Per-seed Best $J_{10}$ Distribution\n(10 seeds, horizontal bar = mean)",
                 loc="left", fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlim(-0.5, 5.0)
    ax.set_ylim(430, 190)

    # (b) Bar chart — discovery rate of global optimum
    ax = axes[1]
    thresholds = [210, 260, 340, 370]
    labels_thr = ["≤210 mV\n(≈global min)", "≤260 mV\n(top-2)", "≤340 mV\n(top-5)", "≤370 mV\n(target)"]
    x = np.arange(len(thresholds))
    w = 0.22

    groups = [
        ("Agg. LCBDS",  agg_lcb,  C["lcb"],    -1.5*w),
        ("Bold LCBDS",  bld_lcb,  C["bold"],   -0.5*w),
        ("Agg. Greedy", agg_grdy, C["greedy"],  0.5*w),
        ("Agg. Random", agg_rnd,  C["random"],  1.5*w),
    ]

    for label, vals, color, offset in groups:
        rates = [(vals <= thr).mean() * 100 for thr in thresholds]
        ax.bar(x + offset, rates, width=w, color=color, alpha=0.85,
               label=label, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(labels_thr, fontsize=7.5)
    ax.set_ylabel("Discovery rate (% of 10 seeds)")
    ax.set_ylim(0, 115)
    ax.set_title("(b) Discovery Rate by Performance Threshold",
                 loc="left", fontsize=8.5)
    ax.legend(loc="upper left", fontsize=7, ncol=2, columnspacing=0.8)

    save(fig, "fig4_strategy_comparison.png")


# ══════════════════════════════════════════════════════════
# Fig 5  Hyperparameter Sensitivity
# ══════════════════════════════════════════════════════════
def fig5_sweeps():
    def load_abl(name):
        return load(RES_DIR / f"active_learning_v6_{name}_results.json")

    def gv(d, k, sub=None):
        v = d.get(k, {})
        return v.get(sub, float("nan")) if sub else v.get("mean", float("nan"))
    def gs(d, k):
        return d.get(k, {}).get("std", 0.0)

    agg = load_abl("al_aggressive")
    rnd_break = gv(agg, "rnd_break")
    rnd_best  = gv(agg, "final_best_mV", "random")

    sweeps = [
        {
            "title" : "(a) $\\beta_{min}$ Sweep",
            "xlabel": "$\\beta_{min}$",
            "cfgs"  : ["al_beta_0p05","al_beta_0p1","al_aggressive",
                       "al_beta_0p3","al_beta_0p5","al_beta_1p0"],
            "xvals" : [0.05, 0.1, 0.2, 0.3, 0.5, 1.0],
        },
        {
            "title" : "(b) $\\gamma$ Sweep (Diversity)",
            "xlabel": "$\\gamma$ (mV)",
            "cfgs"  : ["al_gamma_0","al_gamma_3","al_gamma_5",
                       "al_aggressive","al_gamma_10","al_gamma_15"],
            "xvals" : [0, 3, 5, 8, 10, 15],
        },
        {
            "title" : "(c) $\\delta$ Sweep (Surprise)",
            "xlabel": "$\\delta$ (mV)",
            "cfgs"  : ["al_delta_0","al_delta_5","al_delta_8",
                       "al_aggressive","al_delta_15","al_delta_20"],
            "xvals" : [0, 5, 8, 12, 15, 20],
        },
    ]

    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.8))
    fig.subplots_adjust(hspace=0.52, wspace=0.42,
                        left=0.10, right=0.98, top=0.93, bottom=0.12)

    for col, sw in enumerate(sweeps):
        data_list = [load_abl(n) for n in sw["cfgs"]]
        xvals = np.array(sw["xvals"])
        lcb_break = np.array([gv(d, "lcb_break")           for d in data_list])
        lcb_std   = np.array([gs(d, "lcb_break")            for d in data_list])
        lcb_best  = np.array([gv(d, "final_best_mV", "lcb") for d in data_list])
        base_idx  = sw["cfgs"].index("al_aggressive")

        # top: break_exp
        ax = axes[0, col]
        ax.plot(xvals, lcb_break, "o-", color=C["lcb"], ms=5, lw=1.6,
                label="LCBDS")
        ax.fill_between(xvals, lcb_break - lcb_std, lcb_break + lcb_std,
                        alpha=0.15, color=C["lcb"])
        ax.axhline(rnd_break, color=C["random"], ls="--", lw=1.1,
                   label=f"Random ({rnd_break:.0f})")
        ax.axvline(xvals[base_idx], color=C["gray"], ls=":", lw=1.1,
                   alpha=0.9, label="Baseline config")
        ax.set_ylabel("break$_{exp}$ (↓ better)", fontsize=8)
        ax.set_title(sw["title"], loc="left")
        ax.legend(fontsize=6.5, loc="upper right", handlelength=1.2)
        ax.set_xlabel(sw["xlabel"], fontsize=8)

        # bottom: best_mV
        ax = axes[1, col]
        ax.plot(xvals, lcb_best, "s-", color=C["bold"], ms=5, lw=1.6,
                label="LCBDS")
        ax.axhline(rnd_best, color=C["random"], ls="--", lw=1.1,
                   label=f"Random ({rnd_best:.0f} mV)")
        ax.axvline(xvals[base_idx], color=C["gray"], ls=":", lw=1.1,
                   alpha=0.9, label="Baseline config")
        ax.invert_yaxis()
        ax.set_xlabel(sw["xlabel"], fontsize=8)
        ax.set_ylabel("Best $J_{10}$ (mV, ↓ better)", fontsize=8)
        ax.legend(fontsize=6.5, loc="lower right", handlelength=1.2)

    save(fig, "fig5_hyperparameter_sweep.png")


if __name__ == "__main__":
    print("[plot_journal_figures] Generating figures -> figures/")
    np.random.seed(42)
    fig2_al_curves()
    fig3_ablation()
    fig4_strategy_comparison()
    fig5_sweeps()
    print("[plot_journal_figures] Done.")
