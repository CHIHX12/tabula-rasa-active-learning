"""
Convergence-curve figures (matching the original OER fig2 style):
lines = mean over seeds, shaded = +-1 s.d., x = labeled samples.
600 dpi, large fonts (legible with presbyopia).
Layout: 5 datasets (rows) x 2 metrics (cols):
  col 1 = Best value found   col 2 = Surrogate quality (R^2 on unexplored pool)
"""
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"; FIG.mkdir(exist_ok=True)

rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial"],
    "font.size": 15, "axes.titlesize": 18, "axes.labelsize": 16,
    "xtick.labelsize": 13, "ytick.labelsize": 13, "legend.fontsize": 15,
    "axes.linewidth": 1.5, "xtick.major.width": 1.3, "ytick.major.width": 1.3,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 600, "savefig.dpi": 600, "savefig.bbox": "tight",
    "lines.linewidth": 2.6,
})

# Unified palette: LCBDS blue, Greedy red, Random green, GP-BO orange
STY = {"LCBDS": ("#2166ac", "-"), "GREEDY": ("#d73027", "--"),
       "RANDOM": ("#4dac26", ":"), "GP-BO": ("#e08214", "-.")}
DATASETS = [d for d in ["steel", "concrete", "slump", "wine_red", "energy"]
            if (HERE / f"results/generalize_{d}.json").exists()]
NICE = {"steel": "Steel (alloy)", "concrete": "Concrete", "slump": "Slump",
        "wine_red": "Wine", "energy": "Energy (min load)"}
UNIT = {"steel": "yield strength (MPa)", "concrete": "strength (MPa)",
        "slump": "strength (MPa)", "wine_red": "quality", "energy": "heating load"}

fig, axes = plt.subplots(len(DATASETS), 2, figsize=(13, 4.2*len(DATASETS)))
for row, ds in enumerate(DATASETS):
    d = json.load(open(HERE / f"results/generalize_{ds}.json"))
    ni, nq, nit = d["n_init"], d["n_query"], d["n_iter"]
    x_best = np.array([ni + i*nq for i in range(nit + 1)])
    x_nat = np.array([ni + i*nq for i in range(nit)])
    axb, axn = axes[row, 0], axes[row, 1]
    for s in ["LCBDS", "GREEDY", "RANDOM"]:
        col, ls = STY[s]; r = d["results"][s]
        bm = np.array(r["best_curve_mean"]); bs = np.array(r["best_curve_std"])
        axb.plot(x_best, bm, ls, color=col, label=s)
        axb.fill_between(x_best, bm-bs, bm+bs, color=col, alpha=0.15, lw=0)
        nm = np.array(r["natr2_curve_mean"]); ns = np.array(r["natr2_curve_std"])
        axn.plot(x_nat, nm, ls, color=col, label=s)
        axn.fill_between(x_nat, nm-ns, nm+ns, color=col, alpha=0.15, lw=0)
    gpf = HERE / f"results/gp_{ds}.json"
    if gpf.exists():
        g = json.load(open(gpf)); col, ls = STY["GP-BO"]
        bm = np.array(g["best_curve_mean"]); bs = np.array(g["best_curve_std"])
        xb = x_best[:len(bm)]
        axb.plot(xb, bm, ls, color=col, label="GP-BO")
        axb.fill_between(xb, bm-bs, bm+bs, color=col, alpha=0.12, lw=0)
        nm = np.array(g["natr2_curve_mean"]); ns = np.array(g["natr2_curve_std"])
        xn = x_nat[:len(nm)]
        axn.plot(xn, nm, ls, color=col, label="GP-BO")
        axn.fill_between(xn, nm-ns, nm+ns, color=col, alpha=0.12, lw=0)
    axb.axhline(d["global_best"], ls="--", lw=1.4, color="k", alpha=0.5)
    axb.set_ylabel(f"Best {UNIT[ds]}")
    axb.set_title(f"{NICE[ds]} — best found", loc="left", fontweight="bold")
    axn.axhline(0, lw=1.5, color="k", alpha=0.7)
    axn.set_ylabel("R² on unexplored pool")
    axn.set_title(f"{NICE[ds]} — surrogate quality", loc="left", fontweight="bold")
    if row == len(DATASETS)-1:
        axb.set_xlabel("Labeled samples (experiments)")
        axn.set_xlabel("Labeled samples (experiments)")
    if row == 0:
        axb.legend(frameon=False, loc="best")
fig.tight_layout(pad=1.5)
fig.savefig(FIG / "fig_generalization_curves.png")
fig.savefig(FIG / "fig_generalization_curves.pdf")
print("saved fig_generalization_curves (600 dpi)")
