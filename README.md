# Reproducibility Package
## PC-BAN + LCBDS: Active Learning for OER Catalyst Discovery

This package contains all code, data, and precomputed results needed to
reproduce the figures and tables in the paper.

---

## Directory Structure

```
submission_code/
├── data/
│   └── OER_database.csv          # NiFeCoCe OER dataset (6,074 compositions)
├── src/
│   ├── process_model.py          # PC-BAN surrogate model (MDN + BAN)
│   └── data/
│       └── loader.py             # Dataset loading utilities
├── configs/                      # Experiment configurations (YAML)
│   ├── al_aggressive.yaml        # Main baseline, seeds 0-4
│   ├── al_aggressive_s10.yaml    # Main baseline, seeds 10-14
│   ├── al_bold_206.yaml          # Bold strategy, seeds 0-4
│   ├── al_bold_206_s10.yaml      # Bold strategy, seeds 10-14
│   ├── al_no_mc.yaml             # Ablation: remove MC Dropout
│   ├── al_no_annealing.yaml      # Ablation: remove beta annealing
│   ├── al_no_diversity.yaml      # Ablation: remove diversity term
│   ├── al_no_surprise.yaml       # Ablation: remove surprise term
│   ├── al_single_query.yaml      # Ablation: single query per iter
│   ├── al_pure_lcbds.yaml        # Ablation: pure LCB only
│   ├── al_no_extras.yaml         # Ablation: Points A+B only
│   ├── al_beta_*.yaml            # beta_min sweep (0.05-1.0)
│   ├── al_gamma_*.yaml           # gamma sweep (0-15)
│   └── al_delta_*.yaml           # delta sweep (0-20)
├── precomputed_results/          # JSON results for all experiments
│   └── active_learning_v6_*.json
├── main_active_learning.py       # Main AL training and evaluation script
├── plot_journal_figures.py       # Reproduce Figs. 2-5 from precomputed results
├── run_all_experiments.sh        # Shell script: re-run all experiments from scratch
└── requirements.txt
```

---

## Quick Start (Reproduce Figures Only)

Figures 2-5 can be generated directly from precomputed results **without
re-running any training** (seconds):

```bash
cd submission_code
pip install -r requirements.txt
python plot_journal_figures.py
# Output: figures/fig2_al_curves.png
#         figures/fig3_ablation_components.png
#         figures/fig4_strategy_comparison.png
#         figures/fig5_hyperparameter_sweep.png
```

---

## Full Reproduction (Re-run All Experiments)

To reproduce all results from scratch (~6-10 hours, single GPU recommended):

```bash
cd submission_code
pip install -r requirements.txt
bash run_all_experiments.sh
```

Results are saved to `results/`. After completion, run
`python plot_journal_figures.py` to regenerate figures from fresh results.

### Running Individual Experiments

```bash
# Main comparison baseline (seeds 0-4)
python main_active_learning.py --config configs/al_aggressive.yaml

# Bold strategy targeting global minimum (seeds 0-4)
python main_active_learning.py --config configs/al_bold_206.yaml

# Ablation: remove diversity term
python main_active_learning.py --config configs/al_no_diversity.yaml
```

---

## Dataset

`data/OER_database.csv` contains 6,074 discrete NiFeCoCe quaternary oxide
compositions with measured overpotential J10 (mV) at 10 mA/cm².

| Column | Description |
|--------|-------------|
| Ni     | Ni molar fraction |
| Fe     | Fe molar fraction |
| Co     | Co molar fraction |
| Ce     | Ce molar fraction |
| J10    | Overpotential at 10 mA/cm² (mV) |

The global minimum in this dataset is **206 mV** at
(Ni=0.302, Fe=0.169, Co=0.071, Ce=0.471).

---

## Software Requirements

| Package       | Tested Version |
|---------------|---------------|
| Python        | 3.10          |
| PyTorch       | 2.8.0         |
| NumPy         | 2.2.6         |
| scikit-learn  | 1.5.0         |
| SciPy         | 1.15.3        |
| Pandas        | 2.2.x         |
| Matplotlib    | 3.8.x         |
| PyYAML        | 6.0           |

GPU is recommended but not required. The code automatically falls back to CPU.

---

## Experiment Configurations

All hyperparameters are specified in YAML configs under `configs/`.
Key parameters for the main baseline (`al_aggressive.yaml`):

| Parameter          | Value | Description                          |
|--------------------|-------|--------------------------------------|
| n_init             | 20    | Initial labeled set size             |
| n_iter             | 100   | AL iterations                        |
| n_query            | 2     | Queries per iteration (Points A, B)  |
| beta               | 2.5   | Initial exploration weight           |
| beta_min           | 0.2   | Final exploitation weight            |
| gamma              | 8.0   | Diversity penalty weight (mV)        |
| delta              | 12.0  | Surprise bonus weight (mV)           |
| surp_radius        | 0.10  | Surprise influence radius            |
| surp_thresh_sigma  | 2.0   | Surprise detection threshold (×std)  |
| n_bootstrap        | 5     | Bootstrap ensemble size              |
| n_mc               | 20    | MC Dropout passes per model          |

---

## Expected Key Results (from precomputed_results/)

### Main Comparison (10 seeds, merged)
| Method           | Best J10 (mV) | Notes                        |
|------------------|--------------|------------------------------|
| LCBDS (nq=2)     | 365.8 ± 4.7  | Baseline, all seeds ≤370 mV  |
| LCBDS Bold (nq=3)| 319.9 ± 69.2 | 2/10 seeds find 206 mV       |

### Ablation (5 seeds)
| Variant         | break_exp  | Best J10 (mV) | OOB R²  |
|-----------------|-----------|--------------|---------|
| Full LCBDS      | 113 ± 45  | 363          | 0.872   |
| w/o diversity   | 173 ± 39  | 368          | 0.890   |
| w/o MaxSigma    | 106 ± 18  | 370          | 0.696   |
| Pure LCB        | 87 ± 41   | 376          | 0.674   |

---

## Notes on Stochasticity

Neural network training involves randomness. Results across independent runs
will show small numerical variation (typically ±5 mV in best_J10,
±15 in break_exp). The precomputed results in `precomputed_results/`
represent the exact numbers reported in the paper.

To match paper numbers exactly: use the precomputed results directly
with `python plot_journal_figures.py`.
