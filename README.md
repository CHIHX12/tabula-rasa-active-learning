# Descriptor-free active learning across compositional design spaces

Code and precomputed results for the PC-BAN surrogate and the LCBDS
acquisition function, together with the controlled-experiment harness built
for the revision.

> **No dataset is redistributed here.** All six datasets are third-party
> resources; see [DATASETS.md](DATASETS.md) for provenance, DOIs, expected
> paths and required columns, and run `python fetch_data.py` to retrieve the
> four UCI sets automatically. Every reported figure and table can be
> regenerated from the precomputed trajectories without the raw data.

## Layout

```
main_active_learning.py     original AL pipeline (seed-fixed, see below)
gp_baseline.py              Gaussian-process Bayesian-optimisation baseline
generalize_multi.py         cross-domain study over the five public datasets
plot_journal_figures.py     Figs. 2-5
make_curve_figures.py       Fig. 6
src/                        PC-BAN surrogate and data loader
configs/                    49 experiment configurations (seeds 0-4 and 10-14)
precomputed_results/        trajectories for the published runs
revision_code/              controlled-experiment harness for the revision
revision_results/           trajectories for every revision experiment
```

## Reproducibility

`main_active_learning.py` fixes the NumPy, PyTorch and CUDA (cuDNN) seeds, so
re-running a configuration reproduces the stored trajectory exactly.

> Releases before v2.0.0 seeded only NumPy; PyTorch weight initialisation and
> MC-Dropout were left random, so runs of those versions were not bit-identical
> between repetitions. The three lines that fix it are now present, and
> `revision_code/audit_consistency.py` re-derives every number quoted in the
> manuscript from the stored results.

## The revision harness

`revision_code/al_harness.py` holds the candidate pool, the K-means initial
design, the seeds, the labelled budget and the evaluation metric constant, and
varies one factor at a time: the surrogate (`pcban`, `rf`, `gp`, `mlp`,
`mlp_descriptor`), the batch composition (e.g. `lcbds+maxsigma`, `ei+ei`,
`ei+maxsigma`), the acquisition weights, and the retraining protocol.

```bash
python revision_code/al_harness.py --tag demo --protocol "lcbds+maxsigma" \
       --n-iter 100 --seeds 0,1,2,3,4
python revision_code/analyze.py e1     # budget-matched component ablation
python revision_code/analyze.py e2     # exploration protocol held constant
python revision_code/audit_consistency.py
```

Diagnostics that explain the ablation results are in
`revision_code/diag_terms.py` (magnitude of each acquisition term),
`diag_cliff.py` (whether any surrogate can represent the optimum) and
`diag_recall.py` (region-level signal).

## Environment

Python 3.10, PyTorch 2.11 (CUDA 12.8), scikit-learn 1.5, SciPy 1.15.
`pip install -r requirements.txt`.

## Citation

Archived on Zenodo: [10.5281/zenodo.20272736](https://doi.org/10.5281/zenodo.20272736)
(concept DOI, always resolves to the current version).
