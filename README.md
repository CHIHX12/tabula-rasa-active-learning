# What actually matters in closed-loop compositional discovery

Code and precomputed results for a controlled decomposition of a closed-loop
active-learning framework, and for the **uncertainty-contraction ratio**, a
check on an active-learning surrogate that needs no held-out data.

> **No dataset is redistributed here.** All six datasets are third-party
> resources; see [DATASETS.md](DATASETS.md) for provenance, DOIs, expected
> paths and required columns, and run `python fetch_data.py` to retrieve the
> four UCI sets automatically. Every reported figure and table can be
> regenerated from the precomputed trajectories without the raw data.

## The headline result, in one command

```bash
python3 revision_code/contraction_ratio.py
```

The bootstrap out-of-bag score, the usual way to validate a surrogate in
data-scarce active learning, overstates true out-of-pool accuracy by up to 0.26
here, and unevenly: across twelve budget-matched arms the overstatement ranges
from 0.02 to 0.26. It therefore cannot rank sampling strategies. The
contraction ratio can, and uses no held-out data:

    ratio = mean(sigma over the last tenth of iterations)
          / mean(sigma over the first tenth of iterations)

where sigma is the mean predictive spread over the *unlabelled* candidates,
which the acquisition function already computes. Over fourteen budget-matched
configurations on the oxygen-evolution pool, every configuration at 0.65 or
below ends with an out-of-pool R2 between 0.557 and 0.633, and every one at
0.75 or above ends between 0.305 and 0.495, with nothing in between. It was
then tested on seven systems that played no part in choosing it, five further
datasets and two continuous-simplex oracles with no candidate pool at all, and
orders the strategies correctly on all seven.

It ranks configurations, not individual runs, and its threshold has to be
calibrated within one model class. Both limits are documented in the script.

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
