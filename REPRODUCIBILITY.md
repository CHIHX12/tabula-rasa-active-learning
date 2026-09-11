# Reproducibility

Every reported figure and table can be regenerated from this archive without
the raw data, because the trajectories every result was computed from are
included. The raw data is needed only to re-run the campaigns from scratch,
and it is not redistributed here: see [DATASETS.md](DATASETS.md) for the
source, DOI, expected path and required columns of each of the six datasets,
and run `python3 fetch_data.py` to download the four UCI ones automatically.

## Determinism

The code deposited with the original submission seeded only NumPy, so PyTorch
weight initialisation and MC-Dropout were random on every run and the LCBDS and
greedy trajectories were not bit-reproducible (only the random baseline was).
The claim of bit-reproducibility made in the original submission therefore did
not hold of the code as deposited, and we correct it here.

It holds of this archive. Every driver fixes the NumPy, PyTorch and CUDA
(cuDNN) seeds and sets `torch.backends.cudnn.deterministic = True` with
`benchmark = False`, so re-running a configuration reproduces the stored
best-value, out-of-pool R² and out-of-bag R² trajectories exactly.

To check it yourself:

```bash
python3 verify_reproducibility.py
```

It runs one short configuration twice and compares the two trajectories
element by element. It needs the OER dataset at `data/OER_database.csv` and a
CUDA device, and takes about a minute.

## Layout

```
.
├── main_active_learning.py     # the original seed-fixed AL pipeline
├── src/                        # PC-BAN model and the OER data loader
├── configs/                    # the original configs, plus _s10 seed variants
├── revision_code/              # the controlled harness written for the revision
├── revision_results/           # trajectories for every result reported in the revision
├── precomputed_results/        # trajectories for the originally reported numbers
├── generalize_multi.py         # the five cross-domain datasets, one code path
├── generalize_concrete.py      # the original single-dataset generalization script
├── gp_baseline.py              # the Gaussian-process baseline
├── compare_all.py              # fresh runs against the precomputed numbers
├── plot_journal_figures.py     # figures
├── fetch_data.py               # downloads the four UCI datasets
├── DATASETS.md                 # provenance of all six datasets
└── verify_reproducibility.py   # the determinism check described above
```

`run_all_experiments.sh` drives the original battery. The revision battery,
the ablation, the hyperparameter sweep and the cross-domain runs are driven by
the shell scripts in `revision_code/`.

## The "ten seeds" convention

Ten seeds means 0-4 plus 10-14. In the original layout that was a base config
plus its `_s10` variant, merged; the revision harness takes the whole list
directly via `--seeds 0,1,2,3,4,10,11,12,13,14`.

## No synthetic data

Nothing here generates synthetic targets. "Running an experiment" means
looking up the real measured value for a chosen composition, which is what
makes a controlled comparison between strategies over matched seeds possible.
The two continuous-simplex oracles of Section 3.6 are the one exception and are
labelled as such: one is a thin-plate-spline interpolant fitted to the measured
OER data, the other an analytic test function, and both exist precisely to
check that the pool-based conclusions do not depend on the pool.
