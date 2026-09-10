# Reproducibility Package (seed-fixed)

Permanent location: `/home/cycheng/投稿/reproducibility/`
Created: 2026-06-30. This is a self-contained copy — it does **not** depend on `/tmp`.

## What changed vs the original `submission_code`

The original code only seeded NumPy, so PyTorch weight init + MC-Dropout were
random each run → LCBDS/Greedy results were **not** bit-reproducible (only the
Random baseline was). Three lines were added to make every run deterministic:

`main_active_learning.py`:
- top of file: `torch.backends.cudnn.deterministic = True` / `benchmark = False`
- inside each training function (`bootstrap_train_oob`, warm-start, distill),
  right after the NumPy RNG line:
  ```python
  torch.manual_seed(seed_offset)
  torch.cuda.manual_seed_all(seed_offset)
  ```

**Verified:** running the same config twice now gives bit-identical LCBDS /
Greedy / OOB trajectories (previously they varied ±5–70 mV).

## Layout

```
reproducibility/
├── main_active_learning.py     # seed-fixed AL pipeline
├── src/                        # PC-BAN model + data loader
├── configs/                    # 26 original configs + 22 *_s10 (seeds 10-14)
├── data/OER_database.csv       # original OER data (6074 x 4)  [real]
├── data_external/
│   ├── concrete.csv            # UCI Concrete (1030 x 8)  [real, generalization test]
│   ├── Concrete_Data_UCI_official.xls   # UCI official source
│   └── Concrete_Readme_UCI.txt          # provenance: Yeh (1998)
├── results/                    # fresh-run result JSONs
├── precomputed_results/        # original paper numbers (for comparison)
├── generalize_concrete.py      # generalization test (same method, new dataset)
├── compare_all.py              # fresh vs precomputed comparison
└── run_parallel.sh             # 6-way parallel orchestrator (idempotent)
```

## How to reproduce

```bash
cd /home/cycheng/投稿/reproducibility

# One config (now bit-reproducible):
python3 main_active_learning.py --config configs/al_aggressive.yaml

# All remaining configs, 6-way parallel, skip-if-done:
./run_parallel.sh

# Generalization test on real UCI Concrete data:
python3 generalize_concrete.py        # -> results/generalize_concrete.json

# Compare fresh vs the paper's precomputed numbers:
python3 compare_all.py
```

## 10-seed convention

The paper's "10 seeds" = base config (seeds 0-4) + its `_s10` variant
(seeds 10-14), merged. The 4 main configs already have both; the 22
ablation/sweep configs now have generated `_s10` variants too.

## Datasets are real

- `data/OER_database.csv` — 6074 real NiFeCoCe compositions → J10 (the paper's data).
- `data_external/concrete.csv` — 1030 real UCI Concrete mixes → strength (MPa),
  Prof. I-Cheng Yeh, donated 2007; Yeh, I-C. (1998) *Cement and Concrete Research*
  28(12):1797-1804. Used only for the generalization test.

No synthetic data is generated anywhere; "running an experiment" = looking up the
real measured target for a chosen recipe (oracle setup).
