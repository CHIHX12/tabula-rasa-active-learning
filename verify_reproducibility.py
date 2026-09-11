#!/usr/bin/env python3
"""Check that a configuration reproduces itself exactly.

The code deposited with the original submission seeded only NumPy, so PyTorch
weight initialisation and MC-Dropout varied between runs.  This archive fixes
the NumPy, PyTorch and CUDA seeds and disables cuDNN autotuning, and the claim
in the paper is that re-running a configuration now reproduces its stored
trajectories exactly.  This script is that claim, executable.

It runs one short campaign twice and compares the two results element by
element: the best-value, out-of-pool R2, out-of-bag R2 and mean-predictive-sigma
trajectories, plus the scalar summaries.  Nothing is allowed to differ, not
even in the last bit.

    python3 verify_reproducibility.py            # ~1 minute
    python3 verify_reproducibility.py --n-iter 30 --seed 3

Requires the OER dataset at data/OER_database.csv (see DATASETS.md) and a CUDA
device.  Exits 0 if the two runs are identical, 1 otherwise.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
HARNESS = "revision/al_harness.py"   # relative to the mirror built below

TRAJECTORIES = ("best", "pool_r2", "oob_r2", "sigma")
SCALARS = ("best_final", "best_overall", "final_pool_r2", "final_oob_r2", "n_labeled")


def run_once(tag: str, seed: int, n_iter: int, workdir: Path) -> dict:
    import os
    cmd = [sys.executable, HARNESS, "--tag", tag, "--seeds", str(seed),
           "--protocol", "lcbds+maxsigma", "--n-iter", str(n_iter),
           "--gamma", "8", "--delta", "12"]
    env = dict(os.environ, PYTHONPATH=str(workdir))
    proc = subprocess.run(cmd, cwd=workdir, capture_output=True, text=True, env=env)
    if proc.returncode != 0:
        sys.exit(f"harness failed:\n{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}")
    out = workdir / "revision" / "results" / f"{tag}.json"
    return json.loads(out.read_text())["runs"][0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-iter", type=int, default=12)
    args = ap.parse_args()

    if not (HERE / "data" / "OER_database.csv").exists():
        sys.exit("data/OER_database.csv not found. See DATASETS.md; the dataset "
                 "is third-party and is not redistributed in this archive.")

    # The harness imports itself as `revision.*` and writes under revision/,
    # which is the layout of the working tree it was written in.  Rather than
    # rewrite it, mirror that layout in a throwaway directory: revision_code
    # becomes revision/, and nothing is written inside the archive.
    tmp = Path(tempfile.mkdtemp(prefix="verify_repro_"))
    try:
        for name in ("data", "src"):
            src = HERE / name
            if src.exists():
                shutil.copytree(src, tmp / name, dirs_exist_ok=True)
        shutil.copytree(HERE / "revision_code", tmp / "revision", dirs_exist_ok=True)

        print(f"running seed {args.seed} for {args.n_iter} iterations, twice ...")
        a = run_once("verify_A", args.seed, args.n_iter, tmp)
        b = run_once("verify_B", args.seed, args.n_iter, tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    ok = True
    for key in TRAJECTORIES:
        same = a["history"][key] == b["history"][key]
        ok &= same
        print(f"  {'identical' if same else 'DIFFERS  '}  history[{key}]  "
              f"({len(a['history'][key])} iterations)")
    for key in SCALARS:
        same = a[key] == b[key]
        ok &= same
        print(f"  {'identical' if same else 'DIFFERS  '}  {key} = {a[key]!r}")

    print("\nbit-reproducible" if ok else "\nNOT reproducible")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
