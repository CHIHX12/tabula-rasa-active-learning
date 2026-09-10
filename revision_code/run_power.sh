#!/bin/bash
# The head-to-head that decides the paper's recommendation is underpowered at
# n = 10 (nothing reaches significance).  Add 20 further independent seeds to
# the four arms that matter, so the comparison is settled rather than suggested.
cd "$(dirname "$0")/.." || exit 1
MAXJ=4
SEEDS="20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39"
run () {
  local tag=$1; shift; local gpu=$1; shift
  [ -f "revision/results/${tag}.json" ] && { echo "SKIP $tag"; return; }
  while [ "$(jobs -rp | wc -l)" -ge "$MAXJ" ]; do sleep 5; done
  ( echo "START $tag $(date +%H:%M:%S)"
    CUDA_VISIBLE_DEVICES=$gpu OMP_NUM_THREADS=2 python3 revision/al_harness.py \
      --tag "$tag" --seeds "$SEEDS" "$@" > "revision/logs/${tag}.log" 2>&1 \
      && echo "DONE $tag $(date +%H:%M:%S)" || echo "FAIL $tag $(date +%H:%M:%S)"
  ) &
  sleep 2
}
echo "=== POWER RUN START $(date +%F\ %H:%M:%S) ==="
run P_lcbds   0 --protocol "lcbds+maxsigma" --n-iter 100 --gamma 8 --delta 12
run P_ei      1 --protocol "ei+ei"          --n-iter 100
run P_ei_ms   0 --protocol "ei+maxsigma"    --n-iter 100
run P_greedy  1 --protocol "greedy+greedy"  --n-iter 100
wait
echo "=== POWER RUN DONE $(date +%F\ %H:%M:%S) ==="
