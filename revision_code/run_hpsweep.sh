#!/bin/bash
# Hyperparameter sensitivity, re-run through the controlled harness.
#
# The submitted Section 3.4 quoted a five-seed sweep produced by the original
# v6 pipeline (beta_min default 0.1, NumPy-only seeding).  Recomputing those
# same files at ten seeds already moves the numbers well outside the quoted
# range, and mixing two pipelines inside one study is not defensible.
# Every point of the sweep is therefore regenerated
# here through revision/al_harness.py: same pool, same K-means initial design,
# same ten seeds, same 220-experiment budget, same out-of-pool metric.
#
# Three points of the grid already exist and are not re-run:
#   beta_min = 0.2, gamma = 8, delta = 12  -> E1_full   (the reference point)
#   gamma = 0                              -> E1_nodiv
#   delta = 0                              -> E1_nosurp
cd "$(dirname "$0")/.." || exit 1
mkdir -p revision/logs revision/results revision/.locks

MAXJ=${MAXJ:-4}
SEEDS="0,1,2,3,4,10,11,12,13,14"

run () {   # run <tag> <gpu> <args...>
  local tag=$1; shift
  local gpu=$1; shift
  [ -f "revision/results/${tag}.json" ] && { echo "SKIP $tag"; return; }
  while [ "$(jobs -rp | wc -l)" -ge "$MAXJ" ]; do sleep 15; done
  ( mkdir "revision/.locks/$tag" 2>/dev/null || exit 0
    echo "START $tag $(date +%H:%M:%S)"
    CUDA_VISIBLE_DEVICES=$gpu OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
      python3 revision/al_harness.py --tag "$tag" --seeds "$SEEDS" \
        --protocol "lcbds+maxsigma" --n-iter 100 "$@" \
      > "revision/logs/${tag}.log" 2>&1 \
      && echo "DONE $tag $(date +%H:%M:%S)" || echo "FAIL $tag $(date +%H:%M:%S)"
  ) &
  sleep 3
}

echo "=== HPSWEEP START $(date +%F\ %H:%M:%S) ==="

# annealing floor of the exploration weight
run H_beta005 0 --beta-min 0.05 --gamma 8 --delta 12
run H_beta010 1 --beta-min 0.10 --gamma 8 --delta 12
run H_beta030 0 --beta-min 0.30 --gamma 8 --delta 12
run H_beta050 1 --beta-min 0.50 --gamma 8 --delta 12
run H_beta100 0 --beta-min 1.00 --gamma 8 --delta 12

# spatial diversity weight
run H_gam03   1 --gamma 3  --delta 12
run H_gam05   0 --gamma 5  --delta 12
run H_gam10   1 --gamma 10 --delta 12
run H_gam15   0 --gamma 15 --delta 12

# surprise weight
run H_del05   1 --gamma 8 --delta 5
run H_del08   0 --gamma 8 --delta 8
run H_del15   1 --gamma 8 --delta 15
run H_del20   0 --gamma 8 --delta 20

wait
echo "=== HPSWEEP DONE $(date +%F\ %H:%M:%S) ==="
