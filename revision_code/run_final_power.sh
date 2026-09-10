#!/bin/bash
# Final power run.
#
# Three configurations reach the global optimum in 4/10 seeds against 1/10 for
# Expected Improvement given the SAME surrogate fixes -- a four-fold effect that
# n = 10 cannot resolve (exact McNemar needs five concordant discordant pairs
# just to reach p = 0.031; we currently have three).  Take the three candidates
# and the control to n = 30 on the same seeds.
#
#   FP_allfix   LCBDS with all five numerical fixes
#   FP_v5       the same, with exploration never annealed away (beta fixed at 2)
#   FP_rough    the geometric diversity term replaced by local roughness
#   FP_ei       Expected Improvement with the identical surrogate fixes (control)
cd "$(dirname "$0")/.." || exit 1

MAXJ=4
S20="20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39"
FIX_SUR="--sigma-calib --warm-epochs -1 --n-components 1"
FIX_ALL="--acq-scale standardized --surp-mode local $FIX_SUR"

run () {
  local tag=$1; shift
  local gpu=$1; shift
  local proto=$1; shift
  [ -f "revision/results/${tag}.json" ] && { echo "SKIP $tag"; return; }
  while [ "$(jobs -rp | wc -l)" -ge "$MAXJ" ]; do sleep 10; done
  ( echo "START $tag $(date +%H:%M:%S)"
    CUDA_VISIBLE_DEVICES=$gpu OMP_NUM_THREADS=2 python3 revision/al_harness.py \
      --tag "$tag" --seeds "$S20" --n-iter 100 --protocol "$proto" "$@" \
      > "revision/logs/${tag}.log" 2>&1 \
      && echo "DONE $tag $(date +%H:%M:%S)" || echo "FAIL $tag $(date +%H:%M:%S)" ) &
  sleep 3
}

echo "=== FINAL POWER START $(date +%F\ %H:%M:%S) ==="
run FP_allfix 0 "lcbds+maxsigma" $FIX_ALL --gamma 0.5 --delta 0.5
run FP_v5     1 "lcbds+maxsigma" $FIX_ALL --gamma 1.0 --delta 1.0 \
                 --beta 2.0 --beta-min 2.0 --no-anneal
run FP_rough  0 "lcbdr+maxsigma" $FIX_ALL --eta 1.5 --delta 0.5
run FP_ei     1 "ei+ei"          $FIX_SUR
wait
echo "=== FINAL POWER DONE $(date +%F\ %H:%M:%S) ==="
