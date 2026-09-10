#!/bin/bash
# Two new mechanisms, motivated by the geometry of this landscape rather than
# by parameter tuning.
#
# Diagnostic finding: the global optimum sits at ORDINARY density (mean distance
# to its five nearest neighbours 0.0422 vs a pool median of 0.0427) but at the
# 96.5th percentile of LOCAL ROUGHNESS (sd of J10 among its ten neighbours 18.7
# vs a pool median of 6.3).  It is a cliff inside a well-sampled region, not an
# island in an empty one.  A geometric diversity term rewards empty regions and
# therefore points away from it; Expected Improvement ignores geometry entirely,
# which is why it wins.
#
#   M1  local roughness   eta * rho(x), rho = sd of y among the k nearest
#                         LABELLED points.  Seeks discontinuities directly.
#   M2  mixture EI        exact EI under the K-component mixture instead of its
#                         moment-matched Gaussian.  Expresses "probably
#                         mediocre, small chance of far better", which a single
#                         Gaussian cannot, and finally gives the MDN a purpose.
cd "$(dirname "$0")/.." || exit 1

MAXJ=5
S10="0,1,2,3,4,10,11,12,13,14"
FIX="--sigma-calib --warm-epochs -1"

run () {
  local tag=$1; shift
  local gpu=$1; shift
  local proto=$1; shift
  [ -f "revision/results/${tag}.json" ] && { echo "SKIP $tag"; return; }
  while [ "$(jobs -rp | wc -l)" -ge "$MAXJ" ]; do sleep 10; done
  ( echo "START $tag $(date +%H:%M:%S)"
    CUDA_VISIBLE_DEVICES=$gpu OMP_NUM_THREADS=2 python3 revision/al_harness.py \
      --tag "$tag" --seeds "$S10" --n-iter 100 --protocol "$proto" "$@" \
      > "revision/logs/${tag}.log" 2>&1 \
      && echo "DONE $tag $(date +%H:%M:%S)" || echo "FAIL $tag $(date +%H:%M:%S)" ) &
  sleep 3
}

echo "=== MECHANISM START $(date +%F\ %H:%M:%S) ==="

# M2 alone: mixture EI, K = 10 so the mixture actually exists
run M_mei_k10     0 "mei+maxsigma"      $FIX --n-components 10
run M_mei_k1      1 "mei+maxsigma"      $FIX --n-components 1

# M1 alone: replace the geometric diversity term with local roughness
run M_rough_e05   0 "lcbdr+maxsigma"    $FIX --acq-scale standardized \
                     --surp-mode local --eta 0.5 --delta 0.5 --n-components 1
run M_rough_e15   1 "lcbdr+maxsigma"    $FIX --acq-scale standardized \
                     --surp-mode local --eta 1.5 --delta 0.5 --n-components 1

# M1 + M2 together
run M_both_e05    0 "meirough+maxsigma" $FIX --eta 0.5 --n-components 10
run M_both_e15    1 "meirough+maxsigma" $FIX --eta 1.5 --n-components 10

# roughness as the whole acquisition, to see how much signal it carries alone
run M_roughonly   0 "rough+maxsigma"    $FIX --acq-scale standardized \
                     --eta 1.0 --n-components 1

wait
echo "=== MECHANISM DONE $(date +%F\ %H:%M:%S) ==="
