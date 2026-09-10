#!/bin/bash
# Wave 2 of the improvement campaign.
#   (a) take the all-five configuration to n = 30 so that best_J10 and the
#       deep-optimum hit rate can actually be tested (at n = 10 only the
#       out-of-pool R2 reached significance);
#   (b) sweep gamma and delta on top of the all-five fixes, since the
#       standardisation changed their units and only three points were tried;
#   (c) ask whether the dedicated exploration query is still needed once the
#       acquisition terms are numerically able to act.
cd "$(dirname "$0")/.." || exit 1

MAXJ=5
S20="20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39"
S10="0,1,2,3,4,10,11,12,13,14"
ALLFIX="--acq-scale standardized --sigma-calib --surp-mode local --n-components 1 --warm-epochs -1"

run () {                      # run <tag> <gpu> <seeds> <protocol> <args...>
  local tag=$1; shift
  local gpu=$1; shift
  local seeds=$1; shift
  local proto=$1; shift
  [ -f "revision/results/${tag}.json" ] && { echo "SKIP $tag"; return; }
  while [ "$(jobs -rp | wc -l)" -ge "$MAXJ" ]; do sleep 10; done
  ( echo "START $tag $(date +%H:%M:%S)"
    CUDA_VISIBLE_DEVICES=$gpu OMP_NUM_THREADS=2 python3 revision/al_harness.py \
      --tag "$tag" --seeds "$seeds" --n-iter 100 --protocol "$proto" "$@" \
      > "revision/logs/${tag}.log" 2>&1 \
      && echo "DONE $tag $(date +%H:%M:%S)" || echo "FAIL $tag $(date +%H:%M:%S)" ) &
  sleep 3
}

echo "=== IMPROVE2 START $(date +%F\ %H:%M:%S) ==="

# (a) power: all-five and its matched baseline at 20 further seeds -> n = 30
run I2_ALL_s20   0 "$S20" "lcbds+maxsigma" $ALLFIX --gamma 0.5 --delta 0.5
run I2_base_s20  1 "$S20" "lcbds+maxsigma" --gamma 8 --delta 12

# (b) gamma / delta sweep on top of all five fixes
run I2_g00_d00   0 "$S10" "lcbds+maxsigma" $ALLFIX --gamma 0.0 --delta 0.0
run I2_g02_d02   1 "$S10" "lcbds+maxsigma" $ALLFIX --gamma 0.2 --delta 0.2
run I2_g10_d05   0 "$S10" "lcbds+maxsigma" $ALLFIX --gamma 1.0 --delta 0.5
run I2_g05_d15   1 "$S10" "lcbds+maxsigma" $ALLFIX --gamma 0.5 --delta 1.5
run I2_g15_d15   0 "$S10" "lcbds+maxsigma" $ALLFIX --gamma 1.5 --delta 1.5

# (c) is the dedicated exploration query still needed?
run I2_ALL_2q    1 "$S10" "lcbds+lcbds"    $ALLFIX --gamma 0.5 --delta 0.5

wait
echo "=== IMPROVE2 DONE $(date +%F\ %H:%M:%S) ==="
