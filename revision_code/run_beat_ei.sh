#!/bin/bash
# Can the proposed acquisition beat Expected Improvement?
#
# Three of the five fixes (sigma calibration, K=1, scaled warm-start) are
# surrogate-level and help ANY acquisition, so the honest test is
#
#     LCBDS + fixes    vs    EI + the SAME fixes
#
# Comparing a fixed LCBDS against an unfixed EI would repeat exactly the
# uncontrolled comparison the design requirement objected to.
#
# We also test four LCBDS variants that have a mechanistic reason to beat EI on
# this landscape, whose optimum is an isolated outlier (nearest neighbour 190 mV
# away) that a myopic improvement-based rule has no reason to visit:
#   V1 annealed diversity  - push outward early, exploit late
#   V2 strong late surprise - keep amplifying anomalies after EI has converged
#   V3 batch-diverse       - the second query is repelled from the first
#   V4 wide surprise radius - a surprise recruits a larger neighbourhood
cd "$(dirname "$0")/.." || exit 1

MAXJ=5
S10="0,1,2,3,4,10,11,12,13,14"
FIX_SUR="--sigma-calib --n-components 1 --warm-epochs -1"
FIX_ALL="--acq-scale standardized --surp-mode local $FIX_SUR"

run () {                      # run <tag> <gpu> <protocol> <args...>
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

echo "=== BEAT-EI START $(date +%F\ %H:%M:%S) ==="

# --- the matched control: EI with the same surrogate-level fixes -----------
run X_ei_fix      0 "ei+ei"           $FIX_SUR
run X_ei_ms_fix   1 "ei+maxsigma"     $FIX_SUR

# --- LCBDS variants with a reason to reach an isolated optimum -------------
# V1: diversity annealed the other way round (strong early, weak late)
run X_V1_divhi    0 "lcbds+maxsigma"  $FIX_ALL --gamma 2.0 --delta 0.5
# V2: surprise kept strong; beta floor raised so exploration never dies
run X_V2_surphi   1 "lcbds+maxsigma"  $FIX_ALL --gamma 0.5 --delta 2.0 --beta-min 1.0
# V3: both terms strong
run X_V3_both     0 "lcbds+maxsigma"  $FIX_ALL --gamma 1.5 --delta 2.0 --beta-min 0.8
# V4: wider surprise radius, more sensitive trigger
run X_V4_wide     1 "lcbds+maxsigma"  $FIX_ALL --gamma 0.5 --delta 1.5 \
                      --surp-radius 0.25 --surp-thresh-sigma 1.0
# V5: exploration never annealed away, moderate weights
run X_V5_noanneal 0 "lcbds+maxsigma"  $FIX_ALL --gamma 1.0 --delta 1.0 \
                      --beta 2.0 --beta-min 2.0 --no-anneal

wait
echo "=== BEAT-EI DONE $(date +%F\ %H:%M:%S) ==="
