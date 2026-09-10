#!/bin/bash
# Find-then-sweep: the full test, launched at n = 30 from the start.
#
# Why this design (all three measured, see revision/diag_*.py):
#   1. No surrogate can represent the cliff.  With the twelve nearest
#      neighbours of the 206 mV optimum labelled (379-437 mV) PC-BAN predicts
#      416.9 there, RF 423.1, GP 413.3, MLP 411.0, and all four rank the
#      optimum near the middle of the pool.  A mu-based rule therefore walks
#      away from the optimum, whatever the acquisition function.
#   2. The surrogate does locate the good REGION: recall of the 17 sub-370
#      compositions in a 220-long shortlist is 0.59 (PC-BAN) against 0.04 for a
#      random shortlist -- a 15-fold enrichment.
#   3. The good points are mutually close: from the 252 mV point the optimum is
#      0.083 away (64 candidates), from a 362 mV point 0.102 (102 candidates),
#      and campaigns reach a sub-370 value at about query 102, leaving ~118.
#
# So: let the model choose the region, and let exhaustive local coverage find
# the discontinuity inside it.
cd "$(dirname "$0")/.." || exit 1

MAXJ=8
S30="0,1,2,3,4,10,11,12,13,14,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39"
FIX="--sigma-calib --warm-epochs -1 --n-components 1"

run () {
  local tag=$1; shift
  local gpu=$1; shift
  local proto=$1; shift
  [ -f "revision/results/${tag}.json" ] && { echo "SKIP $tag"; return; }
  while [ "$(jobs -rp | wc -l)" -ge "$MAXJ" ]; do sleep 10; done
  ( echo "START $tag $(date +%H:%M:%S)"
    CUDA_VISIBLE_DEVICES=$gpu OMP_NUM_THREADS=2 python3 revision/al_harness.py \
      --tag "$tag" --seeds "$S30" --n-iter 100 --protocol "$proto" "$@" \
      > "revision/logs/${tag}.log" 2>&1 \
      && echo "DONE $tag $(date +%H:%M:%S)" || echo "FAIL $tag $(date +%H:%M:%S)" ) &
  sleep 3
}

echo "=== SWEEP-FULL START $(date +%F\ %H:%M:%S) ==="

# --- the proposal, on three acquisition families ---------------------------
run S_ei_sweep     0 "ei+sweep"     $FIX
run S_lcbds_sweep  1 "lcbds+sweep"  $FIX --acq-scale standardized \
                      --surp-mode local --gamma 0.5 --delta 0.5
run S_greedy_sweep 0 "greedy+sweep" $FIX

# --- when should the sweep start?  ----------------------------------------
run S_gate380      1 "ei+sweep"     $FIX --sweep-gate 380
run S_gate400      0 "ei+sweep"     $FIX --sweep-gate 400

# --- matched controls at the same n and the same surrogate fixes ----------
run S_ei_ctrl      1 "ei+ei"        $FIX
run S_ei_ms_ctrl   0 "ei+maxsigma"  $FIX
run S_greedy_ctrl  1 "greedy+greedy" $FIX

wait
echo "=== SWEEP-FULL DONE $(date +%F\ %H:%M:%S) ==="
