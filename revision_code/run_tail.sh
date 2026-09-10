#!/bin/bash
# Four levers that the published pipeline never touched, all aimed at the same
# diagnosed problem: the loss is dominated by the 400-450 mV bulk, so a 206 mV
# cliff is fitted as an outlier to be smoothed away.
#
#   tail weighting   PCBAN.full_loss has always accepted a `weights` argument
#                    and nothing in the pipeline ever passed one.  Weighting the
#                    loss towards the best observations sharpens the model
#                    exactly where the search is.
#   target transform a monotone transform that expands resolution at the good
#                    end before fitting.
#   refit frequency  1 query x 200 iterations refits the model twice as often as
#                    2 x 100 at the same budget; 4 x 50 half as often.
#   initial design   20 K-means points is 9% of the budget and was never varied.
#
# All arms use Expected Improvement (the strongest baseline found so far), the
# same 220-experiment budget and the same 30 seeds, so only the named factor
# changes.
cd "$(dirname "$0")/.." || exit 1
MAXJ=8
S30="0,1,2,3,4,10,11,12,13,14,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39"
FIX="--sigma-calib --warm-epochs -1 --n-components 1"
run () { local tag=$1; shift; local gpu=$1; shift; local proto=$1; shift; local iters=$1; shift
  [ -f "revision/results/${tag}.json" ] && { echo "SKIP $tag"; return; }
  while [ "$(jobs -rp | wc -l)" -ge "$MAXJ" ]; do sleep 10; done
  ( echo "START $tag $(date +%H:%M:%S)"
    CUDA_VISIBLE_DEVICES=$gpu OMP_NUM_THREADS=2 python3 revision/al_harness.py \
      --tag "$tag" --seeds "$S30" --n-iter "$iters" --protocol "$proto" "$@" \
      > "revision/logs/${tag}.log" 2>&1 \
      && echo "DONE $tag $(date +%H:%M:%S)" || echo "FAIL $tag $(date +%H:%M:%S)" ) &
  sleep 3; }
echo "=== TAIL START $(date +%F\ %H:%M:%S) ==="
# tail weighting: how sharp?
run T_tau015 0 "ei+ei" 100 $FIX --tail-tau 0.15
run T_tau030 1 "ei+ei" 100 $FIX --tail-tau 0.30
run T_tau060 0 "ei+ei" 100 $FIX --tail-tau 0.60
run T_tau100 1 "ei+ei" 100 $FIX --tail-tau 1.00
# target transform, alone and with weighting
run T_log    0 "ei+ei" 100 $FIX --y-transform log
run T_logtau 1 "ei+ei" 100 $FIX --y-transform log --tail-tau 0.30
# refit frequency at the same 220 budget
run T_1q200  0 "ei"    200 $FIX
run T_4q50   1 "ei+ei+ei+ei" 50 $FIX
# larger initial design
run T_init40 0 "ei+ei" 90  $FIX --n-init 40
wait
echo "=== TAIL DONE $(date +%F\ %H:%M:%S) ==="
