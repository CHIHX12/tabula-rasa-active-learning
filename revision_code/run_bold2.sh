#!/bin/bash
# Follow-on to run_bold.sh: isolate the two mechanisms the paper credits for
# reaching 206 mV at the 470-experiment budget --
#   (a) that the third query is specifically the MAX-MU point ("deliberately
#       synthesising the predicted-worst compositions"), and
#   (b) that the surprise weight must be amplified to delta = 25.
# Both are tested at the SAME 470 budget and the same ten seeds, so the only
# thing that changes is the mechanism under test.
cd "$(dirname "$0")/.." || exit 1
echo "[bold2] waiting for the first Bold battery ..."
until grep -q "BOLD 470 DONE" revision/logs/bold.log 2>/dev/null; do sleep 60; done
echo "[bold2] starting $(date +%H:%M:%S)"
MAXJ=3
SEEDS="0,1,2,3,4,10,11,12,13,14"
BOLD="--n-iter 150 --beta 2.5 --beta-min 0.05 --surp-radius 0.15 --surp-thresh-sigma 1.5"
run () { local tag=$1; shift; local gpu=$1; shift
  [ -f "revision/results/${tag}.json" ] && { echo "SKIP $tag"; return; }
  while [ "$(jobs -rp | wc -l)" -ge "$MAXJ" ]; do sleep 5; done
  ( echo "START $tag $(date +%H:%M:%S)"
    CUDA_VISIBLE_DEVICES=$gpu OMP_NUM_THREADS=2 python3 revision/al_harness.py \
      --tag "$tag" --seeds "$SEEDS" "$@" > "revision/logs/${tag}.log" 2>&1 \
      && echo "DONE $tag $(date +%H:%M:%S)" || echo "FAIL $tag $(date +%H:%M:%S)" ) &
  sleep 2; }

# (a) third query is a RANDOM candidate instead of the predicted-worst point.
#     Same budget, same batch size -> isolates "max-mu" from "a third query".
run B_third_random 0 --protocol "lcbds+maxsigma+random" --gamma 10 --delta 25 $BOLD
# (b) the published Bold protocol but with the DEFAULT surprise weight.
#     Isolates the amplification delta 12 -> 25.
run B_bold_delta12 1 --protocol "lcbds+maxsigma+maxmu"  --gamma 10 --delta 12 $BOLD
# (c) no surprise at all, everything else Bold -> is the mechanism needed here?
run B_bold_nosurp  0 --protocol "lcbds+maxsigma+maxmu"  --gamma 10 --delta 0  $BOLD
wait
echo "=== BOLD2 DONE $(date +%F\ %H:%M:%S) ==="
