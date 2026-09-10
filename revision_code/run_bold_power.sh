#!/bin/bash
# At n=10 the deep-optimum hit rate cannot separate the 470-budget arms: the
# exact McNemar test needs five concordant discordant pairs just to reach
# p=0.031.  Add 20 further seeds to the five arms that carry the mechanism
# question, so that n=30 and the test has real power.
cd "$(dirname "$0")/.." || exit 1
MAXJ=5
SEEDS="20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39"
BOLD="--n-iter 150 --beta 2.5 --beta-min 0.05 --surp-radius 0.15 --surp-thresh-sigma 1.5"
run () { local tag=$1; shift; local gpu=$1; shift
  [ -f "revision/results/${tag}.json" ] && { echo "SKIP $tag"; return; }
  while [ "$(jobs -rp | wc -l)" -ge "$MAXJ" ]; do sleep 10; done
  ( echo "START $tag $(date +%H:%M:%S)"
    CUDA_VISIBLE_DEVICES=$gpu OMP_NUM_THREADS=2 python3 revision/al_harness.py \
      --tag "$tag" --seeds "$SEEDS" "$@" > "revision/logs/${tag}.log" 2>&1 \
      && echo "DONE $tag $(date +%H:%M:%S)" || echo "FAIL $tag $(date +%H:%M:%S)" ) &
  sleep 3; }
echo "=== BOLD POWER START $(date +%F\ %H:%M:%S) ==="
run BP_lcbds_bold   0 --protocol "lcbds+maxsigma+maxmu"  --gamma 10 --delta 25 $BOLD
run BP_lcbds_2q     1 --protocol "lcbds+maxsigma" --n-iter 225 --gamma 10 --delta 25 \
                      --beta 2.5 --beta-min 0.05 --surp-radius 0.15 --surp-thresh-sigma 1.5
run BP_third_random 0 --protocol "lcbds+maxsigma+random" --gamma 10 --delta 25 $BOLD
run BP_ei_bold      1 --protocol "ei+maxsigma+maxmu"     --gamma 10 --delta 25 $BOLD
run BP_bold_nosurp  0 --protocol "lcbds+maxsigma+maxmu"  --gamma 10 --delta 0  $BOLD
wait
echo "=== BOLD POWER DONE $(date +%F\ %H:%M:%S) ==="
