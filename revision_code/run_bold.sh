#!/bin/bash
# The 470-experiment "Bold" configuration had never been run under the
# controlled harness, so earlier comparisons against it were not controlled.
# Here every arm uses the SAME 470-experiment budget, the same three-query
# batch size, the same K-means initial design and the same ten seeds.
cd "$(dirname "$0")/.." || exit 1
MAXJ=3
SEEDS="0,1,2,3,4,10,11,12,13,14"
BOLD="--n-iter 150 --gamma 10 --delta 25 --beta 2.5 --beta-min 0.05 --surp-radius 0.15 --surp-thresh-sigma 1.5"
run () { local tag=$1; shift; local gpu=$1; shift
  [ -f "revision/results/${tag}.json" ] && { echo "SKIP $tag"; return; }
  while [ "$(jobs -rp | wc -l)" -ge "$MAXJ" ]; do sleep 5; done
  ( echo "START $tag $(date +%H:%M:%S)"
    CUDA_VISIBLE_DEVICES=$gpu OMP_NUM_THREADS=2 python3 revision/al_harness.py \
      --tag "$tag" --seeds "$SEEDS" "$@" > "revision/logs/${tag}.log" 2>&1 \
      && echo "DONE $tag $(date +%H:%M:%S)" || echo "FAIL $tag $(date +%H:%M:%S)" ) &
  sleep 2; }
echo "=== BOLD 470 START $(date +%F\ %H:%M:%S) ==="
# the published Bold protocol: LCBDS + max-sigma + max-mu, amplified surprise
run B_lcbds_bold   0 --protocol "lcbds+maxsigma+maxmu" $BOLD
# same budget, same batch size, standard acquisition -- the missing control
run B_ei_bold      1 --protocol "ei+maxsigma+maxmu"    $BOLD
run B_ei_pure      0 --protocol "ei+ei+ei"             --n-iter 150
run B_greedy_bold  1 --protocol "greedy+greedy+greedy" --n-iter 150
run B_random_bold  0 --protocol "random+random+random" --n-iter 150
# is the third (max-mu) query doing anything?  same budget, two queries x 235 iters
run B_lcbds_2q     1 --protocol "lcbds+maxsigma" --n-iter 225 --gamma 10 --delta 25 \
                     --beta 2.5 --beta-min 0.05 --surp-radius 0.15 --surp-thresh-sigma 1.5
wait
echo "=== BOLD 470 DONE $(date +%F\ %H:%M:%S) ==="
