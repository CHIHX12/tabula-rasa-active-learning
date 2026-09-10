#!/bin/bash
# Extend the continuous-simplex experiment from five seeds to ten.
#
# Section 3.5 of this paper criticises its own submitted version for reporting
# five-seed comparisons, at which the two-sided Wilcoxon p-value cannot fall
# below 0.0625.  Section 3.6 must not repeat that: the five existing seeds
# (0-4) are extended with seeds 10-14 so that every comparison is made at
# n = 10, the same as everywhere else in the paper.
cd "$(dirname "$0")/.." || exit 1
mkdir -p revision/logs revision/results

MAXJ=4
S="10,11,12,13,14"

run () {   # run <tag> <gpu> <oracle> <protocol>
  local tag=$1 gpu=$2 oracle=$3 proto=$4
  [ -f "revision/results/${tag}.json" ] && { echo "SKIP $tag"; return; }
  while [ "$(jobs -rp | wc -l)" -ge "$MAXJ" ]; do sleep 15; done
  ( echo "START $tag $(date +%H:%M:%S)"
    CUDA_VISIBLE_DEVICES=$gpu OMP_NUM_THREADS=2 python3 revision/continuous_space.py \
      --tag "$tag" --oracle "$oracle" --protocol "$proto" --seeds "$S" \
      > "revision/logs/${tag}.log" 2>&1 \
      && echo "DONE $tag $(date +%H:%M:%S)" || echo "FAIL $tag $(date +%H:%M:%S)" ) &
  sleep 3
}

echo "=== CONTINUOUS+ START $(date +%F\ %H:%M:%S) ==="
run CP_oer_twin_lcbds       1 oer_twin       "lcbds+maxsigma"
run CP_oer_twin_greedy      1 oer_twin       "greedy+greedy"
run CP_oer_twin_gpei        0 oer_twin       "ei+ei"
run CP_oer_twin_random      1 oer_twin       "random+random"
run CP_analytic_sharp_lcbds 0 analytic_sharp "lcbds+maxsigma"
run CP_analytic_sharp_greedy 1 analytic_sharp "greedy+greedy"
run CP_analytic_sharp_gpei  0 analytic_sharp "ei+ei"
run CP_analytic_sharp_random 1 analytic_sharp "random+random"
wait
echo "=== CONTINUOUS+ DONE $(date +%F\ %H:%M:%S) ==="
