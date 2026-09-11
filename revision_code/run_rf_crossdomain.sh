#!/bin/bash
# Random forest against PC-BAN across the five cross-domain datasets.
#
# The paper's surrogate claim rests on high dimensionality: a Gaussian process
# collapses at N = 13 while the descriptor-free network holds.  But the only
# random-forest comparison in the paper is at N = 4, where it ties the network
# (p = 0.92).  A referee will ask immediately whether the forest also holds at
# N = 13, and the paper cannot answer.  This closes that gap: same datasets,
# same protocol, same initial design, same ten seeds, only the surrogate
# changes.
cd "$(dirname "$0")/.." || exit 1
mkdir -p revision/logs revision/results

MAXJ=5
run () {  # run <dataset> <n_init> <n_iter> <gamma> <delta> <direction-args...>
  local ds=$1 ni=$2 it=$3 ga=$4 de=$5; shift 5
  for seeds_tag in "0,1,2,3,4:R_${ds}_rf" "5,6,7,8,9:RP_${ds}_rf"; do
    local sd=${seeds_tag%%:*} tag=${seeds_tag##*:}
    [ -f "revision/results/${tag}.json" ] && { echo "SKIP $tag"; continue; }
    while [ "$(jobs -rp | wc -l)" -ge "$MAXJ" ]; do sleep 5; done
    ( echo "START $tag $(date +%H:%M:%S)"
      OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES="" python3 revision/al_harness.py \
        --tag "$tag" --dataset "$ds" --surrogate rf --protocol "lcbds+maxsigma" \
        --seeds "$sd" --n-init "$ni" --n-iter "$it" --gamma "$ga" --delta "$de" "$@" \
        > "revision/logs/${tag}.log" 2>&1 \
        && echo "DONE $tag $(date +%H:%M:%S)" || echo "FAIL $tag $(date +%H:%M:%S)" ) &
    sleep 1
  done
}

echo "=== RF CROSS-DOMAIN START $(date +%F\ %H:%M:%S) ==="
run steel     12 20 30.088 45.132
run concrete  15 45  1.605  2.408
run slump      8 12  0.827  1.240
run wine_red  15 40  0.100  0.150
run energy    15 40  0.742  1.113
wait
echo "=== RF CROSS-DOMAIN DONE $(date +%F\ %H:%M:%S) ==="
