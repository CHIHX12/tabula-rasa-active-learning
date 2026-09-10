#!/bin/bash
# The cross-domain study runs 5 seeds per dataset, which puts the two-sided
# Wilcoxon p-value floor at 0.0625: no per-dataset comparison can reach p<0.05
# however large the effect.  Add seeds 5-9 so that n = 10 and the floor drops
# to 0.002, which is enough to actually establish the one consistent
# cross-domain result (LCBDS+maxσ > greedy on out-of-pool R², 5/5 datasets).
cd "$(dirname "$0")/.." || exit 1
MAXJ=2
SEEDS="5,6,7,8,9"
run () { local tag=$1; shift; local gpu=$1; shift
  [ -f "revision/results/${tag}.json" ] && { echo "SKIP $tag"; return; }
  while [ "$(jobs -rp | wc -l)" -ge "$MAXJ" ]; do sleep 10; done
  ( echo "START $tag $(date +%H:%M:%S)"
    CUDA_VISIBLE_DEVICES=$gpu OMP_NUM_THREADS=2 python3 revision/al_harness.py \
      --tag "$tag" --seeds "$SEEDS" "$@" > "revision/logs/${tag}.log" 2>&1 \
      && echo "DONE $tag $(date +%H:%M:%S)" || echo "FAIL $tag $(date +%H:%M:%S)" ) &
  sleep 2; }
echo "=== GEN POWER START $(date +%F\ %H:%M:%S) ==="
g=0
# cheapest datasets first so results arrive early
for ds in slump steel energy concrete wine_red; do
  run GP_${ds}_lcbds   $((g%2)) --dataset $ds --protocol "lcbds+maxsigma" --gamma-frac 0.02 --delta-frac 0.03; g=$((g+1))
  run GP_${ds}_greedy  $((g%2)) --dataset $ds --protocol "greedy+greedy"; g=$((g+1))
  run GP_${ds}_grd_ms  $((g%2)) --dataset $ds --protocol "greedy+maxsigma"; g=$((g+1))
  run GP_${ds}_gpei_ms $((g%2)) --dataset $ds --surrogate gp --protocol "ei+maxsigma"; g=$((g+1))
  run GP_${ds}_random  $((g%2)) --dataset $ds --protocol "random+random"; g=$((g+1))
done
wait
echo "=== GEN POWER DONE $(date +%F\ %H:%M:%S) ==="
