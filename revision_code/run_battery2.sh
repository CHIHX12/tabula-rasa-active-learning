#!/bin/bash
# Wave 2: surrogate ablation, retraining-protocol ablation, and a
# single-code-path re-run of the cross-domain generalization.
cd "$(dirname "$0")/.." || exit 1
mkdir -p revision/logs revision/results revision/.locks

MAXJ=${MAXJ:-5}
SEEDS10="0,1,2,3,4,10,11,12,13,14"
SEEDS5="0,1,2,3,4"

run () {
  local tag=$1; shift
  local gpu=$1; shift
  [ -f "revision/results/${tag}.json" ] && { echo "SKIP $tag"; return; }
  while [ "$(jobs -rp | wc -l)" -ge "$MAXJ" ]; do sleep 5; done
  ( mkdir "revision/.locks/$tag" 2>/dev/null || exit 0
    echo "START $tag $(date +%H:%M:%S)"
    CUDA_VISIBLE_DEVICES=$gpu OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
      python3 revision/al_harness.py --tag "$tag" "$@" \
      > "revision/logs/${tag}.log" 2>&1 \
      && echo "DONE $tag $(date +%H:%M:%S)" || echo "FAIL $tag $(date +%H:%M:%S)"
  ) &
  sleep 2
}

echo "=== BATTERY2 START $(date +%F\ %H:%M:%S) ==="

# ── Referee 1 Q4: swap the surrogate, hold the LCBDS protocol constant ──
run E3_rf        0 --seeds "$SEEDS10" --surrogate rf  --protocol "lcbds+maxsigma" --n-iter 100
run E3_gp_lcbds  1 --seeds "$SEEDS10" --surrogate gp  --protocol "lcbds+maxsigma" --n-iter 100
run E3_mlp       0 --seeds "$SEEDS10" --surrogate mlp --protocol "lcbds+maxsigma" --n-iter 100
run E3_mlpdesc   1 --seeds "$SEEDS10" --surrogate mlp_descriptor --protocol "lcbds+maxsigma" --n-iter 100

# ── Referee 1 Q1: retraining protocol (5 seeds; 'scratch' is ~4x the cost) ──
run E4_warm      0 --seeds "$SEEDS5" --protocol "lcbds+maxsigma" --n-iter 100 --train-mode warm
run E4_scratch   1 --seeds "$SEEDS5" --protocol "lcbds+maxsigma" --n-iter 100 --train-mode scratch

# ── Cross-domain generalization through the SAME code path as OER ──
for ds in steel concrete slump wine_red energy; do
  run G_${ds}_lcbds   0 --seeds "$SEEDS5" --dataset $ds --protocol "lcbds+maxsigma" --gamma-frac 0.02 --delta-frac 0.03
  run G_${ds}_greedy  1 --seeds "$SEEDS5" --dataset $ds --protocol "greedy+greedy"
  run G_${ds}_random  0 --seeds "$SEEDS5" --dataset $ds --protocol "random+random"
  run G_${ds}_gpei    1 --seeds "$SEEDS5" --dataset $ds --surrogate gp --protocol "ei+ei"
  run G_${ds}_gpei_ms 0 --seeds "$SEEDS5" --dataset $ds --surrogate gp --protocol "ei+maxsigma"
  run G_${ds}_grd_ms  1 --seeds "$SEEDS5" --dataset $ds --protocol "greedy+maxsigma"
done

wait
echo "=== BATTERY2 DONE $(date +%F\ %H:%M:%S) ==="
