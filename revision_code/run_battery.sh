#!/bin/bash
# Controlled-experiment battery for the Digital Discovery revision.
# Every arm uses the same pool, the same K-means init, the same 10 seeds and
# the same 220-experiment budget unless explicitly stated.
cd "$(dirname "$0")/.." || exit 1
mkdir -p revision/logs revision/results revision/.locks

MAXJ=${MAXJ:-5}
SEEDS="0,1,2,3,4,10,11,12,13,14"

run () {   # run <tag> <gpu> <args...>
  local tag=$1; shift
  local gpu=$1; shift
  [ -f "revision/results/${tag}.json" ] && { echo "SKIP $tag"; return; }
  while [ "$(jobs -rp | wc -l)" -ge "$MAXJ" ]; do sleep 5; done
  ( mkdir "revision/.locks/$tag" 2>/dev/null || exit 0
    echo "START $tag $(date +%H:%M:%S)"
    CUDA_VISIBLE_DEVICES=$gpu OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
      python3 revision/al_harness.py --tag "$tag" --seeds "$SEEDS" "$@" \
      > "revision/logs/${tag}.log" 2>&1 \
      && echo "DONE $tag $(date +%H:%M:%S)" || echo "FAIL $tag $(date +%H:%M:%S)"
  ) &
  sleep 2
}

echo "=== BATTERY START $(date +%F\ %H:%M:%S) ==="

# ── Battery 2 (cheap, GP surrogate: Editor objection #2 / Referee 1 Q3) ──
run E2_gp_ei            0 --surrogate gp --protocol "ei+ei"        --n-iter 100
run E2_gp_ei_maxsigma   0 --surrogate gp --protocol "ei+maxsigma"  --n-iter 100
run E2_gp_lcb           1 --surrogate gp --protocol "lcb+lcb"      --n-iter 100
run E2_gp_lcb_maxsigma  1 --surrogate gp --protocol "lcb+maxsigma" --n-iter 100
run E2_gp_random        0 --surrogate gp --protocol "random+random" --n-iter 100

# ── Battery 1 (Editor objection #1: budget-matched component ablation) ──
run E1_full             0 --protocol "lcbds+maxsigma" --n-iter 100 --gamma 8  --delta 12
run E1_nodiv            1 --protocol "lcbds+maxsigma" --n-iter 100 --gamma 0  --delta 12
run E1_nosurp           0 --protocol "lcbds+maxsigma" --n-iter 100 --gamma 8  --delta 0
run E1_nodivsurp        1 --protocol "lcbds+maxsigma" --n-iter 100 --gamma 0  --delta 0
run E1_lcbds_rand2      0 --protocol "lcbds+random"   --n-iter 100 --gamma 8  --delta 12
run E1_lcbds_x2         1 --protocol "lcbds+lcbds"    --n-iter 100 --gamma 8  --delta 12
run E1_greedy           0 --protocol "greedy+greedy"  --n-iter 100
run E1_greedy_maxsigma  1 --protocol "greedy+maxsigma" --n-iter 100
run E1_random           0 --protocol "random+random"  --n-iter 100
# budget-matched single-query arms (n_iter 200 x 1 query = 220 labels)
run E1_Aonly_matched    1 --protocol "lcbds" --n-iter 200 --gamma 8 --delta 12
run E1_pureLCB_matched  0 --protocol "lcb"   --n-iter 200 --gamma 0 --delta 0

# ── Battery 2b (same acquisition family on the PC-BAN surrogate) ──
run E2_pcban_ei          1 --protocol "ei+ei"       --n-iter 100
run E2_pcban_ei_maxsigma 0 --protocol "ei+maxsigma" --n-iter 100

wait
echo "=== BATTERY DONE $(date +%F\ %H:%M:%S) ==="
