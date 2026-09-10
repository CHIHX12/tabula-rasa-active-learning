#!/bin/bash
# Improvement campaign.
#
# The diagnostic (revision/diag_terms.py) showed why every ablation of the
# acquisition function came out null:
#   * delta*s(x) is identically ZERO for the whole campaign, because a surprise
#     is triggered only when the residual exceeds 2 x sd(labels) ~ 60 mV, which
#     essentially never happens.  delta multiplies a zero vector.
#   * beta*sigma decays to 0.70 mV against a pool spread of mu of 11.4 mV, i.e.
#     6%, exactly when escaping a local optimum matters most.
#   * gamma*d contributes ~3.2 mV against the same 11.4 mV.
#   * by the last iteration, removing all three extra terms changes only 1 of
#     the top 20 candidates: LCBDS has become greedy.
#
# Five fixes, tested one at a time and then combined, all against the published
# configuration on the same ten seeds and the same 220-experiment budget.
cd "$(dirname "$0")/.." || exit 1
MAXJ=4
S="0,1,2,3,4,10,11,12,13,14"
run () { local tag=$1; shift; local gpu=$1; shift
  [ -f "revision/results/${tag}.json" ] && { echo "SKIP $tag"; return; }
  while [ "$(jobs -rp | wc -l)" -ge "$MAXJ" ]; do sleep 10; done
  ( echo "START $tag $(date +%H:%M:%S)"
    CUDA_VISIBLE_DEVICES=$gpu OMP_NUM_THREADS=2 python3 revision/al_harness.py \
      --tag "$tag" --seeds "$S" --protocol "lcbds+maxsigma" --n-iter 100 "$@" \
      > "revision/logs/${tag}.log" 2>&1 \
      && echo "DONE $tag $(date +%H:%M:%S)" || echo "FAIL $tag $(date +%H:%M:%S)" ) &
  sleep 3; }
echo "=== IMPROVE START $(date +%F\ %H:%M:%S) ==="
# single factors (published weights kept where the units are unchanged)
run I_B_sigcal   0 --sigma-calib --gamma 8 --delta 12
run I_C_k1       1 --n-components 1 --gamma 8 --delta 12
run I_D_warm     0 --warm-epochs -1 --gamma 8 --delta 12
run I_E_surplocal 1 --surp-mode local --gamma 8 --delta 12
# standardized acquisition needs weights in units of sd(mu); sweep three points
run I_A_std_g03  0 --acq-scale standardized --gamma 0.3 --delta 0.5 --beta 2.5 --beta-min 0.2
run I_A_std_g10  1 --acq-scale standardized --gamma 1.0 --delta 1.0 --beta 2.5 --beta-min 0.2
run I_A_std_g05b 0 --acq-scale standardized --gamma 0.5 --delta 0.5 --beta 2.5 --beta-min 1.0
# combinations
run I_AB         1 --acq-scale standardized --sigma-calib --gamma 0.5 --delta 0.5
run I_ABE        0 --acq-scale standardized --sigma-calib --surp-mode local --gamma 0.5 --delta 0.5
run I_ALL        1 --acq-scale standardized --sigma-calib --surp-mode local \
                    --n-components 1 --warm-epochs -1 --gamma 0.5 --delta 0.5
wait
echo "=== IMPROVE DONE $(date +%F\ %H:%M:%S) ==="
