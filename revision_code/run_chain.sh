#!/bin/bash
# Wait for wave 1, then run the remaining revision experiments in order.
cd "$(dirname "$0")/.." || exit 1

echo "[chain] waiting for wave 1 ..."
until grep -q "BATTERY DONE" revision/logs/battery.log 2>/dev/null; do sleep 30; done
echo "[chain] wave 1 done $(date +%H:%M:%S)"

MAXJ=5 ./revision/run_battery2.sh >> revision/logs/battery2.log 2>&1
echo "[chain] wave 2 done $(date +%H:%M:%S)"

# Loss-weight / MDN study
OMP_NUM_THREADS=4 python3 revision/loss_and_mdn_study.py \
  > revision/logs/loss_mdn.log 2>&1
echo "[chain] loss/MDN study done $(date +%H:%M:%S)"

# Continuous compositional space, no fixed pool
for orc in oer_twin analytic_sharp; do
  for spec in "lcbds+maxsigma:pcban:C_${orc}_lcbds" \
              "greedy+greedy:pcban:C_${orc}_greedy" \
              "random+random:pcban:C_${orc}_random" \
              "ei+ei:gp:C_${orc}_gpei"; do
    proto="${spec%%:*}"; rest="${spec#*:}"; sur="${rest%%:*}"; tag="${rest#*:}"
    [ -f "revision/results/${tag}.json" ] && { echo "SKIP $tag"; continue; }
    echo "[chain] continuous $tag $(date +%H:%M:%S)"
    OMP_NUM_THREADS=2 python3 revision/continuous_space.py \
      --tag "$tag" --oracle "$orc" --protocol "$proto" --surrogate "$sur" \
      --seeds 0,1,2,3,4 --n-iter 100 > "revision/logs/${tag}.log" 2>&1
  done
done
echo "[chain] ALL DONE $(date +%F\ %H:%M:%S)"
