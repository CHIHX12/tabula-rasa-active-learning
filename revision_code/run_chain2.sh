#!/bin/bash
# Runs after the main experiment chain: the surrogate-capability study that
# asks where, if anywhere, PC-BAN beats a simple composition-only baseline.
cd "$(dirname "$0")/.." || exit 1
echo "[chain2] waiting for chain 1 ..."
until grep -q "ALL DONE" revision/logs/chain.log 2>/dev/null; do sleep 60; done
echo "[chain2] chain 1 finished $(date +%H:%M:%S)"

OMP_NUM_THREADS=4 python3 revision/capability_study.py \
  --regimes tail,region,random \
  --extrap-sizes 60,220 \
  --seeds 0,1,2,3,4 --region-seeds 0,1 \
  --out capability_study.json > revision/logs/capability_study.log 2>&1
echo "[chain2] capability study done $(date +%F\ %H:%M:%S)"
