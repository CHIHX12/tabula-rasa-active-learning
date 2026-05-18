#!/bin/bash
# run_all_experiments.sh
# Reproduce all experiments reported in the paper.
# Estimated total runtime: ~6-10 hours on a single GPU (NVIDIA RTX-class).
# Each config runs 5 seeds × 3 strategies × n_iter iterations.

set -e
PYTHON=python3

echo "============================================"
echo "  Reproducing all paper experiments"
echo "============================================"

# ── 1. Main comparison (10 seeds each)
echo "[1/4] Main AL comparison (seeds 0-4)..."
$PYTHON main_active_learning.py --config configs/al_aggressive.yaml
$PYTHON main_active_learning.py --config configs/al_bold_206.yaml

echo "[2/4] Main AL comparison (seeds 10-14)..."
$PYTHON main_active_learning.py --config configs/al_aggressive_s10.yaml
$PYTHON main_active_learning.py --config configs/al_bold_206_s10.yaml

# ── 2. Ablation study (5 seeds each)
echo "[3/4] Component ablation study..."
for cfg in al_no_mc al_no_annealing al_no_diversity al_no_surprise \
           al_single_query al_pure_lcbds al_no_extras; do
    echo "  Running: $cfg"
    $PYTHON main_active_learning.py --config configs/${cfg}.yaml
done

# ── 3. Hyperparameter sweeps (5 seeds each)
echo "[4/4] Hyperparameter sweeps..."
for cfg in al_beta_0p05 al_beta_0p1 al_beta_0p3 al_beta_0p5 al_beta_1p0 \
           al_gamma_0 al_gamma_3 al_gamma_5 al_gamma_10 al_gamma_15 \
           al_delta_0 al_delta_5 al_delta_8 al_delta_15 al_delta_20; do
    echo "  Running: $cfg"
    $PYTHON main_active_learning.py --config configs/${cfg}.yaml
done

echo ""
echo "============================================"
echo "  All experiments complete."
echo "  Results saved to: results/"
echo "============================================"

# ── Copy fresh results to precomputed_results/ for archival
echo "Copying fresh results to precomputed_results/..."
cp results/active_learning_v6_*.json precomputed_results/

# ── Generate figures from fresh results
echo "Generating figures..."
$PYTHON plot_journal_figures.py
echo "Figures saved to: figures/"
