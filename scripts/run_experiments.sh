#!/usr/bin/env bash
# End-to-end experiment reproduction.
#
# Usage (from repo root):
#   bash scripts/run_experiments.sh [--model MODEL] [--debug]
#
# Runs:
#   1. Training       → results/checkpoints/best_<model>.pt
#   2. XAI generation → results/figures/<model>/
#   3. Evaluation     → results/metrics/xai_comparison_table.csv

set -euo pipefail

MODEL="densenet121"
DEBUG_FLAG=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model) MODEL="$2"; shift 2 ;;
        --debug) DEBUG_FLAG="--debug"; shift ;;
        *) echo "Unknown flag: $1"; exit 1 ;;
    esac
done

echo "=== Phase 2: CXR-XAI-Clinical ==="
echo "Model : $MODEL"
echo "Debug : ${DEBUG_FLAG:-off}"
echo ""

# ── 1. Train ──────────────────────────────────────────────────────────────────
echo "[1/3] Training $MODEL ..."
python scripts/train.py \
    --config configs/train.yaml \
    --model "$MODEL" \
    $DEBUG_FLAG

# ── 2. Generate XAI heatmaps ─────────────────────────────────────────────────
echo "[2/3] Generating XAI heatmaps ..."
python scripts/generate_xai.py \
    --config configs/train.yaml \
    --xai-config configs/xai.yaml \
    --checkpoint "results/checkpoints/best_${MODEL}.pt" \
    --model "$MODEL"

# ── 3. Quantitative evaluation ────────────────────────────────────────────────
echo "[3/3] Running quantitative evaluation ..."
python scripts/evaluate.py \
    --config configs/eval.yaml \
    --xai-config configs/xai.yaml \
    --checkpoint "results/checkpoints/best_${MODEL}.pt" \
    --model "$MODEL"

echo ""
echo "=== Done ==="
echo "Metrics : results/metrics/xai_comparison_table.csv"
echo "Figures : results/figures/$MODEL/"
