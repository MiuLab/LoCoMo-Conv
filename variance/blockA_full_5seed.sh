#!/usr/bin/env bash
# Serial 5-seed Block A full-set runner.
# Runs response gen for seeds 43..47, then scores fact_used_partial for each.
#
# Usage:
#   bash baselines/AnchorMem/scripts/blockA_full_5seed.sh [start_seed] [n_seeds]
#
# Default: seeds 43, 44, 45, 46, 47.

set -u

START_SEED="${1:-43}"
N_SEEDS="${2:-5}"
TEMP="0.7"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ANCHOR_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "=========================================="
echo "Block A 5-seed full-set orchestrator"
echo "Seeds: $START_SEED..$((START_SEED + N_SEEDS - 1))"
echo "Temperature: $TEMP"
echo "=========================================="

OVERALL_START=$(date +%s)

for i in $(seq 0 $((N_SEEDS - 1))); do
  SEED=$((START_SEED + i))
  OUT_DIR="$ANCHOR_DIR/outputs_response_eval/blockA_seed${SEED}"

  # ---- Response gen ----
  if [ -f "$OUT_DIR/responses.json" ]; then
    N=$(python3 -c "import json; print(len(json.load(open('$OUT_DIR/responses.json'))))" 2>/dev/null || echo 0)
    if [ "$N" = "29060" ]; then
      echo ""
      echo "--- seed $SEED response gen: SKIP (already have $N records) ---"
    else
      echo ""
      echo "--- seed $SEED response gen (partial $N, redoing) ---"
      rm -rf "$OUT_DIR"
      bash "$SCRIPT_DIR/blockA_orchestrator.sh" "$SEED" "$TEMP" 32
    fi
  else
    echo ""
    echo "--- seed $SEED response gen ---"
    bash "$SCRIPT_DIR/blockA_orchestrator.sh" "$SEED" "$TEMP" 32
  fi

  # ---- Score fact_used_partial (dialog + implicit only, since main table uses this metric on those styles) ----
  SCORED="$OUT_DIR/fact_used_partial.json"
  if [ -f "$SCORED" ]; then
    echo "--- seed $SEED fact_used: SKIP (already scored) ---"
  else
    echo "--- seed $SEED fact_used scoring ---"
    cd "$ANCHOR_DIR/../.."
    python "$ANCHOR_DIR/scripts/score_fact_used_partial.py" \
      --responses_path "$OUT_DIR/responses.json" \
      --output "$SCORED" \
      --styles dialog,implicit \
      --variants top_k \
      --concurrency 24
  fi
done

OVERALL_END=$(date +%s)
echo ""
echo "=========================================="
echo "All seeds done in $((OVERALL_END - OVERALL_START))s"
echo "=========================================="
