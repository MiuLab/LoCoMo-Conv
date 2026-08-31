#!/usr/bin/env bash
# Score counterfactual (3-way) and composed (atomic) for each Block A seed.
# Reuses the responses.json produced by blockA_orchestrator.sh.
#
# Usage: bash baselines/AnchorMem/scripts/blockA_score_cf_composed.sh

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ANCHOR_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ANCHOR_DIR/../.."

SEEDS=(43 44 45 46 47)

OVERALL_START=$(date +%s)
for SEED in "${SEEDS[@]}"; do
  DIR="$ANCHOR_DIR/outputs_response_eval/blockA_seed${SEED}"

  # --- Counterfactual 3-way ---
  CF_OUT="$DIR/cf_3way.json"
  if [ -f "$CF_OUT" ]; then
    echo "--- seed $SEED CF-3way: SKIP (already have $(python3 -c "import json;print(len(json.load(open('$CF_OUT'))))" 2>/dev/null || echo 0)) ---"
  else
    echo "--- seed $SEED CF-3way scoring ---"
    python "$ANCHOR_DIR/scripts/score_counterfactual_3way.py" \
      --responses_path "$DIR/responses.json" \
      --output "$CF_OUT" \
      --variants top_k \
      --concurrency 24
  fi

  # --- Composed atomic ---
  COMP_OUT="$DIR/composed_atomic.json"
  if [ -f "$COMP_OUT" ]; then
    echo "--- seed $SEED composed atomic: SKIP ---"
  else
    echo "--- seed $SEED composed atomic scoring ---"
    python "$ANCHOR_DIR/scripts/score_composed_atomic.py" \
      --responses_path "$DIR/responses.json" \
      --multimem_path "$ANCHOR_DIR/data/locomo10_multimem.json" \
      --output "$COMP_OUT" \
      --variants top_k \
      --concurrency 16
  fi
done

OVERALL_END=$(date +%s)
echo ""
echo "All CF+composed scoring done in $((OVERALL_END - OVERALL_START))s"
