#!/usr/bin/env bash
# Block A: response-side variance runs.
#
# For a given seed, generate Gemma responses (temp=0.7) using the SAME
# retrieval outputs as the original main table (variant=top_k, k=10, all
# 5 systems × 4 styles). Outputs go to outputs_response_eval/blockA_seed{N}/.
#
# Usage:
#   ./blockA_orchestrator.sh <seed> [temperature] [concurrency]
#
# Example:
#   bash baselines/AnchorMem/scripts/blockA_orchestrator.sh 43

set -u

SEED="${1:-43}"
TEMP="${2:-0.7}"
CONCURRENCY="${3:-32}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ANCHOR_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

NEW_URL="${LLM_BASE_URL:-http://localhost:8000/v1}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"

OUT_DIR="outputs_response_eval/blockA_seed${SEED}"
LOG_DIR="logs/blockA_seed${SEED}"

cd "$ANCHOR_DIR"
mkdir -p "$OUT_DIR" "$LOG_DIR"

echo "=========================================="
echo "Block A response-side variance run"
echo "Seed:        $SEED"
echo "Temperature: $TEMP"
echo "Concurrency: $CONCURRENCY"
echo "Endpoint:    $NEW_URL"
echo "Output:      $OUT_DIR"
echo "=========================================="

START=$(date +%s)

python scripts/run_response_eval.py \
  --sample_path data/response_eval_full.json \
  --dataset_path data/locomo10_dialog.json \
  --multimem_path data/locomo10_multimem.json \
  --variants top_k \
  --styles dialog,implicit,counterfactual,composed \
  --ks 10 \
  --systems "AnchorMem,A-mem,mem0,BM25,Dense" \
  --answer_model ../gemma-4-31B-it \
  --answer_base_url "$NEW_URL" \
  --output_dir "$OUT_DIR" \
  --concurrency "$CONCURRENCY" \
  --max_tokens 300 \
  --checkpoint_every 500 \
  --seed "$SEED" \
  --temperature "$TEMP" \
  2>&1 | tee "$LOG_DIR/response_gen.log"

END=$(date +%s)
ELAPSED=$((END - START))
echo "=========================================="
echo "Response gen wall-clock: ${ELAPSED}s"
echo "Records:"
python3 -c "import json; d=json.load(open('$OUT_DIR/responses.json')); print(f'  total: {len(d)}'); from collections import Counter; print('  by system:', dict(Counter(r[\"system\"] for r in d))); print('  by style:', dict(Counter(r[\"style\"] for r in d)))" 2>/dev/null || echo "  (responses.json not yet written)"
echo "=========================================="
