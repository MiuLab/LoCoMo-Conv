#!/usr/bin/env bash
# Block B extension: 4 conversational-style retrieval on the seeded indices.
#
# mem0    — search_only against persisted per-sample chroma (zero LLM).
#           seed43 sample_0 is a special case: its memories live in the
#           legacy shared chroma at the seed dir top level.
# A-mem   — re-ingest locally with --disable_evolution (embedding only, zero
#           LLM; retrieval provably invariant to evolution metadata).
#
# AnchorMem is launched separately (needs its own retrieval path via main.py).
#
# Usage: bash retrieval/blockB_conv_retrieval.sh

set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ANCHOR_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ANCHOR_DIR"

export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
LLM_URL="http://localhost:8000/v1"
STYLES="dialog,implicit,counterfactual,composed"

echo "=== mem0 search_only: 3 seeds × 10 samples ==="
for SEED in 43 44 45; do
  for S in 0 1 2 3 4 5 6 7 8 9; do
    EXTRA=""
    if [ "$SEED" = "43" ] && [ "$S" = "0" ]; then
      EXTRA="--chroma_override outputs_mem0/locomo-gemma-4-31B-it_seed43_t07/_chroma_state"
    fi
    python scripts/run_mem0.py \
      --llm_base_url "$LLM_URL" \
      --llm_model ../gemma-4-31B-it \
      --dataset_path data/locomo10_dialog.json \
      --multimem_path data/locomo10_multimem.json \
      --samples "$S" \
      --styles "$STYLES" \
      --save_dir_suffix "_seed${SEED}_t07" \
      --save_dir outputs_mem0 \
      --search_only $EXTRA \
      2>&1 | grep -E "wrote|skipping|search_only|failed" | sed "s/^/[mem0 seed$SEED s$S] /"
  done
done

echo ""
echo "=== A-mem no-evo re-ingest + 4-style retrieval: 3 seeds ==="
for SEED in 43 44 45; do
  python scripts/run_amem.py \
    --llm_base_url "$LLM_URL" \
    --llm_model ../gemma-4-31B-it \
    --dataset_path data/locomo10_dialog.json \
    --locomo_path data/locomo10.json \
    --multimem_path data/locomo10_multimem.json \
    --seed "$SEED" --temperature 0.7 \
    --samples all \
    --styles "$STYLES" \
    --save_dir_suffix "_seed${SEED}_t07" \
    --save_dir outputs_amem \
    --disable_evolution \
    2>&1 | grep -E "wrote|skipping|ingesting|failed" | sed "s/^/[amem seed$SEED] /"
done

echo ""
echo "ALL DONE"
