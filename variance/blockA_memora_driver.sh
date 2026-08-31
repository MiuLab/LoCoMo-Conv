#!/usr/bin/env bash
# Sequential Block A for Memora: seeds 43-47, retry until zero errored responses.
cd "$(dirname "$0")/.."
export OPENAI_API_KEY="${OPENAI_API_KEY:?set OPENAI_API_KEY for the answer-model endpoint}"
PY=~/.pyenv/versions/3.12.8/bin/python3
for S in 43 44 45 46 47; do
  OUT=outputs_response_eval/blockA_memora_seed${S}
  for pass in 1 2 3 4 5 6; do
    $PY scripts/run_response_eval.py \
      --sample_path data/response_eval_full.json --dataset_path data/locomo10_dialog.json \
      --multimem_path data/locomo10_multimem.json \
      --variants top_k --ks 10 --styles dialog,implicit,counterfactual,composed --systems memora \
      --answer_base_url https://integrate.api.nvidia.com/v1 --answer_model google/gemma-4-31b-it \
      --output_dir $OUT --concurrency 4 --max_tokens 300 --seed $S --temperature 0.7 >> logs_blockA_memora_driver.log 2>&1
    NERR=$($PY - <<PYE
import json
p="$OUT/responses.json"; d=json.load(open(p))
good=[x for x in d if not x.get("error") and (x.get("response") or "").strip()]
json.dump(good, open(p,"w"), ensure_ascii=False)
print(len(d)-len(good))
PYE
)
    echo "seed $S pass $pass: errors=$NERR kept=$(($(wc -c < $OUT/responses.json) > 0))" >> logs_blockA_memora_driver.log
    [ "$NERR" = "0" ] && break
    sleep 60
  done
done
echo "DRIVER DONE" >> logs_blockA_memora_driver.log
