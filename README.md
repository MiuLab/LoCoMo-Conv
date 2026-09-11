# LoCoMo-Conv

Code and data for **"When Users Don't Ask: Benchmarking Context-Driven Memory Retrieval in
Conversational Agents"** (LoCoMo-Conv).

LoCoMo-Conv recasts the LoCoMo QA pool into four conversational query styles — **dialog**,
**implicit**, **counterfactual**, and **composed** — while keeping the original gold answers
and evidence `dia_ids` fixed, and evaluates memory systems on both retrieval recall and
free-form response quality.

## Data

| File | Content |
|---|---|
| `data/locomo10.json` | Original LoCoMo10 conversations (Maharana et al., 2024) |
| `data/locomo10_dialog.json` | LoCoMo-Conv: per-QA conversational rewrites (`dialog_query`, `implicit_query`, `counterfactual_query`, `expected_memory_use`, `supportive_memory`) |
| `data/locomo10_multimem_full.json` | 1,069 composed multi-memory clusters (two source QAs each) |
| `data/response_eval_full_ext.json` | Full evaluation job list (all styles + 1,069 composed) |

## Repository layout

- `construction/` — dataset construction: query rewriting (`rewrite_qa_to_dialog.py`),
  composed-cluster mining and extension, leakage repair, rewrite validation.
- `retrieval/` — per-system ingestion + retrieval (`run_{bm25,dense,amem,mem0}.py`,
  `run_memora_retrieval.py`, `benchmark_memora_locomo10_index*.py`, `run_nemori_retrieval.py`),
  multi-facet query rewriting (`rewrite_queries_divergent.py`, `divergent_retrieve.py`), and
  recall computation (`compute_retrieval_metrics.py`, `compute_rewrite_recall.py`).
- `response_eval/` — response generation (`run_response_eval.py`: top-K / oracle / no-memory /
  query-rewriting / CoT variants) and LLM judges (`score_fact_used_partial.py`,
  `score_counterfactual_3way.py`, `score_composed_atomic.py`, hallucination, pairwise
  three-dimension judges, cross-judge agreement).
- `analysis/` — oracle-ceiling decomposition, recall-vs-fact bucketing, table compilation.
- `annotation/` — human-annotation packet builders and the exact Label Studio labeling
  configurations (`label_studio_configs/`).
- `variance/` — index-rebuild (3-seed) and answer-sampling (5-seed) variance runs.

## Setup

```bash
pip install openai httpx tqdm sentence-transformers chromadb rank_bm25
```

Models used in the paper:
- **Answer model**: `gemma-4-31B-it` served through any OpenAI-compatible endpoint
  (set `ANSWER_BASE_URL`, or pass `--answer_base_url`; the API key is read from
  `OPENAI_API_KEY`).
- **Rewriting / judging**: `gpt-5.4-mini` (key in `~/.openai-key` or `OPENAI_API_KEY`).
- **Pairwise quality judge**: `claude-opus-4-7` (key in `~/.anthropic-key` or
  `ANTHROPIC_API_KEY`).
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` (local).

Memory systems (A-MEM, mem0, AnchorMem, Memora, Nemori) are installed from their own
repositories; the run scripts in `retrieval/` follow each system's distributed
implementation with the embedding model and backbone LLM unified as above.

## Typical pipeline

```bash
# 1. (Optional) regenerate the conversational rewrites
python construction/rewrite_qa_to_dialog.py --input data/locomo10.json --output data/locomo10_dialog.json

# 2. Ingest + retrieve for one system (example: mem0)
python retrieval/run_mem0.py --top_k 10

# 3. Retrieval recall
python retrieval/compute_retrieval_metrics.py --results_dir <outputs_dir>

# 4. Generate responses (top-K / oracle / no-memory / CoT variants)
python response_eval/run_response_eval.py --sample_path data/response_eval_full_ext.json \
  --variants top_k,oracle --styles dialog,implicit,counterfactual,composed

# 5. Judge
python response_eval/score_fact_used_partial.py --responses_path <dir>/responses.json --output <dir>/fact_used_partial.json
python response_eval/score_counterfactual_3way.py --responses_path <dir>/responses.json --output <dir>/cf_3way.json
python response_eval/score_composed_atomic.py --responses_path <dir>/responses.json \
  --multimem_path data/locomo10_multimem_full.json --output <dir>/composed_atomic.json
```

## Citation

```bibtex
@misc{chang2026usersdontaskbenchmarking,
      title={When Users Don't Ask: Benchmarking Context-Driven Memory Retrieval in Conversational Agents},
      author={Wen-Yu Chang and Yun-Nung Chen},
      year={2026},
      eprint={2609.03467},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2609.03467},
}
```

## License / attribution

The underlying conversations and QA annotations come from
[LoCoMo](https://github.com/snap-research/locomo) (Maharana et al., 2024); LoCoMo-Conv adds
the conversational rewrites, composed clusters, and `supportive_memory` annotations on top.
Please cite both papers when using this benchmark.
