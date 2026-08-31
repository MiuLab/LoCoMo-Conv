"""Run AnchorMem.retrieve() with rewritten queries per sample.

Reuses existing cached embedding/fact stores from outputs/locomo-gemma-4-31B-it/sample_<i>/
so we don't re-extract facts. Only the retrieve() call is run — no answer generation.

Output: appends "AnchorMem|<style>|<sample>|<q_id>" → [doc, ...] to
        outputs_response_eval/<run>/rewrite_retrievals.json
"""
from __future__ import annotations

import sys, os, json
sys.path.append(".")
import multiprocessing as _mp
try:
    _mp.set_start_method("fork", force=True)
except RuntimeError:
    pass

import argparse
from collections import defaultdict
from typing import Dict, List

from AnchorMem import AnchorMem
from src.datasets.locomo10_loader import make_docs_from_locomo10_conversations
from src.utils.config_utils import BaseConfig


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--locomo_path", default="data/locomo10.json")
    p.add_argument("--sample_path", default="data/response_eval_sample.json")
    p.add_argument("--rewrites_path", default="outputs_response_eval/oracle_all_styles/rewrites.json")
    p.add_argument("--anchormem_save_dir", default="outputs/locomo-gemma-4-31B-it",
                   help="Per-sample cache dir from the original AnchorMem run.")
    p.add_argument("--output_dir", default="outputs_response_eval/oracle_all_styles")
    p.add_argument("--llm_base_url",
                   default="http://localhost:8000/v1")
    p.add_argument("--llm_name", default="./gemma-4-31B-it")
    p.add_argument("--embedding_name", default="Transformers/all-MiniLM-L6-v2")
    p.add_argument("--top_k", type=int, default=20)
    args = p.parse_args()

    with open(args.locomo_path) as f:
        locomo = json.load(f)
    sampled = json.load(open(args.sample_path))
    rewrites = json.load(open(args.rewrites_path))

    # Group jobs by sample_idx
    jobs = []  # (style, sample_idx, q_id_str, rewritten_query)
    for style, items in sampled["samples"].items():
        for it in items:
            if style == "composed":
                orig = it.get("composed_query")
                q_id = str(it.get("cluster_id"))
            else:
                orig = it.get("query")
                q_id = str(it.get("q_idx"))
            if not orig:
                continue
            rw = rewrites.get(orig, orig)
            jobs.append((style, it.get("sample_idx"), q_id, rw))

    by_sample = defaultdict(list)
    for j in jobs:
        by_sample[j[1]].append(j)
    print(f"Total jobs: {len(jobs)} across {len(by_sample)} samples", flush=True)

    rr_path = os.path.join(args.output_dir, "rewrite_retrievals.json")
    existing: Dict[str, List[str]] = json.load(open(rr_path)) if os.path.exists(rr_path) else {}

    for s_idx in sorted(by_sample.keys()):
        sample_save_dir = os.path.join(args.anchormem_save_dir, f"sample_{s_idx}")
        if not os.path.isdir(sample_save_dir):
            print(f"[sample {s_idx}] cache dir missing: {sample_save_dir}", flush=True)
            continue

        # Build docs (same as main.py)
        docs, _ = make_docs_from_locomo10_conversations(
            samples=[locomo[s_idx]],
            per_session=True,
            only_referenced=True,
            overlap=1,
        )

        config = BaseConfig(
            save_dir=sample_save_dir,
            llm_base_url=args.llm_base_url,
            llm_name=args.llm_name,
            dataset="locomo",
            embedding_model_name=args.embedding_name,
            embedding_base_url=None,
            force_index_from_scratch=False,
            force_fact_extraction_from_scratch=False,
            rerank_dspy_file_path="src/hipporag/prompts/dspy_prompts/filter_llama3.3-70B-Instruct.json",
            retrieval_top_k=args.top_k,
            fact_sim_threshold=0.85,
            related_fact_top_k=3,
            linking_top_k=5,
            max_qa_steps=3,
            qa_top_k=args.top_k,
            embedding_batch_size=8,
            max_new_tokens=256,
            corpus_len=len(docs),
        )

        print(f"[sample {s_idx}] loading AnchorMem...", flush=True)
        anchormem = AnchorMem(global_config=config)
        anchormem.index(docs)  # idempotent: reloads from cache

        sample_jobs = by_sample[s_idx]
        queries = [j[3] for j in sample_jobs]
        print(f"[sample {s_idx}] retrieving for {len(queries)} rewritten queries...", flush=True)
        results = anchormem.retrieve(queries)

        for (style, _, q_id, _q), sol in zip(sample_jobs, results):
            key = f"AnchorMem|{style}|{s_idx}|{q_id}"
            existing[key] = (sol.docs or [])[: args.top_k]

        # Save after each sample (resume-friendly)
        with open(rr_path, "w") as f:
            json.dump(existing, f, ensure_ascii=False, indent=2)
        print(f"[sample {s_idx}] saved (total entries: {len(existing)})", flush=True)


if __name__ == "__main__":
    main()
