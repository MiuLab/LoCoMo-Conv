"""Run AnchorMem retrieval on Style-5 composed (multi-memory) queries.

Reads data/locomo10_multimem_full.json (composed clusters across 10 samples) and runs
each cluster's composed_query through AnchorMem retrieval (reusing the indexing
cache built by main.py). Outputs:
  outputs/locomo-<llm>/sample_<i>/composed_solutions.json
each entry is the QuerySolution for one cluster's composed_query.

Recall@K / precision@K vs gold_dia_ids is computed in a separate post-process
script that maps retrieved docs back to dia_ids.
"""
from __future__ import annotations

import sys
sys.path.append(".")

# Same multiprocessing fix as main.py (macOS spawn → fork).
import multiprocessing as _mp
try:
    _mp.set_start_method("fork", force=True)
except RuntimeError:
    pass

import argparse
import dataclasses
import json
import logging
import os
import random
from collections import defaultdict
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from AnchorMem import AnchorMem
from src.utils.config_utils import BaseConfig
from src.datasets.locomo10_loader import make_docs_from_locomo10_conversations


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def _to_str(x: Any) -> str:
    if isinstance(x, list):
        return ", ".join(str(i) for i in x)
    return str(x)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--locomo_path", default="data/locomo10.json")
    p.add_argument("--multimem_path", default="data/locomo10_multimem_full.json")
    p.add_argument("--llm_base_url",
                   default="http://localhost:8000/v1")
    p.add_argument("--llm_name", default="./gemma-4-31B-it")
    p.add_argument("--embedding_name", default="Transformers/all-MiniLM-L6-v2")
    p.add_argument("--embedding_base_url", default=None)
    p.add_argument("--save_dir", default="outputs")
    p.add_argument("--dataset", default="locomo")
    p.add_argument("--seed", type=int, default=42, help="Random seed for module RNG and LLM sampling")
    p.add_argument("--temperature", type=float, default=0.0, help="LLM sampling temperature")
    p.add_argument("--save_dir_suffix", default="", help="Suffix appended to base_save_dir (e.g. _seed43_t07)")
    p.add_argument("--samples", default="all", help="comma-separated sample indices or 'all'")
    args = p.parse_args()

    global SEED
    SEED = args.seed
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    llm_name_for_path = args.llm_name.lstrip("./").replace("/", "_")
    base_save_dir = args.save_dir
    if base_save_dir == "outputs":
        base_save_dir = f"{base_save_dir}/{args.dataset}-{llm_name_for_path}"
    else:
        base_save_dir = f"{base_save_dir}_{args.dataset}-{llm_name_for_path}"
    base_save_dir += args.save_dir_suffix
    os.makedirs(base_save_dir, exist_ok=True)

    logger = logging.getLogger("composed_runner")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(h)
    logger.propagate = False

    with open(args.locomo_path) as f:
        locomo = json.load(f)
    with open(args.multimem_path) as f:
        clusters = json.load(f)

    # Group clusters by sample
    clusters_by_sample: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for c in clusters:
        if c and c.get("composed_query"):
            clusters_by_sample[c["sample_idx"]].append(c)

    wanted = None if args.samples == "all" else {int(x) for x in args.samples.split(",") if x.strip()}

    for s_idx in sorted(clusters_by_sample.keys()):
        if wanted is not None and s_idx not in wanted:
            continue
        sample_clusters = clusters_by_sample[s_idx]
        sample_data = locomo[s_idx]
        sample_save_dir = os.path.join(base_save_dir, f"sample_{s_idx}")
        os.makedirs(sample_save_dir, exist_ok=True)

        out_path = os.path.join(sample_save_dir, "composed_solutions.json")
        if os.path.exists(out_path):
            logger.info(f"[sample {s_idx}] composed solutions already exist, skipping.")
            continue

        logger.info(f"[sample {s_idx}] preparing docs + AnchorMem")
        docs, docs_with_ids = make_docs_from_locomo10_conversations(
            samples=[sample_data], per_session=True, only_referenced=True, overlap=1
        )

        config = BaseConfig(
            save_dir=sample_save_dir,
            seed=args.seed,
            temperature=args.temperature,
            llm_base_url=args.llm_base_url,
            llm_name=args.llm_name,
            dataset=args.dataset,
            embedding_model_name=args.embedding_name,
            embedding_base_url=args.embedding_base_url,
            force_index_from_scratch=False,
            force_fact_extraction_from_scratch=False,
            rerank_dspy_file_path="src/hipporag/prompts/dspy_prompts/filter_llama3.3-70B-Instruct.json",
            retrieval_top_k=10,
            fact_sim_threshold=0.85,
            related_fact_top_k=3,
            linking_top_k=5,
            max_qa_steps=3,
            qa_top_k=5,
            embedding_batch_size=8,
            max_new_tokens=256,
            corpus_len=len(docs),
        )

        anchormem = AnchorMem(global_config=config)
        anchormem.index(docs)

        queries = [c["composed_query"] for c in sample_clusters]
        logger.info(f"[sample {s_idx}] running {len(queries)} composed queries")
        sols, _, _ = anchormem.rag_qa(queries=queries, gold_docs=None, gold_answers=None)

        out_entries = []
        for c, sol in zip(sample_clusters, sols):
            entry = {
                "cluster_id": c["cluster_id"],
                "sample_idx": c["sample_idx"],
                "member_q_idxs": c["member_q_idxs"],
                "gold_dia_ids": c["gold_dia_ids"],
                "composed_query": c["composed_query"],
                "expected_memory_use": c.get("expected_memory_use", ""),
                "answer": getattr(sol, "answer", None),
                "docs": getattr(sol, "docs", []),
                "topk_facts": getattr(sol, "topk_facts", None),
            }
            out_entries.append(entry)

        with open(out_path, "w") as f:
            json.dump(out_entries, f, ensure_ascii=False, indent=2)
        logger.info(f"[sample {s_idx}] wrote {out_path} ({len(out_entries)} clusters)")

        anchormem.save_cost_summary()


if __name__ == "__main__":
    main()
