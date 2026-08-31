"""For each system, run retrieve on EACH facet of the divergent rewrite,
then fuse with Reciprocal Rank Fusion to produce a single top-K per query.

Reads:
  outputs_response_eval/divergent_rewrite/rewrites_divergent.json
  data/response_eval_sample.json
Writes:
  outputs_response_eval/divergent_rewrite/rewrite_retrievals.json
  { "<system>|<style>|<sample>|<q_id>": [doc1, doc2, ...] }
"""
from __future__ import annotations

import sys, os, json
sys.path.append(".")
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "A-mem"))
import multiprocessing as _mp
try:
    _mp.set_start_method("fork", force=True)
except RuntimeError:
    pass

import argparse
from collections import defaultdict
from typing import Dict, List, Tuple


def rrf_fuse(ranked_lists: List[List[str]], k_param: int = 60, top_k: int = 20) -> List[str]:
    """Reciprocal Rank Fusion. ranked_lists: list of ranked docs (one per facet)."""
    scores: Dict[str, float] = defaultdict(float)
    for ranked in ranked_lists:
        for rank, doc in enumerate(ranked):
            scores[doc] += 1.0 / (k_param + rank + 1)
    return [d for d, _ in sorted(scores.items(), key=lambda x: -x[1])][: top_k]


def _load_jobs(sample_path, divergent_path):
    sampled = json.load(open(sample_path))
    divergent = json.load(open(divergent_path))
    jobs = []  # (style, sample_idx, q_id_str, facets, original)
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
            facets = divergent.get(orig, [orig])
            jobs.append((style, it.get("sample_idx"), q_id, facets, orig))
    return jobs


def bm25_run(jobs, top_k=20, per_facet_k=20):
    from rank_bm25 import BM25Okapi
    sys.path.insert(0, "scripts")
    from run_bm25 import build_units, tokenize  # type: ignore

    data = json.load(open("data/locomo10_dialog.json"))
    by_sample = defaultdict(list)
    for j in jobs:
        by_sample[j[1]].append(j)
    out: Dict[str, List[str]] = {}
    for s_idx, sample_jobs in by_sample.items():
        sample = data[s_idx]
        chunks = build_units(sample, "turn")
        if not chunks:
            continue
        corpus_tokens = [tokenize(c[0]) for c in chunks]
        bm25 = BM25Okapi(corpus_tokens)
        texts = [c[0] for c in chunks]
        for style, _, q_id, facets, _orig in sample_jobs:
            ranked_lists = []
            for f in facets:
                scores = bm25.get_scores(tokenize(f))
                top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[: per_facet_k]
                ranked_lists.append([texts[i] for i in top_idx])
            fused = rrf_fuse(ranked_lists, top_k=top_k)
            out[f"BM25|{style}|{s_idx}|{q_id}"] = fused
        print(f"[BM25 sample {s_idx}] {len(sample_jobs)} jobs", flush=True)
    return out


def amem_run(jobs, top_k=20, per_facet_k=20):
    sys.path.insert(0, "scripts")
    from run_amem import turns_with_dia_ids, _disable_evolution  # type: ignore
    from agentic_memory.memory_system import AgenticMemorySystem
    from agentic_memory.retrievers import ChromaRetriever

    print("Init AgenticMemorySystem (no-evo)...", flush=True)
    memsys = AgenticMemorySystem(
        model_name="all-MiniLM-L6-v2",
        llm_backend="openai",
        llm_model="./gemma-4-31B-it",
        evo_threshold=10**6,
        api_key="EMPTY",
    )
    _disable_evolution(memsys)

    data = json.load(open("data/locomo10_dialog.json"))
    by_sample = defaultdict(list)
    for j in jobs:
        by_sample[j[1]].append(j)
    out: Dict[str, List[str]] = {}
    for s_idx, sample_jobs in by_sample.items():
        memsys.memories = {}
        try:
            memsys.retriever.client.reset()
        except Exception:
            pass
        memsys.retriever = ChromaRetriever(collection_name="memories", model_name=memsys.model_name)
        sample = data[s_idx]
        chunks = turns_with_dia_ids(sample)
        print(f"[A-mem sample {s_idx}] ingesting {len(chunks)} turns", flush=True)
        for chunk_text, _dia in chunks:
            try:
                memsys.add_note(content=chunk_text)
            except Exception:
                pass
        for style, _, q_id, facets, _orig in sample_jobs:
            ranked_lists = []
            for f in facets:
                try:
                    results = memsys.search_agentic(f, k=per_facet_k)
                    docs = [r.get("content", "") for r in results if r]
                except Exception:
                    docs = []
                ranked_lists.append(docs)
            fused = rrf_fuse(ranked_lists, top_k=top_k)
            out[f"A-mem|{style}|{s_idx}|{q_id}"] = fused
        print(f"[A-mem sample {s_idx}] {len(sample_jobs)} jobs done", flush=True)
    return out


def anchormem_run(jobs, top_k=20, per_facet_k=20):
    from AnchorMem import AnchorMem  # type: ignore
    from src.datasets.locomo10_loader import make_docs_from_locomo10_conversations  # type: ignore
    from src.utils.config_utils import BaseConfig  # type: ignore

    locomo = json.load(open("data/locomo10.json"))
    by_sample = defaultdict(list)
    for j in jobs:
        by_sample[j[1]].append(j)
    out: Dict[str, List[str]] = {}
    for s_idx, sample_jobs in by_sample.items():
        sample_save_dir = f"outputs/locomo-gemma-4-31B-it/sample_{s_idx}"
        if not os.path.isdir(sample_save_dir):
            continue
        docs, _ = make_docs_from_locomo10_conversations(
            samples=[locomo[s_idx]], per_session=True, only_referenced=True, overlap=1,
        )
        config = BaseConfig(
            save_dir=sample_save_dir,
            llm_base_url="http://localhost:8000/v1",
            llm_name="./gemma-4-31B-it",
            dataset="locomo",
            embedding_model_name="Transformers/all-MiniLM-L6-v2",
            embedding_base_url=None,
            force_index_from_scratch=False,
            force_fact_extraction_from_scratch=False,
            rerank_dspy_file_path="src/hipporag/prompts/dspy_prompts/filter_llama3.3-70B-Instruct.json",
            retrieval_top_k=per_facet_k,
            fact_sim_threshold=0.85,
            related_fact_top_k=3,
            linking_top_k=5,
            max_qa_steps=3,
            qa_top_k=per_facet_k,
            embedding_batch_size=8,
            max_new_tokens=256,
            corpus_len=len(docs),
        )
        print(f"[AnchorMem sample {s_idx}] loading...", flush=True)
        anchormem = AnchorMem(global_config=config)
        anchormem.index(docs)
        # Gather all facets across sample jobs into one retrieve call (efficient)
        all_facets = []
        facet_owner = []  # for each facet position, the (job_idx_in_sample_jobs, facet_idx)
        for ji, (_style, _s, _q, facets, _orig) in enumerate(sample_jobs):
            for fi, f in enumerate(facets):
                all_facets.append(f)
                facet_owner.append((ji, fi))
        results = anchormem.retrieve(all_facets)
        # Group results back by job
        job_facet_docs: Dict[int, List[List[str]]] = defaultdict(list)
        for sol, (ji, _fi) in zip(results, facet_owner):
            job_facet_docs[ji].append(sol.docs or [])
        for ji, (style, _s, q_id, _facets, _orig) in enumerate(sample_jobs):
            ranked_lists = job_facet_docs.get(ji, [])
            fused = rrf_fuse(ranked_lists, top_k=top_k)
            out[f"AnchorMem|{style}|{s_idx}|{q_id}"] = fused
        print(f"[AnchorMem sample {s_idx}] {len(sample_jobs)} jobs done", flush=True)
    return out


def dense_run(jobs, top_k=20, per_facet_k=20, model_name="sentence-transformers/all-MiniLM-L6-v2"):
    """Dense MiniLM over raw turns + RRF per query."""
    import numpy as np
    from sentence_transformers import SentenceTransformer
    sys.path.insert(0, "scripts")
    from run_dense import build_units  # type: ignore

    data = json.load(open("data/locomo10_dialog.json"))
    model = SentenceTransformer(model_name)

    by_sample = defaultdict(list)
    for j in jobs:
        by_sample[j[1]].append(j)
    out: Dict[str, List[str]] = {}
    for s_idx, sample_jobs in by_sample.items():
        sample = data[s_idx]
        chunks = build_units(sample)
        if not chunks:
            continue
        texts = [c[0] for c in chunks]
        doc_vecs = model.encode(texts, normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True)
        for style, _, q_id, facets, _orig in sample_jobs:
            f_vecs = model.encode(facets, normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True)
            ranked_lists = []
            for fv in f_vecs:
                sims = doc_vecs @ fv
                top_idx = np.argsort(-sims)[: per_facet_k].tolist()
                ranked_lists.append([texts[i] for i in top_idx])
            fused = rrf_fuse(ranked_lists, top_k=top_k)
            out[f"Dense|{style}|{s_idx}|{q_id}"] = fused
        print(f"[Dense sample {s_idx}] {len(sample_jobs)} jobs", flush=True)
    return out


def mem0_run(jobs, top_k=20, per_facet_k=20,
             chroma_path="outputs_mem0/locomo-gemma-4-31B-it/_chroma_state",
             llm_base_url=None, llm_model="./gemma-4-31B-it",
             embed_model="sentence-transformers/all-MiniLM-L6-v2"):
    """Connect to EXISTING Mem0 chroma and RRF across rewritten facets. No re-ingest."""
    sys.path.insert(0, "scripts")
    from run_mem0 import _build_mem0_config  # type: ignore
    from mem0 import Memory  # type: ignore

    data = json.load(open("data/locomo10_dialog.json"))
    if llm_base_url is None:
        llm_base_url = os.environ.get("OPENAI_API_BASE", "http://localhost:8000/v1")
    print(f"Connecting to existing Mem0 chroma at {chroma_path}...", flush=True)
    cfg = _build_mem0_config(llm_base_url, llm_model, chroma_path, embed_model)
    memory = Memory.from_config(cfg)

    by_sample = defaultdict(list)
    for j in jobs:
        by_sample[j[1]].append(j)
    out: Dict[str, List[str]] = {}
    for s_idx, sample_jobs in by_sample.items():
        sample = data[s_idx]
        speaker_a = sample.get("conversation", {}).get("speaker_a", f"speaker_a_{s_idx}")
        speaker_b = sample.get("conversation", {}).get("speaker_b", f"speaker_b_{s_idx}")
        user_id_a = f"{speaker_a}_{s_idx}"
        user_id_b = f"{speaker_b}_{s_idx}"
        for style, _, q_id, facets, _orig in sample_jobs:
            qa_list = sample.get("qa", [])
            try:
                qi = int(q_id)
                qa = qa_list[qi] if qi < len(qa_list) else {}
            except ValueError:
                qa = {}
            subj_key = (qa.get("counterfactual_subject_speaker") if style == "counterfactual"
                        else qa.get("implicit_subject_speaker") if style == "implicit"
                        else qa.get("subject_speaker")) or "speaker_a"
            uid = user_id_a if subj_key == "speaker_a" else user_id_b
            ranked_lists = []
            doc_meta: Dict[str, Dict] = {}
            for f in facets:
                try:
                    res = memory.search(query=f, filters={"user_id": uid}, limit=per_facet_k)
                    items = (res.get("results") or []) if isinstance(res, dict) else (res or [])
                except Exception:
                    items = []
                docs = []
                for r in items:
                    if not r: continue
                    m = r.get("memory", "")
                    docs.append(m)
                    doc_meta.setdefault(m, r.get("metadata", {}) or {})
                ranked_lists.append(docs)
            fused = rrf_fuse(ranked_lists, top_k=top_k)
            key = f"mem0|{style}|{s_idx}|{q_id}"
            out[key] = fused
            _MEM0_META_SIDECAR_DIV[key] = [doc_meta.get(d, {}) for d in fused]
        print(f"[Mem0 sample {s_idx}] {len(sample_jobs)} jobs done", flush=True)
    return out


_MEM0_META_SIDECAR_DIV: Dict[str, List[Dict]] = {}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sample_path", default="data/response_eval_sample.json")
    p.add_argument("--output_dir", default="outputs_response_eval/divergent_rewrite")
    p.add_argument("--systems", default="bm25,amem,anchormem")
    p.add_argument("--top_k", type=int, default=20)
    p.add_argument("--per_facet_k", type=int, default=20)
    args = p.parse_args()

    divergent_path = os.path.join(args.output_dir, "rewrites_divergent.json")
    if not os.path.exists(divergent_path):
        print(f"Missing {divergent_path}. Run rewrite_queries_divergent.py first.", file=sys.stderr)
        sys.exit(1)

    jobs = _load_jobs(args.sample_path, divergent_path)
    print(f"Total jobs: {len(jobs)}  (avg facets/q: {sum(len(j[3]) for j in jobs)/len(jobs):.2f})", flush=True)

    rr_path = os.path.join(args.output_dir, "rewrite_retrievals.json")
    existing: Dict[str, List[str]] = json.load(open(rr_path)) if os.path.exists(rr_path) else {}

    systems = [s.strip().lower() for s in args.systems.split(",")]
    for system in systems:
        if system == "bm25":
            res = bm25_run(jobs, top_k=args.top_k, per_facet_k=args.per_facet_k)
        elif system == "amem":
            res = amem_run(jobs, top_k=args.top_k, per_facet_k=args.per_facet_k)
        elif system == "anchormem":
            res = anchormem_run(jobs, top_k=args.top_k, per_facet_k=args.per_facet_k)
        elif system == "dense":
            res = dense_run(jobs, top_k=args.top_k, per_facet_k=args.per_facet_k)
        elif system == "mem0":
            res = mem0_run(jobs, top_k=args.top_k, per_facet_k=args.per_facet_k)
        else:
            print(f"Unknown system: {system}", file=sys.stderr)
            continue
        existing.update(res)
        json.dump(existing, open(rr_path, "w"), ensure_ascii=False, indent=2)
        print(f"Saved {rr_path}  (total: {len(existing)} entries after {system})", flush=True)
        if system == "mem0" and _MEM0_META_SIDECAR_DIV:
            meta_path = os.path.join(args.output_dir, "rewrite_retrievals_meta.json")
            existing_meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
            existing_meta.update(_MEM0_META_SIDECAR_DIV)
            json.dump(existing_meta, open(meta_path, "w"), ensure_ascii=False, indent=2)
            print(f"Saved {meta_path}  (total {len(existing_meta)} meta entries)", flush=True)


if __name__ == "__main__":
    main()
