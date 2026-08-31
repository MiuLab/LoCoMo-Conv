"""Re-run per-system retrieval using gpt-5.4-mini rewritten queries.

Reads:
  - data/response_eval_sample.json (the 1200 sampled queries)
  - outputs_response_eval/<run>/rewrites.json (orig_query -> rewritten_query)

For each (system, sample_idx, q_idx_or_cluster) in the sample, runs the system's
retrieve() with the rewritten query and saves top-K docs.

Output:
  outputs_response_eval/<run>/rewrite_retrievals.json
  { "<system>|<style>|<sample_idx>|<q_idx_or_cluster_id>": [doc1, doc2, ...], ... }

The driver consumes this when --variants reasoning_rewrite is run.

Supports --system bm25 (fast, no LLM, no embeddings).
A-mem and AnchorMem support require their environment to be loaded; if you
specify those, the script will attempt to import their packages.
"""
from __future__ import annotations

import argparse, json, os, sys
from collections import defaultdict
from typing import Dict, List

# A-mem lives in a sibling directory
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "A-mem"))
sys.path.append(".")


def _load_sampled_queries(sample_path: str, rewrites_path: str):
    """Return list of (system_label_for_output, style, sample_idx, q_id_str, rewritten_query, original_query)."""
    sampled = json.load(open(sample_path))
    rewrites = json.load(open(rewrites_path))
    out = []
    for style, items in sampled["samples"].items():
        for it in items:
            if style == "composed":
                orig = it.get("composed_query")
                q_id_str = str(it.get("cluster_id"))
            else:
                orig = it.get("query")
                q_id_str = str(it.get("q_idx"))
            if not orig:
                continue
            rw = rewrites.get(orig, orig)
            out.append((style, it.get("sample_idx"), q_id_str, rw, orig))
    return out


def _bm25_retrieve(jobs, top_k=20):
    """For BM25: per-sample build index from turns of the sample, then run BM25Okapi on rewritten queries."""
    from rank_bm25 import BM25Okapi
    sys.path.insert(0, "scripts")
    from run_bm25 import build_units, tokenize  # type: ignore

    data = json.load(open("data/locomo10_dialog.json"))

    # Group jobs by sample_idx
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
        chunk_texts = [c[0] for c in chunks]

        for style, _, q_id_str, rw, _orig in sample_jobs:
            scores = bm25.get_scores(tokenize(rw))
            top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
            key = f"BM25|{style}|{s_idx}|{q_id_str}"
            out[key] = [chunk_texts[i] for i in top_idx]
        print(f"[BM25 sample {s_idx}] {len(sample_jobs)} queries", file=sys.stderr, flush=True)
    return out


def _dense_retrieve(jobs, top_k=20, model_name="sentence-transformers/all-MiniLM-L6-v2"):
    """Pure dense retrieval (MiniLM-L6-v2) over raw turns. No LLM needed."""
    import numpy as np
    from sentence_transformers import SentenceTransformer
    sys.path.insert(0, "scripts")
    from run_dense import build_units  # type: ignore

    data = json.load(open("data/locomo10_dialog.json"))
    print(f"Loading {model_name}...", file=sys.stderr, flush=True)
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
        qs = [j[3] for j in sample_jobs]  # rewritten queries
        q_vecs = model.encode(qs, normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True)
        for (style, _, q_id_str, rw, _orig), q_vec in zip(sample_jobs, q_vecs):
            sims = doc_vecs @ q_vec
            top_idx = np.argsort(-sims)[:top_k].tolist()
            key = f"Dense|{style}|{s_idx}|{q_id_str}"
            out[key] = [texts[i] for i in top_idx]
        print(f"[Dense sample {s_idx}] {len(sample_jobs)} queries", file=sys.stderr, flush=True)
    return out


def _mem0_retrieve(jobs, top_k=20,
                   chroma_path="outputs_mem0/locomo-gemma-4-31B-it/_chroma_state",
                   llm_base_url=None, llm_model="./gemma-4-31B-it",
                   embed_model="sentence-transformers/all-MiniLM-L6-v2"):
    """For Mem0: connect to the EXISTING chroma store and search with rewritten queries.
    Does NOT re-ingest. Uses two user_ids per sample (mirroring scripts/run_mem0.py)."""
    sys.path.insert(0, "scripts")
    from run_mem0 import _build_mem0_config  # type: ignore
    from mem0 import Memory  # type: ignore

    data = json.load(open("data/locomo10_dialog.json"))
    if llm_base_url is None:
        llm_base_url = os.environ.get("OPENAI_API_BASE", "http://localhost:8000/v1")
    print(f"Connecting to existing Mem0 chroma at {chroma_path}...", file=sys.stderr, flush=True)
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

        # Search with rewritten queries
        for style, _, q_id_str, rw, _orig in sample_jobs:
            # Determine subject speaker from sample
            qa_list = sample.get("qa", [])
            try:
                q_idx_int = int(q_id_str)
                qa = qa_list[q_idx_int] if q_idx_int < len(qa_list) else {}
            except ValueError:
                qa = {}
            subj_key = (qa.get("counterfactual_subject_speaker") if style == "counterfactual"
                        else qa.get("implicit_subject_speaker") if style == "implicit"
                        else qa.get("subject_speaker")) or "speaker_a"
            uid = user_id_a if subj_key == "speaker_a" else user_id_b
            try:
                res = memory.search(query=rw, filters={"user_id": uid}, limit=top_k)
                items = (res.get("results") or []) if isinstance(res, dict) else (res or [])
                docs = [r.get("memory", "") for r in items if r]
                metas = [r.get("metadata", {}) for r in items if r]
            except Exception:
                docs = []
                metas = []
            key = f"mem0|{style}|{s_idx}|{q_id_str}"
            out[key] = docs
            _MEM0_META_SIDECAR[key] = metas
        print(f"[Mem0 sample {s_idx}] {len(sample_jobs)} queries done", file=sys.stderr, flush=True)
    return out


_MEM0_META_SIDECAR: Dict[str, List[Dict]] = {}


def _amem_retrieve(jobs, top_k=20, llm_base_url=None, llm_model="./gemma-4-31B-it"):
    """For A-mem: per-sample ingest turn-level notes, then search_agentic with rewritten queries.

    This re-ingests memories — slow (~3 min per sample). Reuses the same
    granularity (1 note per turn) as scripts/run_amem.py.
    """
    sys.path.insert(0, "scripts")
    from run_amem import turns_with_dia_ids, _disable_evolution  # type: ignore
    from agentic_memory.memory_system import AgenticMemorySystem
    from agentic_memory.retrievers import ChromaRetriever

    data = json.load(open("data/locomo10_dialog.json"))

    # Initialize once. Disable evolution so add_note doesn't call the LLM
    # on every turn (otherwise ingestion takes hours).
    print("Initializing AgenticMemorySystem (no-evo)...", file=sys.stderr, flush=True)
    memsys = AgenticMemorySystem(
        model_name="all-MiniLM-L6-v2",
        llm_backend="openai",
        llm_model=llm_model,
        evo_threshold=10**6,  # high threshold (extra safety)
        api_key="EMPTY",
    )
    _disable_evolution(memsys)
    print("Evolution disabled.", file=sys.stderr, flush=True)

    by_sample = defaultdict(list)
    for j in jobs:
        by_sample[j[1]].append(j)

    out: Dict[str, List[str]] = {}
    for s_idx, sample_jobs in by_sample.items():
        # Reset per sample
        memsys.memories = {}
        try:
            memsys.retriever.client.reset()
        except Exception:
            pass
        memsys.retriever = ChromaRetriever(collection_name="memories", model_name=memsys.model_name)

        sample = data[s_idx]
        chunks = turns_with_dia_ids(sample)
        print(f"[A-mem sample {s_idx}] ingesting {len(chunks)} turns", file=sys.stderr, flush=True)
        for chunk_text, _dia in chunks:
            try:
                memsys.add_note(content=chunk_text)
            except Exception:
                pass

        for style, _, q_id_str, rw, _orig in sample_jobs:
            try:
                results = memsys.search_agentic(rw, k=top_k)
                docs = [r.get("content", "") for r in results if r]
            except Exception:
                docs = []
            key = f"A-mem|{style}|{s_idx}|{q_id_str}"
            out[key] = docs
        print(f"[A-mem sample {s_idx}] {len(sample_jobs)} queries done", file=sys.stderr, flush=True)
    return out


def _anchormem_retrieve(jobs, top_k=20):
    """For AnchorMem: reload each sample's existing index from outputs/locomo-gemma-4-31B-it/sample_<i>/
    and call retrieve(rewritten_queries). This avoids re-ingestion since AnchorMem persists its
    embedding store + fact graph to disk.
    """
    # Defer import; setting up AnchorMem here requires the full pipeline.
    raise NotImplementedError(
        "AnchorMem rewrite-retrieval requires re-loading per-sample state; "
        "implement after BM25/A-mem results validate the rewrite hypothesis."
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sample_path", default="data/response_eval_sample.json")
    p.add_argument("--output_dir", default="outputs_response_eval/oracle_all_styles")
    p.add_argument("--systems", default="bm25", help="comma-sep: bm25, amem, anchormem")
    p.add_argument("--top_k", type=int, default=20)
    p.add_argument("--llm_base_url", default=os.environ.get(
        "ANSWER_BASE_URL",
        "http://localhost:8000/v1"))
    args = p.parse_args()

    rewrites_path = os.path.join(args.output_dir, "rewrites.json")
    if not os.path.exists(rewrites_path):
        print(f"Missing {rewrites_path}. Run scripts/rewrite_queries.py first.", file=sys.stderr, flush=True)
        sys.exit(1)

    jobs = _load_sampled_queries(args.sample_path, rewrites_path)
    print(f"Total jobs: {len(jobs)}", file=sys.stderr, flush=True)

    out_path = os.path.join(args.output_dir, "rewrite_retrievals.json")
    existing: Dict[str, List[str]] = {}
    if os.path.exists(out_path):
        existing = json.load(open(out_path))
        print(f"Existing entries: {len(existing)}", file=sys.stderr, flush=True)

    systems = [s.strip().lower() for s in args.systems.split(",") if s.strip()]

    for system in systems:
        if system == "bm25":
            res = _bm25_retrieve(jobs, top_k=args.top_k)
        elif system == "amem":
            res = _amem_retrieve(jobs, top_k=args.top_k, llm_base_url=args.llm_base_url)
        elif system == "anchormem":
            res = _anchormem_retrieve(jobs, top_k=args.top_k)
        elif system == "dense":
            res = _dense_retrieve(jobs, top_k=args.top_k)
        elif system == "mem0":
            res = _mem0_retrieve(jobs, top_k=args.top_k)
        else:
            print(f"Unknown system: {system}", file=sys.stderr, flush=True)
            continue
        existing.update(res)
        json.dump(existing, open(out_path, "w"), ensure_ascii=False, indent=2)
        print(f"Wrote {out_path}  (total {len(existing)} entries after {system})", file=sys.stderr, flush=True)
        if system == "mem0" and _MEM0_META_SIDECAR:
            meta_path = os.path.join(args.output_dir, "rewrite_retrievals_meta.json")
            existing_meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
            existing_meta.update(_MEM0_META_SIDECAR)
            json.dump(existing_meta, open(meta_path, "w"), ensure_ascii=False, indent=2)
            print(f"Wrote {meta_path}  (total {len(existing_meta)} meta entries)", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
