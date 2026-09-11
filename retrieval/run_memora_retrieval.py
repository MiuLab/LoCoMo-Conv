"""Run memora retrieval over LoCoMo10 queries and compute session-level dia_id recall.

Uses the pre-built memora chroma index at ./indices/memora_locomo10_local_index.
Each memora memory carries a `timestamp` metadata field (session date_time). Within
a sample_id, session timestamps are unique, so we can map retrieved memories to
session-level dia_ids via a LoCoMo timestamp lookup.

Output format mimics the other systems' queries_solutions_{field}.json files:
  outputs_memora/locomo-gemma-4-31B-it/sample_{X}/queries_solutions_{field}.json
"""
from __future__ import annotations
import argparse, json, sys, time, os
from collections import defaultdict
from pathlib import Path


def build_timestamp_maps(locomo):
    """Return (sample_idx → sample_id, sample_idx → {timestamp: [dia_ids]}, sample_idx → conv)."""
    idx_to_id = {}
    ts_to_dia = defaultdict(lambda: defaultdict(list))
    idx_to_conv = {}
    for i, sample in enumerate(locomo):
        sid = sample.get("sample_id") or f"sample_{i}"
        idx_to_id[i] = sid
        idx_to_conv[i] = sample.get("conversation", {})
        conv = sample.get("conversation", {})
        # Build timestamp → dia_ids
        for k in list(conv.keys()):
            if not (k.startswith("session_") and not k.endswith("_date_time")):
                continue
            turns = conv.get(k)
            if not isinstance(turns, list):
                continue
            ts_key = f"{k}_date_time"
            ts = conv.get(ts_key)
            if not ts:
                continue
            for turn in turns:
                did = turn.get("dia_id")
                if did:
                    ts_to_dia[i][ts].append(str(did))
    return idx_to_id, ts_to_dia, idx_to_conv


FIELD_FOR_STYLE = {
    "dialog": "dialog_query",
    "implicit": "implicit_query",
    "counterfactual": "counterfactual_query",
    "composed": "composed",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--memora_src", default="third_party/Memora/src")
    ap.add_argument("--persist_path", default="./indices/memora_locomo10_local_index")
    ap.add_argument("--dataset_path", default="./data/locomo10.json")
    ap.add_argument("--dialog_data_path", default="./data/locomo10_dialog.json")
    ap.add_argument("--multimem_path", default="data/locomo10_multimem_full.json")
    ap.add_argument("--output_dir", default="./outputs_memora/locomo-gemma-4-31B-it")
    ap.add_argument("--top_k", type=int, default=10)
    ap.add_argument("--styles", default="question,dialog,implicit,counterfactual,composed")
    ap.add_argument("--samples", default="all")
    ap.add_argument("--endpoint", default="http://localhost:8000/v1/chat/completions")
    ap.add_argument("--embed_model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--chat_model", default="../gemma-4-31B-it")
    ap.add_argument("--dia_source", choices=["timestamp", "original_text"], default="timestamp",
                    help="How to map retrieved memories to dia_ids: session timestamp lookup (v1 index) "
                         "or the turn-level SOURCE citations stored in original_text (v3 index).")
    ap.add_argument("--query_mode", choices=["BOTH","PRIMARY_ONLY","CUE_ONLY","ORIGINAL"], default=None,
                    help="Override Memora's query mode. Default (None) follows the config: BOTH when "
                         "the cue index is enabled, which is what the main results used.")
    ap.add_argument("--rewrites", default=None,
                    help="Path to a {original_query: rewritten_query} JSON (query rewriting). "
                         "When set, client.query() is fed the rewritten query.")
    ap.add_argument("--rewrites_divergent", default=None,
                    help="Path to a {original_query: [facet, ...]} JSON (multi-facet divergent "
                         "rewriting). Each facet is queried separately (per_facet_k) and results "
                         "are RRF-fused (k=60), mirroring divergent_retrieve.py for the other systems.")
    ap.add_argument("--per_facet_k", type=int, default=20,
                    help="top_k per facet before RRF fusion (divergent mode only).")
    ap.add_argument("--score_threshold", type=float, default=None,
                    help="Override memora's query_score_threshold (default keeps its 0.4).")
    args = ap.parse_args()

    sys.path.insert(0, args.memora_src)

    # Reuse v1's patches for llm + embedding (so no external OPENAI_API_KEY needed).
    import importlib.util
    v1_path = Path(__file__).parent.parent.parent.parent / "scripts" / "benchmark_memora_locomo10_index.py"
    spec = importlib.util.spec_from_file_location("bench_memora_v1", v1_path)
    v1 = importlib.util.module_from_spec(spec)
    sys.modules["bench_memora_v1"] = v1
    spec.loader.exec_module(v1)

    # Stub args for patcher
    class PatchArgs:
        endpoint = args.endpoint
        chat_model = args.chat_model
        embedding_model = args.embed_model
        persist_path = args.persist_path
        memora_src = args.memora_src
        request_timeout = 300.0
        max_tokens = 2048
        temperature = 0.0
        no_cue_index = False
        enable_episodic = False
    stats = v1.LLMStats()
    v1.patch_memora(PatchArgs(), stats)

    from memora.core.local_client import LocalMemoraClient as MemoraClient
    from memora.core.memory import QueryMode
    QMODE = {"BOTH": QueryMode.BOTH, "PRIMARY_ONLY": QueryMode.PRIMARY_ONLY,
             "CUE_ONLY": QueryMode.CUE_ONLY, "ORIGINAL": QueryMode.ORIGINAL}.get(args.query_mode)
    cfg = v1.make_config(PatchArgs())
    if args.score_threshold is not None:
        cfg.memory.query_score_threshold = args.score_threshold

    # Optional query rewriting: map original query text -> rewritten search query.
    # Uses the same convergent rewrites the paper's +Query Rewriting rows use, so
    # Memora's +QR is comparable to the other systems. Retrieval still runs against
    # the same index; only the query string fed to client.query() changes.
    rewrites = {}
    if args.rewrites:
        rewrites = json.load(open(args.rewrites))
        print(f"[rewrites] loaded {len(rewrites)} query rewrites from {args.rewrites}", flush=True)
    def _rw(q):
        return rewrites.get(q, q) if rewrites else q

    # Divergent multi-facet rewriting: query each facet, RRF-fuse the ranked lists.
    # Same fusion as divergent_retrieve.py (k_param=60, per_facet_k=20).
    divergent = {}
    if args.rewrites_divergent:
        divergent = json.load(open(args.rewrites_divergent))
        print(f"[divergent] loaded {len(divergent)} facet sets from {args.rewrites_divergent}", flush=True)

    def _query(client, q, top_k):
        """Single query, or divergent multi-facet + RRF fusion when enabled."""
        if not divergent:
            return client.query(context=_rw(q), top_k=top_k, query_mode=QMODE)
        facets = divergent.get(q, [q])
        scores, seen = {}, {}
        for f in facets:
            try:
                ranked = client.query(context=f, top_k=args.per_facet_k, query_mode=QMODE)
            except Exception as e:
                print(f"  facet query err: {e}", flush=True)
                continue
            for rank, r in enumerate(ranked):
                key = (r.value, getattr(r, 'timestamp', ''), getattr(r, 'original_text', ''))
                scores[key] = scores.get(key, 0.0) + 1.0 / (60 + rank + 1)
                seen.setdefault(key, r)
        fused = sorted(scores, key=lambda k: -scores[k])[:top_k]
        return [seen[k] for k in fused]

    # Load data
    locomo = json.load(open(args.dataset_path))
    dialog_data = json.load(open(args.dialog_data_path))
    multimem = json.load(open(args.multimem_path)) if os.path.exists(args.multimem_path) else []
    idx_to_id, ts_to_dia, idx_to_conv = build_timestamp_maps(locomo)

    styles = [s.strip() for s in args.styles.split(",") if s.strip()]
    if args.samples == "all":
        sample_indices = list(range(len(locomo)))
    else:
        sample_indices = [int(x) for x in args.samples.split(",") if x.strip()]

    os.makedirs(args.output_dir, exist_ok=True)

    def _extract_dia_ids_from_memory(sample_idx, mem_meta_or_entry):
        """Look up dia_ids for a retrieved memory via its timestamp."""
        ts = None
        if hasattr(mem_meta_or_entry, 'timestamp'):
            ts = mem_meta_or_entry.timestamp
        elif isinstance(mem_meta_or_entry, dict):
            ts = mem_meta_or_entry.get('timestamp')
        if not ts:
            return []
        return ts_to_dia[sample_idx].get(ts, [])

    for s_idx in sample_indices:
        sample_id = idx_to_id[s_idx]
        user_id = f"locomo10_{sample_id}"
        client = MemoraClient(cfg=cfg, user_id=user_id)
        n_mem = client.count()
        print(f"[sample {s_idx}] user_id={user_id}  memories in index: {n_mem}", flush=True)
        if n_mem == 0:
            print(f"[sample {s_idx}] SKIP (no memories indexed)", flush=True)
            continue

        sample_dir = os.path.join(args.output_dir, f"sample_{s_idx}")
        os.makedirs(sample_dir, exist_ok=True)

        # Get all queries for this sample
        dialog_sample = dialog_data[s_idx]
        qa_items = dialog_sample.get("qa", [])
        conv = idx_to_conv[s_idx]
        speaker_a = conv.get("speaker_a", "SpeakerA")
        speaker_b = conv.get("speaker_b", "SpeakerB")

        for style in styles:
            if style == "composed":
                out_path = os.path.join(sample_dir, "composed_solutions.json")
                if os.path.exists(out_path):
                    print(f"[sample {s_idx}] composed: skip (exists)", flush=True); continue
                clusters = [c for c in multimem if c.get("sample_idx") == s_idx and c.get("composed_query")]
                entries = []
                t0 = time.time()
                for c in clusters:
                    q = c["composed_query"]
                    try:
                        results = _query(client, q, args.top_k)
                    except Exception as e:
                        print(f"  composed cluster {c.get('cluster_id')} query err: {e}", flush=True)
                        results = []
                    docs = [r.value for r in results]
                    retrieved_dia_ids = set()
                    ts_list = []
                    for r in results:
                        ts = getattr(r, 'timestamp', '')
                        ts_list.append(ts)
                        if args.dia_source == "original_text":
                            src = getattr(r, 'original_text', '') or ''
                            retrieved_dia_ids.update(d for d in src.split(';') if d)
                        else:
                            retrieved_dia_ids.update(ts_to_dia[s_idx].get(ts, []))
                    entries.append({
                        "cluster_id": c["cluster_id"],
                        "sample_idx": s_idx,
                        "member_q_idxs": c.get("member_q_idxs", []),
                        "gold_dia_ids": c.get("gold_dia_ids", []),
                        "composed_query": q,
                        "docs": docs,
                        "retrieved_dia_ids": sorted(retrieved_dia_ids),
                        "retrieved_timestamps": ts_list,
                    })
                json.dump(entries, open(out_path, "w"), ensure_ascii=False, indent=2)
                print(f"[sample {s_idx}] composed: wrote {len(entries)} in {time.time()-t0:.1f}s", flush=True)
                continue

            # Non-composed styles: read from dialog_data
            field_name = FIELD_FOR_STYLE.get(style)
            if field_name is None:
                continue
            out_path = os.path.join(sample_dir, f"queries_solutions_{field_name}.json")
            if os.path.exists(out_path):
                print(f"[sample {s_idx}] {style}: skip (exists)", flush=True); continue

            entries = []
            t0 = time.time()
            for q_idx, qa in enumerate(qa_items):
                q_text = qa.get(field_name) if style != "question" else qa.get("question")
                if not q_text:
                    continue
                try:
                    results = _query(client, q_text, args.top_k)
                except Exception as e:
                    print(f"  q_idx={q_idx} query err: {e}", flush=True)
                    results = []
                docs = [r.value for r in results]
                retrieved_dia_ids = set()
                ts_list = []
                for r in results:
                    ts = getattr(r, 'timestamp', '')
                    ts_list.append(ts)
                    if args.dia_source == "original_text":
                        src = getattr(r, 'original_text', '') or ''
                        retrieved_dia_ids.update(d for d in src.split(';') if d)
                    else:
                        retrieved_dia_ids.update(ts_to_dia[s_idx].get(ts, []))
                entries.append({
                    "q_idx": q_idx,
                    "question": qa.get("question", ""),
                    "query": q_text,
                    "category": qa.get("category"),
                    "evidence": qa.get("evidence") or [],
                    "answer": qa.get("answer"),
                    "docs": docs,
                    "retrieved_dia_ids": sorted(retrieved_dia_ids),
                    "retrieved_timestamps": ts_list,
                })
            json.dump(entries, open(out_path, "w"), ensure_ascii=False, indent=2)
            print(f"[sample {s_idx}] {style}: wrote {len(entries)} in {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
