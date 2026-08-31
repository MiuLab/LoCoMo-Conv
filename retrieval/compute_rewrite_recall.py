"""Compute dia_id-based retrieval recall for rewrite_retrievals.json entries.

Compares convergent vs divergent rewrite retrievals against gold evidence
(LoCoMo's evidence field per QA, or gold_dia_ids per composed cluster).

Output: prints macro recall@K per (system, style) for each retrieval cache.
"""
from __future__ import annotations
import argparse, json, os, sys
from collections import defaultdict


def build_dia_id_text_map(sample):
    out = {}
    for k, v in sample.get("conversation", {}).items():
        if not (isinstance(k, str) and k.startswith("session_") and "date" not in k):
            continue
        if not isinstance(v, list):
            continue
        for t in v:
            did = t.get("dia_id")
            txt = (t.get("text") or "").strip().lower()
            if did and txt:
                out[did] = txt
    return out


def docs_to_dia_ids(docs, id_to_text):
    out = set()
    for d in docs or []:
        d_lower = (d or "").lower()
        for did, txt in id_to_text.items():
            if txt and txt in d_lower:
                out.add(did)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rewrite_retrievals", required=True,
                   help="path to rewrite_retrievals.json")
    p.add_argument("--dataset_path", default="data/locomo10_dialog.json")
    p.add_argument("--multimem_path", default="data/locomo10_multimem.json")
    p.add_argument("--top_k", type=int, default=10, help="recall@K")
    p.add_argument("--restrict_to_sample", default=None,
                   help="optional path to response_eval_sample.json; if given, only include those (style, sample_idx, q_id) tuples in the metric")
    args = p.parse_args()

    rr = json.load(open(args.rewrite_retrievals))
    data = json.load(open(args.dataset_path))
    mm = json.load(open(args.multimem_path)) if os.path.exists(args.multimem_path) else []
    meta_path = args.rewrite_retrievals.replace("rewrite_retrievals.json", "rewrite_retrievals_meta.json")
    meta_map = json.load(open(meta_path)) if os.path.exists(meta_path) else {}

    allow_set = None
    if args.restrict_to_sample:
        sampled = json.load(open(args.restrict_to_sample))
        allow_set = set()
        for style, items in sampled["samples"].items():
            for it in items:
                if style == "composed":
                    allow_set.add((style, it["sample_idx"], str(it["cluster_id"])))
                else:
                    allow_set.add((style, it["sample_idx"], str(it["q_idx"])))

    # Build per-sample dia_id text map
    sample_maps = {i: build_dia_id_text_map(s) for i, s in enumerate(data)}

    # Build gold lookups
    def gold_for(style, sample_idx, q_id):
        if style == "composed":
            for c in mm:
                if c.get("sample_idx") == sample_idx and str(c.get("cluster_id")) == str(q_id):
                    return set(c.get("gold_dia_ids", []))
            return set()
        else:
            qa_list = data[sample_idx].get("qa", [])
            try:
                qa = qa_list[int(q_id)]
            except (IndexError, ValueError):
                return set()
            return set(qa.get("evidence", []) or [])

    # Aggregate by (system, style)
    metrics = defaultdict(lambda: {"n": 0, "recall": 0.0, "precision": 0.0, "f1": 0.0})

    for key, docs in rr.items():
        try:
            system, style, s_idx_str, q_id = key.split("|", 3)
            s_idx = int(s_idx_str)
        except ValueError:
            continue
        if allow_set is not None and (style, s_idx, q_id) not in allow_set:
            continue
        gold = gold_for(style, s_idx, q_id)
        if not gold:
            continue
        topk_docs = (docs or [])[: args.top_k]
        retrieved = docs_to_dia_ids(topk_docs, sample_maps.get(s_idx, {}))
        # mem0 stores abstracted memory text; fall back to metadata sidecar dia_ids
        if system == "mem0" and key in meta_map:
            for meta in (meta_map[key] or [])[: args.top_k]:
                if isinstance(meta, dict):
                    dia_str = meta.get("dia_ids", "")
                    if isinstance(dia_str, str) and dia_str:
                        for did in dia_str.split(","):
                            did = did.strip()
                            if did:
                                retrieved.add(did)
        if not retrieved:
            recall = precision = f1 = 0.0
        else:
            tp = len(retrieved & gold)
            recall = tp / len(gold)
            precision = tp / len(retrieved)
            f1 = (2 * recall * precision / (recall + precision)) if (recall + precision) > 0 else 0.0
        m = metrics[(system, style)]
        m["n"] += 1
        m["recall"] += recall
        m["precision"] += precision
        m["f1"] += f1

    # Print
    print(f"\nFile: {args.rewrite_retrievals}  (K={args.top_k})")
    print(f"{'system':<10} | {'style':<14} | {'n':<5} | {'recall':>7} | {'prec':>7} | {'f1':>7}")
    print("-" * 65)
    for (system, style), m in sorted(metrics.items()):
        n = m["n"]
        if n == 0:
            continue
        print(f"{system:<10} | {style:<14} | {n:<5} | {m['recall']/n:>7.3f} | {m['precision']/n:>7.3f} | {m['f1']/n:>7.3f}")


if __name__ == "__main__":
    main()
