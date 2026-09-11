"""Recompute retrieval recall at a uniform top-K across all systems for fair comparison."""
from __future__ import annotations
import argparse, json, os, sys
from collections import defaultdict
from typing import Any, Dict, List, Set, Tuple, Optional


SYSTEMS = {
    "AnchorMem":     "outputs/locomo-gemma-4-31B-it",
    "A-mem":         "outputs_amem/no_evo/locomo-gemma-4-31B-it",
    "mem0":          "outputs_mem0/locomo-gemma-4-31B-it",
    "BM25":          "outputs_bm25/locomo-bm25",
    "Dense":         "outputs_dense/locomo-dense",
}
FIELD_FOR_STYLE = {"dialog": "dialog_query", "implicit": "implicit_query", "counterfactual": "counterfactual_query"}


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


def retrieved_dia_ids(docs, id_to_text, retrieved_metadata=None):
    out = set()
    for d in docs or []:
        d_l = (d or "").lower()
        for did, txt in id_to_text.items():
            if txt and txt in d_l:
                out.add(did)
    if retrieved_metadata:
        for meta in retrieved_metadata:
            if isinstance(meta, dict):
                dia_str = meta.get("dia_ids", "")
                if isinstance(dia_str, str) and dia_str:
                    for did in dia_str.split(","):
                        did = did.strip()
                        if did:
                            out.add(did)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_path", default="data/locomo10_dialog.json")
    p.add_argument("--multimem_path", default="data/locomo10_multimem_full.json")
    p.add_argument("--top_k", type=int, required=True, help="Truncate retrieved docs to top-K before scoring")
    p.add_argument("--styles", default="dialog,implicit,counterfactual,composed")
    args = p.parse_args()

    data = json.load(open(args.dataset_path))
    mm = json.load(open(args.multimem_path)) if os.path.exists(args.multimem_path) else []
    sample_maps = {i: build_dia_id_text_map(s) for i, s in enumerate(data)}
    styles = [s.strip() for s in args.styles.split(",")]

    print(f"Recall@{args.top_k} across systems:")
    print(f"{'System':<14} | " + " | ".join(f'{s:>8}' for s in styles))
    print("-" * (16 + 11*len(styles)))

    for sys_name, sys_dir in SYSTEMS.items():
        row_vals = []
        for style in styles:
            if style == "composed":
                records = []
                clusters_by_sample = defaultdict(list)
                for c in mm:
                    if c and c.get("composed_query"):
                        clusters_by_sample[c["sample_idx"]].append(c)
                for s_idx, _ in enumerate(data):
                    cl = clusters_by_sample.get(s_idx, [])
                    if not cl:
                        continue
                    cache = os.path.join(sys_dir, f"sample_{s_idx}", "composed_solutions.json")
                    if not os.path.exists(cache):
                        continue
                    sols = json.load(open(cache))
                    id_map = sample_maps[s_idx]
                    for entry in sols:
                        gold = set(entry.get("gold_dia_ids", []))
                        docs = (entry.get("docs") or [])[: args.top_k]
                        retr = retrieved_dia_ids(docs, id_map, retrieved_metadata=(entry.get("retrieved_metadata") or [])[: args.top_k])
                        if gold:
                            r = len(retr & gold) / len(gold)
                            records.append(r)
            else:
                qfield = FIELD_FOR_STYLE.get(style)
                if qfield is None:
                    row_vals.append("-")
                    continue
                records = []
                for s_idx, sample in enumerate(data):
                    cache = os.path.join(sys_dir, f"sample_{s_idx}", f"queries_solutions_{qfield}.json")
                    if not os.path.exists(cache):
                        continue
                    sols = json.load(open(cache))
                    id_map = sample_maps[s_idx]
                    qa_list = sample.get("qa", [])
                    ordered = []
                    for q_idx, qa in enumerate(qa_list):
                        if int(qa.get("category", 0)) == 5 and style == "counterfactual":
                            continue
                        if qa.get(qfield):
                            ordered.append((q_idx, qa))
                    for i, (q_idx, qa) in enumerate(ordered):
                        if i >= len(sols):
                            break
                        sol = sols[i] or {}
                        gold = set(qa.get("evidence", []))
                        if not gold:
                            continue
                        docs = (sol.get("docs") or [])[: args.top_k]
                        retr = retrieved_dia_ids(docs, id_map, retrieved_metadata=(sol.get("retrieved_metadata") or [])[: args.top_k])
                        r = len(retr & gold) / len(gold)
                        records.append(r)
            if records:
                row_vals.append(f"{sum(records)/len(records):>8.3f}")
            else:
                row_vals.append("       -")
        print(f"{sys_name:<14} | " + " | ".join(row_vals))


if __name__ == "__main__":
    main()
