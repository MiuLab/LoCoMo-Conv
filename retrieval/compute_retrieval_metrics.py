"""Compute retrieval recall@K / precision@K / F1@K against locomo dia_ids
across all 4 rewrite styles for AnchorMem.

For each sample, we map retrieved docs/facts back to dia_ids via substring
matching against the conversation's turn texts. (Sample dir already contains
the cached queries_solutions_<style>.json files from main.py runs.)

Output:
  outputs/locomo-<llm>/per_style_retrieval_metrics.json
  {
    "by_style": {
       "dialog": {"n":..., "macro_recall":..., "precision":..., "f1":..., "per_cat":{...}},
       "implicit": {...},
       "counterfactual": {...},
       "composed": {...},
       "question": {...},   # if baseline available
    }
  }

Note: For Styles 1/2/4 the "gold dia_ids" per query is the original locomo
evidence list. For Style 5 (composed) the gold dia_ids is the union of the
cluster members' evidence (in data/locomo10_multimem_full.json).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple


def select_qas_in_main_order(sample_qa: List[Dict[str, Any]], query_field: str) -> List[Tuple[int, Dict[str, Any]]]:
    """Reproduce main.py's per-sample selection order so cache idx ↔ qa idx maps cleanly."""
    selected: List[Tuple[int, Dict[str, Any]]] = []
    for q_idx, qa in enumerate(sample_qa):
        try:
            cat = int(qa.get("category", 0))
        except (TypeError, ValueError):
            cat = 0
        if cat == 5 and query_field == "question":
            continue
        if cat == 5 and query_field == "counterfactual_query":
            continue
        if query_field == "question":
            qstr = qa.get("question") or ""
        elif query_field == "dialog_query":
            qstr = qa.get("dialog_query") or ""
        elif query_field == "implicit_query":
            qstr = qa.get("implicit_query") or ""
        elif query_field == "counterfactual_query":
            qstr = qa.get("counterfactual_query") or ""
        elif query_field == "auto":
            qstr = qa.get("implicit_query") or qa.get("dialog_query") or qa.get("question") or ""
        else:
            qstr = ""
        if not qstr:
            continue
        selected.append((q_idx, qa))
    return selected


def build_dia_id_text_map(sample: Dict[str, Any]) -> Dict[str, str]:
    """Map every dia_id in the sample's conversation to its (lowercased) text."""
    out: Dict[str, str] = {}
    for k, v in sample.get("conversation", {}).items():
        if not (isinstance(k, str) and k.startswith("session_") and "date" not in k):
            continue
        if not isinstance(v, list):
            continue
        for turn in v:
            did = turn.get("dia_id")
            txt = (turn.get("text") or "").strip().lower()
            if did and txt:
                out[did] = txt
    return out


def retrieved_dia_ids(retrieved_docs: List[str], id_to_text: Dict[str, str],
                     retrieved_metadata: Optional[List[Dict[str, Any]]] = None) -> Set[str]:
    """For each retrieved doc, find which dia_ids it covers.

    Tries TWO sources:
      1. Substring match: dia_id's turn text appears in retrieved doc
         (works for raw-turn / chunk-based systems)
      2. Metadata dia_ids field: if metadata for each retrieved item is given,
         parse 'dia_ids' (comma-separated) and union into the set
         (necessary for abstractive systems like mem0 whose doc text doesn't
         contain the original turn verbatim)
    """
    out: Set[str] = set()
    if retrieved_docs:
        for doc in retrieved_docs:
            d = (doc or "").lower()
            if not d:
                continue
            for did, txt in id_to_text.items():
                if txt and txt in d:
                    out.add(did)
    if retrieved_metadata:
        for meta in retrieved_metadata:
            if not isinstance(meta, dict):
                continue
            dia_str = meta.get("dia_ids", "")
            if isinstance(dia_str, str) and dia_str:
                for did in dia_str.split(","):
                    did = did.strip()
                    if did:
                        out.add(did)
    return out


def prf(retrieved: Set[str], gold: Set[str]) -> Tuple[float, float, float]:
    if not gold:
        return 0.0, 0.0, 0.0
    if not retrieved:
        return 0.0, 0.0, 0.0
    tp = len(retrieved & gold)
    recall = tp / len(gold)
    precision = tp / len(retrieved)
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return recall, precision, f1


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_path", default="data/locomo10_dialog.json")
    p.add_argument("--multimem_path", default="data/locomo10_multimem_full.json")
    p.add_argument("--results_dir", default="outputs/locomo-gemma-4-31B-it")
    p.add_argument("--styles", default="question,dialog,implicit,counterfactual,composed")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    out_path = args.out or os.path.join(args.results_dir, "per_style_retrieval_metrics.json")
    styles = [s.strip() for s in args.styles.split(",") if s.strip()]

    field_for = {
        "question": "question",
        "dialog": "dialog_query",
        "implicit": "implicit_query",
        "counterfactual": "counterfactual_query",
    }
    cache_suffix_for = {
        "question": "",
        "dialog": "_dialog_query",
        "implicit": "_implicit_query",
        "counterfactual": "_counterfactual_query",
    }
    # Some adapters (A-mem) write `queries_solutions_question.json` instead of
    # the AnchorMem default `queries_solutions.json`. Try both.
    alt_cache_suffix_for = {"question": "_question"}

    with open(args.dataset_path) as f:
        data = json.load(f)
    mm = []
    if "composed" in styles and os.path.exists(args.multimem_path):
        with open(args.multimem_path) as f:
            mm = json.load(f)

    by_style: Dict[str, Dict[str, Any]] = {}

    # Styles 1/2/4/question (from QA-list)
    for style in styles:
        if style == "composed":
            continue
        qfield = field_for.get(style)
        if qfield is None:
            continue
        cache_path_suffix = cache_suffix_for[style]
        records: List[Dict[str, Any]] = []
        for s_idx, sample in enumerate(data):
            sample_dir = os.path.join(args.results_dir, f"sample_{s_idx}")
            cache_path = os.path.join(sample_dir, f"queries_solutions{cache_path_suffix}.json")
            if not os.path.exists(cache_path):
                alt = alt_cache_suffix_for.get(style)
                if alt:
                    alt_path = os.path.join(sample_dir, f"queries_solutions{alt}.json")
                    if os.path.exists(alt_path):
                        cache_path = alt_path
                    else:
                        print(f"[{style}] missing cache {cache_path} (also tried {alt_path})", file=sys.stderr)
                        continue
                else:
                    print(f"[{style}] missing cache {cache_path}", file=sys.stderr)
                    continue
            with open(cache_path) as f:
                sols = json.load(f)
            ordered = select_qas_in_main_order(sample.get("qa", []), qfield)
            if len(ordered) != len(sols):
                print(f"[{style}] sample {s_idx}: ordered={len(ordered)} sols={len(sols)}", file=sys.stderr)
            id_to_text = build_dia_id_text_map(sample)
            for i, (q_idx, qa) in enumerate(ordered):
                if i >= len(sols):
                    break
                sol = sols[i] or {}
                gold = set(qa.get("evidence", []))
                retr = retrieved_dia_ids(
                    sol.get("docs") or [], id_to_text,
                    retrieved_metadata=sol.get("retrieved_metadata"),
                )
                recall, prec, f1 = prf(retr, gold)
                records.append({
                    "sample_idx": s_idx, "q_idx": q_idx, "category": int(qa.get("category", 0)),
                    "gold_dia_ids": sorted(gold), "retrieved_dia_ids": sorted(retr),
                    "recall": recall, "precision": prec, "f1": f1,
                })
        if not records:
            continue
        # Aggregate
        macro_r = sum(r["recall"] for r in records) / len(records)
        macro_p = sum(r["precision"] for r in records) / len(records)
        macro_f1 = sum(r["f1"] for r in records) / len(records)
        by_cat: Dict[int, Dict[str, float]] = defaultdict(lambda: {"n": 0, "recall": 0.0, "precision": 0.0, "f1": 0.0})
        for r in records:
            c = r["category"]
            by_cat[c]["n"] += 1
            by_cat[c]["recall"] += r["recall"]
            by_cat[c]["precision"] += r["precision"]
            by_cat[c]["f1"] += r["f1"]
        per_cat = {}
        for c, d in by_cat.items():
            per_cat[str(c)] = {
                "n": d["n"],
                "recall": d["recall"] / d["n"],
                "precision": d["precision"] / d["n"],
                "f1": d["f1"] / d["n"],
            }
        by_style[style] = {
            "n": len(records),
            "macro_recall": macro_r,
            "macro_precision": macro_p,
            "macro_f1": macro_f1,
            "per_category": per_cat,
        }

    # Style 5: composed
    if "composed" in styles and mm:
        records = []
        # Group clusters by sample
        clusters_by_sample: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        for c in mm:
            if c and c.get("composed_query"):
                clusters_by_sample[c["sample_idx"]].append(c)

        for s_idx, sample in enumerate(data):
            clusters = clusters_by_sample.get(s_idx, [])
            if not clusters:
                continue
            sample_dir = os.path.join(args.results_dir, f"sample_{s_idx}")
            cache_path = os.path.join(sample_dir, "composed_solutions.json")
            if not os.path.exists(cache_path):
                print(f"[composed] missing cache {cache_path}", file=sys.stderr)
                continue
            with open(cache_path) as f:
                sols = json.load(f)
            id_to_text = build_dia_id_text_map(sample)
            for entry in sols:
                gold = set(entry.get("gold_dia_ids", []))
                retr = retrieved_dia_ids(
                    entry.get("docs") or [], id_to_text,
                    retrieved_metadata=entry.get("retrieved_metadata"),
                )
                recall, prec, f1 = prf(retr, gold)
                records.append({
                    "sample_idx": s_idx, "cluster_id": entry.get("cluster_id"),
                    "gold_dia_ids": sorted(gold), "retrieved_dia_ids": sorted(retr),
                    "gold_size": len(gold),
                    "recall": recall, "precision": prec, "f1": f1,
                })
        if records:
            macro_r = sum(r["recall"] for r in records) / len(records)
            macro_p = sum(r["precision"] for r in records) / len(records)
            macro_f1 = sum(r["f1"] for r in records) / len(records)
            # Bucketed by gold size
            by_size: Dict[int, Dict[str, float]] = defaultdict(lambda: {"n": 0, "recall": 0.0, "precision": 0.0, "f1": 0.0})
            for r in records:
                bucket = min(r["gold_size"], 5)  # cap to 5+
                by_size[bucket]["n"] += 1
                by_size[bucket]["recall"] += r["recall"]
                by_size[bucket]["precision"] += r["precision"]
                by_size[bucket]["f1"] += r["f1"]
            per_size = {}
            for sz, d in by_size.items():
                per_size[str(sz)] = {
                    "n": d["n"],
                    "recall": d["recall"] / d["n"],
                    "precision": d["precision"] / d["n"],
                    "f1": d["f1"] / d["n"],
                }
            by_style["composed"] = {
                "n": len(records),
                "macro_recall": macro_r,
                "macro_precision": macro_p,
                "macro_f1": macro_f1,
                "per_gold_size": per_size,
            }

    out = {"by_style": by_style}
    with open(out_path, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    # Pretty print
    print(json.dumps(out, indent=2))
    print(f"\nWrote {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
