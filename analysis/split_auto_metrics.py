"""Split Run 2 (--query_field auto) metrics into explicit vs implicit subsets.

For each qa item we compute Recall@5 (vs locomo evidence) and EM/F1 (vs answer).
We bucket each item into 'explicit' (used dialog_query) or 'implicit' (used implicit_query).
"""

from __future__ import annotations
import json
import os
import sys
from collections import defaultdict
from typing import Dict, List, Any

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data/locomo10_dialog.json")
RESULTS_DIR = os.path.join(ROOT, "outputs/locomo-gemma-4-31B-it")
CACHE_SUFFIX = "_auto"  # queries_solutions_auto.json


def f1_score(pred: str, gold: str) -> float:
    pred_tokens = pred.lower().split()
    gold_tokens = gold.lower().split()
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = set(pred_tokens) & set(gold_tokens)
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def em_score(pred: str, gold: str) -> int:
    return int(gold.lower().strip() in pred.lower().strip())


def select_qas_in_main_order(sample_qa: List[Dict[str, Any]], query_field: str) -> List[Dict[str, Any]]:
    """Reproduce main.py's selection logic for --query_field=auto."""
    selected = []
    for q_idx, qa in enumerate(sample_qa):
        cat = int(qa.get("category", 0))
        if cat == 5 and query_field == "question":
            continue
        if query_field == "question":
            qstr = qa.get("question") or ""
        elif query_field == "dialog_query":
            qstr = qa.get("dialog_query") or ""
        elif query_field == "implicit_query":
            qstr = qa.get("implicit_query") or ""
        else:  # auto
            qstr = qa.get("implicit_query") or qa.get("dialog_query") or qa.get("question", "")
        if not qstr:
            continue
        used_field = "implicit_query" if qa.get("implicit_query") else "dialog_query"
        selected.append({**qa, "_q_idx": q_idx, "_used_field": used_field})
    return selected


def main():
    with open(DATA) as f:
        data = json.load(f)

    # Stat buckets: bucket -> list of dicts {category, recall, em, f1}
    rows: List[Dict[str, Any]] = []

    for s_idx, sample in enumerate(data):
        sample_dir = os.path.join(RESULTS_DIR, f"sample_{s_idx}")
        cache_path = os.path.join(sample_dir, f"queries_solutions{CACHE_SUFFIX}.json")
        if not os.path.exists(cache_path):
            print(f"[sample {s_idx}] cache missing", file=sys.stderr)
            continue
        with open(cache_path) as f:
            sols = json.load(f)

        ordered = select_qas_in_main_order(sample.get("qa", []), "auto")
        if len(ordered) != len(sols):
            print(f"[sample {s_idx}] mismatch len(ordered)={len(ordered)} len(sols)={len(sols)}",
                  file=sys.stderr)

        # Build doc_id -> doc text map for THIS sample (for recall computation)
        # We need to match retrieved docs to evidence dia_ids. Rebuild from conversation.
        id_doc_map: Dict[str, str] = {}
        for k, v in sample.get("conversation", {}).items():
            if not (isinstance(k, str) and k.startswith("session_") and "date" not in k):
                continue
            if not isinstance(v, list):
                continue
            for turn in v:
                did = turn.get("dia_id")
                txt = (turn.get("text") or "").strip()
                if did and txt:
                    id_doc_map[did] = txt.lower()

        for i, qa in enumerate(ordered):
            if i >= len(sols):
                break
            sol = sols[i] or {}
            cat = int(qa.get("category", 0))
            evidence_set = set(qa.get("evidence", []))
            retrieved_docs = sol.get("docs") or []

            # Recall: same logic as main.py (substring match into retrieved doc text)
            found = set()
            for retrieved in retrieved_docs:
                rt = (retrieved or "").lower()
                for eid in evidence_set:
                    if eid in id_doc_map and id_doc_map[eid] in rt:
                        found.add(eid)
            recall = len(found) / len(evidence_set) if evidence_set else 0.0

            # Answer-side metrics
            response = sol.get("answer") or ""
            gold = qa.get("answer")
            if gold is None or cat == 5:
                em = None
                f1 = None
            else:
                gold_str = ", ".join(str(a) for a in gold) if isinstance(gold, list) else str(gold)
                em = em_score(response, gold_str)
                f1 = f1_score(response, gold_str)

            rows.append({
                "sample_idx": s_idx,
                "q_idx": qa["_q_idx"],
                "category": cat,
                "naturalness": qa.get("naturalness", "n/a"),
                "used_field": qa["_used_field"],
                "recall": recall,
                "em": em,
                "f1": f1,
            })

    # Aggregate by bucket
    def agg(rs: List[Dict[str, Any]], field: str) -> float:
        vals = [r[field] for r in rs if r[field] is not None]
        return sum(vals) / len(vals) if vals else 0.0

    def report(name: str, rs: List[Dict[str, Any]]):
        if not rs:
            print(f"\n## {name}: empty")
            return
        print(f"\n## {name} (n={len(rs)})")
        print(f"  Macro Recall: {agg(rs, 'recall'):.4f}")
        print(f"  EM:           {agg(rs, 'em'):.4f}  (n={sum(1 for r in rs if r['em'] is not None)})")
        print(f"  F1:           {agg(rs, 'f1'):.4f}  (n={sum(1 for r in rs if r['f1'] is not None)})")
        by_cat = defaultdict(list)
        for r in rs:
            by_cat[r["category"]].append(r)
        for cat in sorted(by_cat):
            sub = by_cat[cat]
            print(f"    cat {cat} (n={len(sub)}): "
                  f"Recall={agg(sub, 'recall'):.4f}  "
                  f"EM={agg(sub, 'em'):.4f}  F1={agg(sub, 'f1'):.4f}")

    explicit = [r for r in rows if r["used_field"] == "dialog_query"]
    implicit = [r for r in rows if r["used_field"] == "implicit_query"]

    report("ALL (auto run, mixed)", rows)
    report("EXPLICIT subset (dialog_query path)", explicit)
    report("IMPLICIT subset (implicit_query path)", implicit)

    out = {
        "all": {"n": len(rows), "macro_recall": agg(rows, "recall"), "em": agg(rows, "em"), "f1": agg(rows, "f1")},
        "explicit": {"n": len(explicit), "macro_recall": agg(explicit, "recall"), "em": agg(explicit, "em"), "f1": agg(explicit, "f1")},
        "implicit": {"n": len(implicit), "macro_recall": agg(implicit, "recall"), "em": agg(implicit, "em"), "f1": agg(implicit, "f1")},
        "rows": rows,
    }
    out_path = os.path.join(RESULTS_DIR, "auto_split_metrics.json")
    with open(out_path, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
