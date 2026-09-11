"""Sample queries for response-quality evaluation.

Per style (dialog, implicit, counterfactual, composed), pick 300 queries
stratified by LoCoMo category. Lock with seed=42 for reproducibility.
"""
from __future__ import annotations
import argparse, json, os, random, sys
from collections import defaultdict
from typing import Any, Dict, List


def has_style_query(qa: Dict[str, Any], style: str) -> bool:
    field = {
        "dialog": "dialog_query",
        "implicit": "implicit_query",
        "counterfactual": "counterfactual_query",
    }.get(style)
    return bool(field and qa.get(field))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_path", default="data/locomo10_dialog.json")
    p.add_argument("--multimem_path", default="data/locomo10_multimem_full.json")
    p.add_argument("--output", default="data/response_eval_sample.json")
    p.add_argument("--n_per_style", type=int, default=300)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    rng = random.Random(args.seed)
    with open(args.dataset_path) as f:
        data = json.load(f)

    out = {"meta": {"seed": args.seed, "n_per_style": args.n_per_style},
           "samples": {"dialog": [], "implicit": [], "counterfactual": [], "composed": []}}

    # Stratified sample per category for each QA-based style
    for style in ("dialog", "implicit", "counterfactual"):
        pool: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        for s_idx, sample in enumerate(data):
            for q_idx, qa in enumerate(sample.get("qa", [])):
                if not has_style_query(qa, style):
                    continue
                # counterfactual: skip cat 5 (no gold to be wrong about)
                if style == "counterfactual" and int(qa.get("category", 0)) == 5:
                    continue
                pool[int(qa.get("category", 0))].append({
                    "sample_idx": s_idx,
                    "q_idx": q_idx,
                    "category": int(qa.get("category", 0)),
                    "question": qa.get("question", ""),
                    "answer": qa.get("answer"),
                    "evidence": qa.get("evidence", []),
                    "query_field": {"dialog": "dialog_query", "implicit": "implicit_query",
                                    "counterfactual": "counterfactual_query"}[style],
                    "query": qa.get({"dialog": "dialog_query", "implicit": "implicit_query",
                                     "counterfactual": "counterfactual_query"}[style]),
                    "subject_speaker": (qa.get("counterfactual_subject_speaker") if style == "counterfactual"
                                         else qa.get("implicit_subject_speaker") if style == "implicit"
                                         else qa.get("subject_speaker")) or "speaker_a",
                    "adversarial_answer": qa.get("adversarial_answer"),
                })
        # Total target per style; distribute across present categories proportionally
        cats = sorted(pool.keys())
        # Equal-per-cat where possible; if some cats are small, redistribute
        target = args.n_per_style
        # First pass: cap each cat at target/|cats|
        equal_share = target // len(cats)
        sampled: List[Dict[str, Any]] = []
        leftover = 0
        for cat in cats:
            items = pool[cat]
            rng.shuffle(items)
            take = min(equal_share, len(items))
            sampled.extend(items[:take])
            leftover += equal_share - take
        # Distribute leftover from cats with more items
        if leftover > 0:
            for cat in cats:
                items = pool[cat]
                already = sum(1 for s in sampled if s["category"] == cat)
                remaining = items[already:]
                add = min(leftover, len(remaining))
                sampled.extend(remaining[:add])
                leftover -= add
                if leftover <= 0:
                    break
        out["samples"][style] = sampled[: target]

    # Composed: separate file
    if os.path.exists(args.multimem_path):
        with open(args.multimem_path) as f:
            mm = json.load(f)
        clusters = [c for c in mm if c and c.get("composed_query")]
        rng.shuffle(clusters)
        out["samples"]["composed"] = clusters[: args.n_per_style]

    # Stats
    print(f"Sampled per style:", file=sys.stderr)
    for style, items in out["samples"].items():
        cat_counts = defaultdict(int)
        for it in items:
            cat_counts[it.get("category", "?")] += 1
        print(f"  {style:<16}  n={len(items):<4}  per-cat={dict(sorted(cat_counts.items()))}", file=sys.stderr)

    with open(args.output, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"Wrote {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
