"""Emit the FULL set of LoCoMo QA queries (not sampled) in the same format
the response-eval driver consumes.

Output: data/response_eval_full.json
Same schema as data/response_eval_sample.json.
"""
from __future__ import annotations
import argparse, json, os, sys
from collections import defaultdict


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_path", default="data/locomo10_dialog.json")
    p.add_argument("--multimem_path", default="data/locomo10_multimem_full.json")
    p.add_argument("--output", default="data/response_eval_full.json")
    args = p.parse_args()

    with open(args.dataset_path) as f:
        data = json.load(f)

    out = {"meta": {"mode": "full"},
           "samples": {"dialog": [], "implicit": [], "counterfactual": [], "composed": []}}

    field_for = {"dialog": "dialog_query", "implicit": "implicit_query",
                 "counterfactual": "counterfactual_query"}

    for style in ("dialog", "implicit", "counterfactual"):
        field = field_for[style]
        for s_idx, sample in enumerate(data):
            for q_idx, qa in enumerate(sample.get("qa", [])):
                cat = int(qa.get("category", 0))
                if style == "counterfactual" and cat == 5:
                    continue
                if not qa.get(field):
                    continue
                out["samples"][style].append({
                    "sample_idx": s_idx,
                    "q_idx": q_idx,
                    "category": cat,
                    "question": qa.get("question", ""),
                    "answer": qa.get("answer"),
                    "evidence": qa.get("evidence", []),
                    "query_field": field,
                    "query": qa.get(field),
                    "subject_speaker": (qa.get("counterfactual_subject_speaker") if style == "counterfactual"
                                         else qa.get("implicit_subject_speaker") if style == "implicit"
                                         else qa.get("subject_speaker")) or "speaker_a",
                    "adversarial_answer": qa.get("adversarial_answer"),
                })

    if os.path.exists(args.multimem_path):
        with open(args.multimem_path) as f:
            mm = json.load(f)
        out["samples"]["composed"] = [c for c in mm if c.get("composed_query")]

    print("Per-style counts:", file=sys.stderr)
    for style, items in out["samples"].items():
        cat_counts = defaultdict(int)
        for it in items:
            cat_counts[it.get("category", "?")] += 1
        print(f"  {style:<16}  n={len(items):<5}  per-cat={dict(sorted(cat_counts.items()))}", file=sys.stderr)

    with open(args.output, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"Wrote {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
