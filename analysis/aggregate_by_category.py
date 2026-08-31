"""Per-LoCoMo-category aggregation of response eval results.

Reads outputs_response_eval/<run>/scored.json, groups by
(style, variant, system, category), prints judge_acc / halluc_rate / mean_f1.

Cat 1: time
Cat 2: location/entity
Cat 3: multi-hop
Cat 4: open-domain reasoning
Cat 5: adversarial/unanswerable
"""
from __future__ import annotations
import argparse, json, sys
from collections import defaultdict


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="scored.json path")
    p.add_argument("--style_filter", default=None, help="optional: restrict to one style")
    p.add_argument("--variant_filter", default=None, help="optional: restrict to one variant")
    args = p.parse_args()

    data = json.load(open(args.input))

    # Group: (style, variant, system, category) → list of records
    groups = defaultdict(list)
    for r in data:
        if args.style_filter and r["style"] != args.style_filter:
            continue
        if args.variant_filter and r["variant"] != args.variant_filter:
            continue
        cat = r.get("category")
        if cat is None:
            cat = "composed"
        groups[(r["style"], r["variant"], r.get("system") or "-", cat)].append(r)

    # Aggregate
    rows = []
    for key, items in groups.items():
        style, variant, system, cat = key
        f1_items = [it["token_f1"] for it in items if it.get("gold_answer")]
        judge_items = [it["llm_judge"] for it in items if it.get("llm_judge") in ("yes", "no")]
        halluc_items = [it["cat5_halluc"] for it in items if it.get("cat5_halluc") in (0, 1)]
        rows.append({
            "style": style, "variant": variant, "system": system, "category": cat,
            "n": len(items),
            "mean_f1": sum(f1_items)/len(f1_items) if f1_items else 0.0,
            "judge_acc": sum(1 for j in judge_items if j == "yes")/len(judge_items) if judge_items else None,
            "halluc_rate": sum(halluc_items)/len(halluc_items) if halluc_items else None,
        })

    rows.sort(key=lambda r: (r["style"], r["variant"], r["system"], str(r["category"])))

    print(f"{'style':<10} | {'variant':<18} | {'system':<10} | {'cat':<4} | {'n':<4} | {'judge':>6} | {'halluc':>7} | {'F1':>5}")
    print('-' * 95)
    for r in rows:
        j = f"{r['judge_acc']:>6.3f}" if r['judge_acc'] is not None else "   -  "
        h = f"{r['halluc_rate']:>7.3f}" if r['halluc_rate'] is not None else "   -   "
        print(f"{r['style']:<10} | {r['variant']:<18} | {r['system']:<10} | {str(r['category']):<4} | {r['n']:<4} | {j} | {h} | {r['mean_f1']:>5.3f}")


if __name__ == "__main__":
    main()
