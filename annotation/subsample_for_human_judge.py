"""Subsample scored responses for cross-judge + human annotation.

Outputs:
  outputs_response_eval/human_judge/subset.json  — list of items with
    {item_id, sample_idx, q_idx, style, variant, system, k, query, gold, response, gpt_judge}
  outputs_response_eval/human_judge/label_studio.json  — same items in LS task format
"""
from __future__ import annotations
import argparse, json, os, random, sys
from collections import defaultdict


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="outputs_response_eval/oracle_all_styles/scored.json")
    p.add_argument("--output_dir", default="outputs_response_eval/human_judge")
    p.add_argument("--target_total", type=int, default=1500)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    rng = random.Random(args.seed)
    data = json.load(open(args.input))

    # Stratify by (style, variant). Within each cell, sample proportionally.
    cells = defaultdict(list)
    for r in data:
        # Skip cat-5 (only halluc, no judge yes/no on gold) for cleaner human task
        cat = r.get('category')
        if cat == 5: continue
        if not r.get('gold_answer'): continue
        if not r.get('response'): continue
        if r.get('llm_judge') not in ('yes','no'): continue
        cells[(r['style'], r['variant'])].append(r)

    print(f"Cells: {len(cells)}", file=sys.stderr)
    per_cell = args.target_total // len(cells)
    print(f"Per-cell target: {per_cell}", file=sys.stderr)

    subset = []
    for cell, items in cells.items():
        rng.shuffle(items)
        sampled = items[:per_cell]
        subset.extend(sampled)

    rng.shuffle(subset)
    # Assign stable item_id
    for i, r in enumerate(subset):
        r['item_id'] = i

    # Save raw subset
    out_subset = []
    for r in subset:
        out_subset.append({
            'item_id': r['item_id'],
            'sample_idx': r['sample_idx'],
            'q_idx': r.get('q_idx'),
            'style': r['style'],
            'variant': r['variant'],
            'system': r.get('system'),
            'k': r.get('k'),
            'category': r.get('category'),
            'query': r['query'],
            'gold_answer': r['gold_answer'],
            'response': r['response'],
            'gpt_judge': r['llm_judge'],
        })
    subset_path = os.path.join(args.output_dir, "subset.json")
    with open(subset_path, 'w') as f:
        json.dump(out_subset, f, ensure_ascii=False, indent=2)
    print(f"Wrote subset ({len(out_subset)} items) → {subset_path}", file=sys.stderr)

    # Distribution check
    from collections import Counter
    print("Per (style, variant) count:", file=sys.stderr)
    by = Counter((r['style'], r['variant']) for r in out_subset)
    for k, v in sorted(by.items()):
        print(f"  {k}: {v}", file=sys.stderr)

    # Label Studio format: each task wraps fields in `data`
    ls_tasks = []
    for r in out_subset:
        ls_tasks.append({
            "id": r['item_id'],
            "data": {
                "item_id": r['item_id'],
                "style": r['style'],
                "variant": r['variant'],
                "system": r.get('system') or "-",
                "category": r.get('category') or "-",
                "query": r['query'],
                "gold": r['gold_answer'],
                "response": r['response'],
                "gpt_judge": r['gpt_judge'],
            }
        })
    ls_path = os.path.join(args.output_dir, "label_studio.json")
    with open(ls_path, 'w') as f:
        json.dump(ls_tasks, f, ensure_ascii=False, indent=2)
    print(f"Wrote LS tasks → {ls_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
