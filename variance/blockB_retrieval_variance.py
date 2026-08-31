"""Compute retrieval recall mean±std across 3 seeds (43, 44, 45) for the
LLM-indexed systems in Block B: AnchorMem, A-mem, mem0.

Uses the `question` style outputs from each seed's re-indexed retrieval:
  outputs/locomo-gemma-4-31B-it_seed{N}_t07/sample_X/queries_solutions.json
  outputs_amem/locomo-gemma-4-31B-it_seed{N}_t07/...
  outputs_mem0/...

Also computes BM25 and Dense recall as fixed baselines (deterministic, no
Block B seeds — recall is invariant).
"""
from __future__ import annotations
import json, os, sys
from collections import defaultdict
import statistics as st

sys.path.insert(0, 'baselines/AnchorMem/scripts')
from bucket_recall_vs_fact import (
    build_dia_id_text_map, extract_dia_ids, load_amem_note_to_dia,
)

ANCHOR = 'baselines/AnchorMem'
SEEDS = [43, 44, 45]

# Retrieval file layouts per system (Block B seeded)
SEEDED_DIRS = {
    "AnchorMem": "outputs/locomo-gemma-4-31B-it_seed{seed}_t07",
    "A-mem":     "outputs_amem/locomo-gemma-4-31B-it_seed{seed}_t07",
    "mem0":      "outputs_mem0/locomo-gemma-4-31B-it_seed{seed}_t07",
}
# For BM25/Dense — no seeded run, use main index
FIXED_DIRS = {
    "BM25":  "outputs_bm25/locomo-bm25",
    "Dense": "outputs_dense/locomo-dense",
}

# Style-specific filename
SEEDED_FILE = {
    "AnchorMem": "queries_solutions.json",
    "A-mem":     "queries_solutions_question.json",
    "mem0":      "queries_solutions_question.json",
}
FIXED_FILE = "queries_solutions_question.json"


def load_dataset():
    dialog_data = json.load(open(f'{ANCHOR}/data/locomo10_dialog.json'))
    sample_maps = {i: build_dia_id_text_map(s) for i, s in enumerate(dialog_data)}
    # Gold evidence per (sample_idx, q_idx)
    gold_lookup = {}
    for si, sample in enumerate(dialog_data):
        for q_idx, qa in enumerate(sample.get('qa', [])):
            ev = qa.get('evidence') or []
            gold_lookup[(si, q_idx)] = set(ev)
    return sample_maps, gold_lookup


def compute_seed_recall(system: str, seed: int | None, sample_idx: int,
                        sample_maps, gold_lookup, amem_note_maps):
    """Return list of (recall_val) for this (system, seed, sample)."""
    if seed is None:
        base = FIXED_DIRS[system]
        fname = FIXED_FILE
    else:
        base = SEEDED_DIRS[system].format(seed=seed)
        fname = SEEDED_FILE[system]
    p = f'{ANCHOR}/{base}/sample_{sample_idx}/{fname}'
    if not os.path.exists(p):
        return []
    data = json.load(open(p))
    note_map = amem_note_maps.get((system, seed, sample_idx))
    if system == "A-mem" and note_map is None:
        # Load note-to-dia map for A-mem seeded run
        note_p = f'{ANCHOR}/{base}/sample_{sample_idx}/amem_note_to_dia.json'
        note_map = json.load(open(note_p)) if os.path.exists(note_p) else {}
        amem_note_maps[(system, seed, sample_idx)] = note_map
    recalls = []
    for i, r in enumerate(data):
        q_idx = r.get('q_idx', i)
        gold = gold_lookup.get((sample_idx, q_idx))
        if not gold: continue
        retrieved = extract_dia_ids(system, r, sample_maps[sample_idx], note_map)
        recall = len(retrieved & gold) / len(gold)
        recalls.append(recall)
    return recalls


def main():
    sample_maps, gold_lookup = load_dataset()
    amem_note_maps = {}

    # Aggregate: {system: {seed: mean_recall_across_samples}}
    per_seed = defaultdict(dict)

    print("=== per-seed per-system recall (mean over 10 samples) ===")
    print(f"{'system':<10} {'seed':<6} {'n_queries':>10} {'recall':>8}")
    for system in ('AnchorMem', 'A-mem', 'mem0'):
        for seed in SEEDS:
            all_recall = []
            for s in range(10):
                all_recall.extend(compute_seed_recall(system, seed, s, sample_maps, gold_lookup, amem_note_maps))
            if all_recall:
                m = st.mean(all_recall)
                per_seed[system][seed] = m
                print(f"{system:<10} {seed:<6} {len(all_recall):>10} {m:>8.4f}")

    for system in ('BM25', 'Dense'):
        all_recall = []
        for s in range(10):
            all_recall.extend(compute_seed_recall(system, None, s, sample_maps, gold_lookup, amem_note_maps))
        if all_recall:
            m = st.mean(all_recall)
            per_seed[system][None] = m
            print(f"{system:<10} {'-':<6} {len(all_recall):>10} {m:>8.4f}")

    print()
    print("=== 3-seed variance summary ===")
    print(f"{'system':<12} {'mean':>8} {'std':>8} {'range':>18}")
    for system in ('AnchorMem', 'A-mem', 'mem0'):
        vals = list(per_seed[system].values())
        if not vals: continue
        m = st.mean(vals); sd = st.stdev(vals) if len(vals) > 1 else 0
        print(f"{system:<12} {m:>8.4f} {sd:>8.5f}  [{min(vals):.4f}, {max(vals):.4f}]")
    for system in ('BM25', 'Dense'):
        vals = list(per_seed[system].values())
        if not vals: continue
        m = vals[0]
        print(f"{system:<12} {m:>8.4f} {'--':>8}  (deterministic)")


if __name__ == "__main__":
    main()
