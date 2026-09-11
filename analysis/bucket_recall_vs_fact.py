"""Bucket queries by retrieval recall level and report response-quality (fact_used) mean per bucket.

Buckets:
  A. recall == 1.0     (perfect retrieval, all gold turns in top-k)
  B. 0 < recall < 1.0  (partial)
  C. recall == 0       (no gold turn retrieved)

For each (system, style, seed), aggregate:
  - #queries in bucket
  - mean fact_used in bucket

This tells us whether failure is retrieval-side (many queries in bucket C)
or response-side (fact_used low even in bucket A).
"""
import json, os, argparse
from collections import defaultdict

# Retrieval output paths per system (main-run, no seed suffix)
SYSTEM_DIRS = {
    "AnchorMem": "outputs/locomo-gemma-4-31B-it",
    "A-mem":     "outputs_amem/with_evo/locomo-gemma-4-31B-it",
    "mem0":      "outputs_mem0/locomo-gemma-4-31B-it",
    "BM25":      "outputs_bm25/locomo-bm25",
    "Dense":     "outputs_dense/locomo-dense",
}
FIELD_FOR_STYLE = {
    "dialog": "dialog_query",
    "implicit": "implicit_query",
    "counterfactual": "counterfactual_query",
    "composed": "composed",  # composed has different structure
}


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
    """Reconstruct dia_ids by substring matching turn text in docs."""
    out = set()
    for d in docs or []:
        d_lower = (d or "").lower()
        for did, txt in id_to_text.items():
            if txt and txt in d_lower:
                out.add(did)
    return out


def load_retrieval(anchor_dir, system, style, sample_idx):
    field = FIELD_FOR_STYLE.get(style)
    if field is None: return None
    base = SYSTEM_DIRS[system]
    p = os.path.join(anchor_dir, base, f"sample_{sample_idx}", f"queries_solutions_{field}.json")
    if not os.path.exists(p): return None
    try:
        return json.load(open(p))
    except Exception:
        return None


def load_amem_note_to_dia(anchor_dir, sample_idx):
    p = os.path.join(anchor_dir, SYSTEM_DIRS["A-mem"], f"sample_{sample_idx}", "amem_note_to_dia.json")
    if not os.path.exists(p): return {}
    return json.load(open(p))


def extract_dia_ids(system, record, dia_text_map, note_to_dia=None):
    """System-specific extraction of retrieved dia_ids."""
    out = set()
    if system == "mem0":
        for meta in record.get("retrieved_metadata", []) or []:
            if isinstance(meta, dict) and meta.get("dia_ids"):
                for did in str(meta["dia_ids"]).split(","):
                    did = did.strip()
                    if did:
                        out.add(did)
    elif system == "A-mem":
        for nid in record.get("retrieved_note_ids", []) or []:
            for did in (note_to_dia or {}).get(nid, []):
                out.add(did)
    else:  # AnchorMem, BM25, Dense — text-based reconstruction
        for d in record.get("docs", []) or []:
            d_lower = (d or "").lower()
            for did, txt in dia_text_map.items():
                if txt and txt in d_lower:
                    out.add(did)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--anchor_dir", default=".")
    p.add_argument("--seeds", default="43,44,45,46,47")
    p.add_argument("--top_k", type=int, default=10)
    p.add_argument("--styles", default="dialog,implicit,counterfactual")
    args = p.parse_args()

    seeds = [int(x) for x in args.seeds.split(",")]
    styles = args.styles.split(",")
    dataset = json.load(open(os.path.join(args.anchor_dir, "data/locomo10_dialog.json")))
    sample_maps = {i: build_dia_id_text_map(s) for i, s in enumerate(dataset)}

    # Cache retrieval-file lookups by (system, style, sample_idx) → {q_idx: retrieved_dia_ids}
    retrieval_cache = {}
    amem_note_maps = {}

    def _get_note_map(sample_idx):
        if sample_idx not in amem_note_maps:
            amem_note_maps[sample_idx] = load_amem_note_to_dia(args.anchor_dir, sample_idx)
        return amem_note_maps[sample_idx]

    def _retrieval_lookup(system, style, sample_idx, q_idx):
        key = (system, style, sample_idx)
        if key not in retrieval_cache:
            data = load_retrieval(args.anchor_dir, system, style, sample_idx)
            m = {}
            if data:
                note_map = _get_note_map(sample_idx) if system == "A-mem" else None
                for i, r in enumerate(data):
                    qidx_from = r.get("q_idx", i)
                    m[qidx_from] = extract_dia_ids(system, r, sample_maps[sample_idx], note_map)
            retrieval_cache[key] = m
        return retrieval_cache[key].get(q_idx)

    # Bucket & aggregate: {(system, style, bucket): {seed: {'n': N, 'fact_sum': X}}}
    agg = defaultdict(lambda: defaultdict(lambda: {"n": 0, "fact_sum": 0.0}))
    for seed in seeds:
        d = json.load(open(f"{args.anchor_dir}/outputs_response_eval/blockA_seed{seed}/fact_used_partial.json"))
        # Also need gold_evidence per record — this is on the responses.json
        resps = json.load(open(f"{args.anchor_dir}/outputs_response_eval/blockA_seed{seed}/responses.json"))
        gold_lookup = {(r["style"], r["sample_idx"], r["q_idx"], r["system"]): r.get("gold_evidence") or []
                       for r in resps}
        for r in d:
            if r.get("variant") != "top_k": continue
            style = r["style"]
            if style not in styles: continue
            system = r["system"]
            if system not in SYSTEM_DIRS: continue
            sample_idx = r["sample_idx"]
            q_idx = r["q_idx"]
            fact = r.get("fact_score", 0.0)
            gold = set(gold_lookup.get((style, sample_idx, q_idx, system), []))
            if not gold: continue
            retrieved = _retrieval_lookup(system, style, sample_idx, q_idx)
            if retrieved is None:
                continue
            recall = len(retrieved & gold) / len(gold)
            if recall >= 0.999:
                bucket = "A_full"
            elif recall > 0:
                bucket = "B_partial"
            else:
                bucket = "C_zero"
            agg[(system, style, bucket)][seed]["n"] += 1
            agg[(system, style, bucket)][seed]["fact_sum"] += fact

    # Report per (system, style): mean±std across seeds within each bucket
    import statistics as st
    print(f"{'system':<10} {'style':<14} {'bucket':<10} {'n_mean':>7} {'fact_mean':>10} {'fact_std':>9}")
    print("-" * 65)
    for system in SYSTEM_DIRS:
        for style in styles:
            for bucket in ("A_full", "B_partial", "C_zero"):
                ss = agg.get((system, style, bucket), {})
                if not ss:
                    continue
                n_vals = [v["n"] for v in ss.values()]
                fact_vals = [v["fact_sum"] / v["n"] if v["n"] > 0 else 0 for v in ss.values()]
                if len(n_vals) == 0: continue
                nm = st.mean(n_vals)
                fm = st.mean(fact_vals)
                fs = st.stdev(fact_vals) if len(fact_vals) > 1 else 0
                print(f"{system:<10} {style:<14} {bucket:<10} {nm:>7.1f} {fm:>10.3f} {fs:>9.4f}")
        print()


if __name__ == "__main__":
    main()
