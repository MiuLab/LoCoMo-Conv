"""Cleaner oracle-ceiling decomposition.

For each system:
  1. Bucket every query as A/B/C by retrieval recall.
  2. Compute oracle fact_used on that SAME bucket's queries.
  3. Decompose (1 − sys) into:
       Response ceiling  = 1 − oracle_mean_all
       Retrieval loss    = B% × (oracle_B − fB) + C% × (oracle_C − fC)
       Representation loss = A% × (oracle_A − fA)
     These three + sys ≈ 1.0.

Also reports oracle score per bucket to show whether hard queries (bucket C)
have lower oracle ceilings than easy queries (bucket A).
"""
import json, os, sys
from collections import defaultdict
import statistics as st

for _d in ('retrieval','response_eval','analysis','construction'):
    sys.path.insert(0, _d)
from bucket_recall_vs_fact import (
    SYSTEM_DIRS, build_dia_id_text_map, load_amem_note_to_dia,
    load_retrieval, extract_dia_ids,
)

ANCHOR = '.'
SEEDS = [43, 44, 45, 46, 47]
STYLES = ['dialog', 'implicit']

orig_fs = json.load(open(f'{ANCHOR}/outputs_response_eval/fact_used_partial/convergent.json'))
oracle_by_q = {}
for r in orig_fs:
    if r.get('variant') != 'oracle': continue
    oracle_by_q[(r['style'], r['sample_idx'], r['q_idx'])] = r.get('fact_score', 0.0)

dataset = json.load(open(f'{ANCHOR}/data/locomo10_dialog.json'))
sample_maps = {i: build_dia_id_text_map(s) for i, s in enumerate(dataset)}
retrieval_cache = {}
amem_note_maps = {}

def _get_note_map(sample_idx):
    if sample_idx not in amem_note_maps:
        amem_note_maps[sample_idx] = load_amem_note_to_dia(ANCHOR, sample_idx)
    return amem_note_maps[sample_idx]

def _retrieval_lookup(system, style, sample_idx, q_idx):
    key = (system, style, sample_idx)
    if key not in retrieval_cache:
        data = load_retrieval(ANCHOR, system, style, sample_idx)
        m = {}
        if data:
            note_map = _get_note_map(sample_idx) if system == "A-mem" else None
            for i, r in enumerate(data):
                qidx = r.get("q_idx", i)
                m[qidx] = extract_dia_ids(system, r, sample_maps[sample_idx], note_map)
        retrieval_cache[key] = m
    return retrieval_cache[key].get(q_idx)

# {(system, style): {seed: {"A": [(fact, oracle)], "B": [...], "C": [...]}}}
per_seed = defaultdict(lambda: defaultdict(lambda: {"A": [], "B": [], "C": []}))

for seed in SEEDS:
    fs = json.load(open(f'{ANCHOR}/outputs_response_eval/blockA_seed{seed}/fact_used_partial.json'))
    resps = json.load(open(f'{ANCHOR}/outputs_response_eval/blockA_seed{seed}/responses.json'))
    gold_lookup = {(r['style'], r['sample_idx'], r['q_idx'], r['system']): r.get('gold_evidence') or []
                   for r in resps}
    for r in fs:
        if r.get('variant') != 'top_k': continue
        style = r['style']; system = r['system']
        if style not in STYLES: continue
        if system not in SYSTEM_DIRS: continue
        sample_idx = r['sample_idx']; q_idx = r['q_idx']
        gold = set(gold_lookup.get((style, sample_idx, q_idx, system), []))
        if not gold: continue
        retrieved = _retrieval_lookup(system, style, sample_idx, q_idx)
        if retrieved is None: continue
        recall = len(retrieved & gold) / len(gold)
        oracle_score = oracle_by_q.get((style, sample_idx, q_idx))
        if oracle_score is None: continue  # need paired oracle to make comparison honest
        fact = r.get('fact_score', 0.0)
        bucket = "A" if recall >= 0.999 else ("B" if recall > 0 else "C")
        per_seed[(system, style)][seed][bucket].append((fact, oracle_score))

# Report
print()
print("=" * 118)
print("Oracle-per-bucket decomposition (5-seed mean)")
print("=" * 118)
print(f"{'system':<10} {'style':<10} "
      f"{'sys':>6} {'oracle':>7} "
      f"{'A%':>5} {'B%':>5} {'C%':>5}   "
      f"{'orl_A':>6} {'orl_B':>6} {'orl_C':>6}   "
      f"{'fA':>6} {'fB':>6} {'fC':>6}   "
      f"{'RETR':>7} {'REPR':>7} {'RESP':>7}")
print("-" * 118)

for system in ['AnchorMem', 'A-mem', 'Dense', 'BM25', 'mem0']:
    for style in STYLES:
        sd = per_seed.get((system, style), {})
        if not sd: continue

        # Aggregate per seed → mean across seeds
        agg_seed = {"A_pct": [], "B_pct": [], "C_pct": [],
                    "fA": [], "fB": [], "fC": [],
                    "oA": [], "oB": [], "oC": [],
                    "sys": [], "oracle": []}
        for seed, buckets in sd.items():
            A, B, C = buckets["A"], buckets["B"], buckets["C"]
            nA, nB, nC = len(A), len(B), len(C)
            total = nA + nB + nC
            if total == 0: continue
            agg_seed["A_pct"].append(nA / total)
            agg_seed["B_pct"].append(nB / total)
            agg_seed["C_pct"].append(nC / total)
            agg_seed["fA"].append(sum(x[0] for x in A)/nA if nA else 0)
            agg_seed["fB"].append(sum(x[0] for x in B)/nB if nB else 0)
            agg_seed["fC"].append(sum(x[0] for x in C)/nC if nC else 0)
            agg_seed["oA"].append(sum(x[1] for x in A)/nA if nA else 0)
            agg_seed["oB"].append(sum(x[1] for x in B)/nB if nB else 0)
            agg_seed["oC"].append(sum(x[1] for x in C)/nC if nC else 0)
            sys_mean = (sum(x[0] for x in A) + sum(x[0] for x in B) + sum(x[0] for x in C)) / total
            oracle_mean = (sum(x[1] for x in A) + sum(x[1] for x in B) + sum(x[1] for x in C)) / total
            agg_seed["sys"].append(sys_mean)
            agg_seed["oracle"].append(oracle_mean)

        m = {k: st.mean(v) if v else 0 for k, v in agg_seed.items()}

        # New decomposition:
        # RESP = 1 − oracle_mean
        # REPR = A% × (orl_A − fA)  (bucket A: retrieval perfect, gap is representation)
        # RETR = B% × (orl_B − fB) + C% × (orl_C − fC)  (gap where retrieval was imperfect)
        # sys + RESP + REPR + RETR = 1.0

        resp = 1 - m["oracle"]
        repr_loss = m["A_pct"] * (m["oA"] - m["fA"])
        retr_loss = m["B_pct"] * (m["oB"] - m["fB"]) + m["C_pct"] * (m["oC"] - m["fC"])

        print(f"{system:<10} {style:<10} "
              f"{m['sys']:>6.3f} {m['oracle']:>7.3f} "
              f"{m['A_pct']*100:>4.1f}% {m['B_pct']*100:>4.1f}% {m['C_pct']*100:>4.1f}%   "
              f"{m['oA']:>6.3f} {m['oB']:>6.3f} {m['oC']:>6.3f}   "
              f"{m['fA']:>6.3f} {m['fB']:>6.3f} {m['fC']:>6.3f}   "
              f"{retr_loss:>+7.3f} {repr_loss:>+7.3f} {resp:>7.3f}")

        # Verification
        total_check = m["sys"] + retr_loss + repr_loss + resp
        # print(f"  sanity: {m['sys']:.3f} + {retr_loss:.3f} + {repr_loss:.3f} + {resp:.3f} = {total_check:.3f}")

print()
print("=" * 118)
print("Comparison of oracle per bucket:")
print("  If orl_A > orl_B > orl_C, that confirms bucket-C queries are intrinsically harder.")
print("  New RETR loss uses each bucket's actual ceiling, not fA blanket assumption.")
