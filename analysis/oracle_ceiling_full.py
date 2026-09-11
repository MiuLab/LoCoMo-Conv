"""4-style oracle-ceiling decomposition.

For each (system, style) reports two key numbers PER BUCKET, plus interpretation caveat:

  RETR_B = orl_B - fB  (bucket B: partial recall queries — how much oracle beats sys)
  RETR_C = orl_C - fC  (bucket C: zero recall queries)
  REPR   = orl_A - fA  (bucket A: perfect recall — pure representation gap)
  CEIL   = 1 - orl_overall (score-metric limitation:
             fact_used / cf_score / coverage all reward only EXPLICIT fact restatement.
             Silent grounding / hedged-but-correct responses are penalized.
             CEIL therefore captures metric strictness, NOT model capability.)

Aggregated across 5 seeds.
"""
import json, sys
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

# ---- Load dataset + build helpers ----
dataset = json.load(open(f'{ANCHOR}/data/locomo10_dialog.json'))
sample_maps = {i: build_dia_id_text_map(s) for i, s in enumerate(dataset)}
retrieval_cache = {}
amem_note_maps = {}

def _get_note_map(sample_idx):
    if sample_idx not in amem_note_maps:
        amem_note_maps[sample_idx] = load_amem_note_to_dia(ANCHOR, sample_idx)
    return amem_note_maps[sample_idx]

def _retrieval_lookup(system, style, sample_idx, q_idx=None, cluster_id=None):
    """style=='composed' → keyed by cluster_id (use composed_solutions.json)."""
    key = (system, style, sample_idx)
    if key not in retrieval_cache:
        import os
        if style == 'composed':
            base = SYSTEM_DIRS[system]
            p = f"{ANCHOR}/{base}/sample_{sample_idx}/composed_solutions.json"
            m = {}
            if os.path.exists(p):
                data = json.load(open(p))
                note_map = _get_note_map(sample_idx) if system == "A-mem" else None
                for r in data:
                    cid = r.get("cluster_id")
                    m[cid] = extract_dia_ids(system, r, sample_maps[sample_idx], note_map)
            retrieval_cache[key] = m
        else:
            data = load_retrieval(ANCHOR, system, style, sample_idx)
            m = {}
            if data:
                note_map = _get_note_map(sample_idx) if system == "A-mem" else None
                for i, r in enumerate(data):
                    qidx = r.get("q_idx", i)
                    m[qidx] = extract_dia_ids(system, r, sample_maps[sample_idx], note_map)
            retrieval_cache[key] = m
    lookup_key = cluster_id if style == 'composed' else q_idx
    return retrieval_cache[key].get(lookup_key)

# ---- Style-specific config ----
STYLES_CFG = {
    'dialog':         {'score_file': 'fact_used_partial.json', 'score_field': 'fact_score', 'oracle_source': 'fact_used_partial/convergent.json', 'metric_name': 'fact_used'},
    'implicit':       {'score_file': 'fact_used_partial.json', 'score_field': 'fact_score', 'oracle_source': 'fact_used_partial/convergent.json', 'metric_name': 'fact_used'},
    'counterfactual': {'score_file': 'cf_3way.json',           'score_field': 'cf_score',   'oracle_source': 'cf_3way/convergent.json',            'metric_name': 'cf_score'},
    'composed':       {'score_file': 'composed_atomic.json',   'score_field': 'coverage_score', 'oracle_source': 'full_convergent/composed_atomic.json', 'metric_name': 'coverage'},
}

# ---- Build oracle lookup: (style, sample_idx, key) -> score ----
oracle_by_q = {}
for style, cfg in STYLES_CFG.items():
    orig = json.load(open(f"{ANCHOR}/outputs_response_eval/{cfg['oracle_source']}"))
    for r in orig:
        if r.get('variant') != 'oracle': continue
        if r.get('style') and r['style'] != style and style != 'composed':
            continue
        if style == 'composed':
            key_in_sample = r.get('cluster_id')
            sample_idx = int(r.get('sample_idx', -1))
        else:
            key_in_sample = r.get('q_idx')
            sample_idx = r.get('sample_idx')
        val = r.get(cfg['score_field'])
        if val is None: continue
        try:
            val = float(val)
        except:
            continue
        oracle_by_q[(style, sample_idx, key_in_sample)] = val

print(f"Loaded oracle scores: {len(oracle_by_q)} entries")

# ---- Accumulate per-bucket scores ----
# per_seed[(system, style)][seed]["A"|"B"|"C"] = list of (score, oracle_score)
per_seed = defaultdict(lambda: defaultdict(lambda: {"A": [], "B": [], "C": []}))

for seed in SEEDS:
    resps = json.load(open(f"{ANCHOR}/outputs_response_eval/blockA_seed{seed}/responses.json"))
    gold_lookup = {}
    for r in resps:
        style = r['style']
        if style == 'composed':
            k = (style, int(r['sample_idx']), r.get('cluster_id'), r['system'])
        else:
            k = (style, r['sample_idx'], r['q_idx'], r['system'])
        gold_lookup[k] = r.get('gold_evidence') or []

    for style, cfg in STYLES_CFG.items():
        p = f"{ANCHOR}/outputs_response_eval/blockA_seed{seed}/{cfg['score_file']}"
        d = json.load(open(p))
        for r in d:
            if r.get('variant') != 'top_k': continue
            system = r.get('system')
            if system not in SYSTEM_DIRS: continue
            if style == 'composed':
                if 'coverage_score' not in r: continue
                sample_idx = int(r['sample_idx'])
                key_in_sample = r.get('cluster_id')
                score = r['coverage_score']
            else:
                if r.get('style') != style: continue
                sample_idx = r['sample_idx']
                key_in_sample = r['q_idx']
                score = r.get(cfg['score_field'], 0.0)
            gold = set(gold_lookup.get((style, sample_idx, key_in_sample, system), []))
            if not gold: continue
            retrieved = _retrieval_lookup(system, style, sample_idx,
                                          q_idx=key_in_sample if style != 'composed' else None,
                                          cluster_id=key_in_sample if style == 'composed' else None)
            if retrieved is None: continue
            recall = len(retrieved & gold) / len(gold)
            oracle_score = oracle_by_q.get((style, sample_idx, key_in_sample))
            if oracle_score is None: continue
            bucket = "A" if recall >= 0.999 else ("B" if recall > 0 else "C")
            per_seed[(system, style)][seed][bucket].append((score, oracle_score))

# ---- Report ----
for style in ('dialog', 'implicit', 'counterfactual', 'composed'):
    print()
    print('=' * 116)
    metric = STYLES_CFG[style]['metric_name']
    print(f"### {style.upper()} — metric: {metric}")
    print('=' * 116)
    print(f"{'system':<10} {'sys':>6} {'oracle':>7}  "
          f"{'A%':>5} {'B%':>5} {'C%':>5}  "
          f"{'orl_A':>6} {'orl_B':>6} {'orl_C':>6}  "
          f"{'fA':>6} {'fB':>6} {'fC':>6}  "
          f"{'RETR_B':>7} {'RETR_C':>7} {'REPR':>7} {'CEIL':>7}")
    print('-' * 116)

    for system in ['AnchorMem', 'A-mem', 'Dense', 'BM25', 'mem0']:
        sd = per_seed.get((system, style), {})
        if not sd: continue

        agg = defaultdict(list)
        for seed, buckets in sd.items():
            A, B, C = buckets["A"], buckets["B"], buckets["C"]
            nA, nB, nC = len(A), len(B), len(C)
            total = nA + nB + nC
            if total == 0: continue
            agg["A_pct"].append(nA / total); agg["B_pct"].append(nB / total); agg["C_pct"].append(nC / total)
            agg["fA"].append(sum(x[0] for x in A)/nA if nA else 0)
            agg["fB"].append(sum(x[0] for x in B)/nB if nB else 0)
            agg["fC"].append(sum(x[0] for x in C)/nC if nC else 0)
            agg["oA"].append(sum(x[1] for x in A)/nA if nA else 0)
            agg["oB"].append(sum(x[1] for x in B)/nB if nB else 0)
            agg["oC"].append(sum(x[1] for x in C)/nC if nC else 0)
            sys_mean = (sum(x[0] for x in A)+sum(x[0] for x in B)+sum(x[0] for x in C))/total
            oracle_mean = (sum(x[1] for x in A)+sum(x[1] for x in B)+sum(x[1] for x in C))/total
            agg["sys"].append(sys_mean); agg["oracle"].append(oracle_mean)

        m = {k: st.mean(v) if v else 0 for k, v in agg.items()}

        RETR_B = m["oB"] - m["fB"]
        RETR_C = m["oC"] - m["fC"]
        REPR = m["oA"] - m["fA"]
        CEIL = 1.0 - m["oracle"]

        print(f"{system:<10} {m['sys']:>6.3f} {m['oracle']:>7.3f}  "
              f"{m['A_pct']*100:>4.1f}% {m['B_pct']*100:>4.1f}% {m['C_pct']*100:>4.1f}%  "
              f"{m['oA']:>6.3f} {m['oB']:>6.3f} {m['oC']:>6.3f}  "
              f"{m['fA']:>6.3f} {m['fB']:>6.3f} {m['fC']:>6.3f}  "
              f"{RETR_B:>+7.3f} {RETR_C:>+7.3f} {REPR:>+7.3f} {CEIL:>7.3f}")

print()
print("=" * 116)
print("Column caveats:")
print("  RETR_B / RETR_C : oracle − sys on bucket-B / bucket-C queries only")
print("  REPR            : oracle − sys on bucket-A queries only (retrieval perfect)")
print("  CEIL            : 1 − oracle overall. NOT an answer-model capability ceiling —")
print("                    reflects that fact_used / cf_score / coverage all reward")
print("                    EXPLICIT fact restatement. Responses using memory for silent")
print("                    grounding without stating gold facts are scored low here.")
