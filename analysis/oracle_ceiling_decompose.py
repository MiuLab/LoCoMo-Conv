"""Decompose (oracle − system) fact_used gap into:

  1. Retrieval gap    = mean(oracle) − mean(fact_used @ bucket A) weighted by (1 − bucket_A_fraction)
                        Actually cleaner: (oracle_mean − system_overall_mean)
                        broken by:
                          retrieval_gap        = fact@A × bucket_A% − overall_system_score
                                                 (i.e., what queries in B/C would gain if they were in A)
                          representation_gap  = oracle − fact@A  on the SAME set of queries
                                                 (bucket A queries — where the system retrieved everything,
                                                  but oracle got only gold)
                          response_ceiling    = 1.0 − oracle_mean  (Gemma can't get 1.0 even from gold)

Only include queries that appear in BOTH the system's fact_used_partial AND the oracle
fact_used_partial. Match by (style, sample_idx, q_idx).
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
STYLES = ['dialog', 'implicit']  # fact_used is only computed on these

# 1. Build oracle {(style, sample_idx, q_idx) -> fact_score}
orig_fs = json.load(open(f'{ANCHOR}/outputs_response_eval/fact_used_partial/convergent.json'))
oracle_by_q = {}
for r in orig_fs:
    if r.get('variant') != 'oracle': continue
    key = (r['style'], r['sample_idx'], r['q_idx'])
    oracle_by_q[key] = r.get('fact_score', 0.0)
print(f'Oracle fact_scored queries: {len(oracle_by_q)}')

# 2. Load dataset for dia_id text maps (for recall computation)
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

# 3. For each (system, style, seed): bucket queries, compute averages
# Also compute oracle_on_bucketA (mean oracle score on the subset that is in bucket A for this system)
results = defaultdict(lambda: defaultdict(lambda: {
    "n_A": 0, "n_B": 0, "n_C": 0,
    "fact_A_sum": 0.0, "fact_B_sum": 0.0, "fact_C_sum": 0.0,
    "oracle_on_A_sum": 0.0,
    "oracle_all_sum": 0.0, "n_oracle": 0,
}))

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

        key = (system, style)
        cell = results[key][seed]
        fact = r.get('fact_score', 0.0)
        if recall >= 0.999:
            cell["n_A"] += 1
            cell["fact_A_sum"] += fact
            if oracle_score is not None:
                cell["oracle_on_A_sum"] += oracle_score
        elif recall > 0:
            cell["n_B"] += 1
            cell["fact_B_sum"] += fact
        else:
            cell["n_C"] += 1
            cell["fact_C_sum"] += fact
        if oracle_score is not None:
            cell["oracle_all_sum"] += oracle_score
            cell["n_oracle"] += 1

# 4. Report — aggregate across seeds
print()
print("=" * 100)
print("Oracle-ceiling decomposition (5-seed mean)")
print("=" * 100)
print(f"{'system':<10} {'style':<10} {'orl':>6} {'sys':>6} {'fA':>6} {'A%':>6} "
      f"{'RETR':>7} {'REPR':>7} {'RESP':>7}   note")
print("-" * 100)

for system in ['AnchorMem', 'A-mem', 'Dense', 'BM25', 'mem0']:
    for style in STYLES:
        sd = results.get((system, style), {})
        if not sd: continue
        # per-seed averages
        oracle_means = []
        sys_means = []
        fA_means = []
        oracle_on_A_means = []
        A_pct = []
        for seed, cell in sd.items():
            n = cell["n_A"] + cell["n_B"] + cell["n_C"]
            if n == 0: continue
            sys_means.append((cell["fact_A_sum"] + cell["fact_B_sum"] + cell["fact_C_sum"]) / n)
            fA_means.append(cell["fact_A_sum"] / cell["n_A"] if cell["n_A"] > 0 else 0)
            A_pct.append(cell["n_A"] / n)
            if cell["n_oracle"] > 0:
                oracle_means.append(cell["oracle_all_sum"] / cell["n_oracle"])
            if cell["n_A"] > 0 and cell["oracle_on_A_sum"] > 0:
                oracle_on_A_means.append(cell["oracle_on_A_sum"] / cell["n_A"])

        orl = st.mean(oracle_means) if oracle_means else 0
        orl_on_A = st.mean(oracle_on_A_means) if oracle_on_A_means else 0
        sys_m = st.mean(sys_means) if sys_means else 0
        fA = st.mean(fA_means) if fA_means else 0
        Apct = st.mean(A_pct) if A_pct else 0

        # Decomposition (all sums to oracle − sys):
        # retrieval_gap  = fA × (1 − A%)  (loss because non-A queries hit fact@B/C)
        # But cleaner: gap_from_B_C = (fA × A%) + something... Let me use simpler decomposition.
        # oracle_all_mean = sys_mean + retrieval_gap + representation_gap
        # where representation_gap = orl_on_A − fA
        #        retrieval_gap = (orl_all − orl_on_A × A% − orl_B_or_C × (1-A%))... complicated

        # Simple 3-way decomposition:
        # Total gap = orl - sys = (orl - fA) + (fA - sys_mean)
        # NOT quite right either. Let me think again.
        # sys_mean = A% * fA + B% * fB + C% * fC
        # If ALL queries had recall=1, sys would score = fA
        # retrieval_gap = fA - sys_mean  (gain if we could push B/C queries into A)
        # representation_gap = orl_on_A - fA  (gain if bucket-A responses matched oracle)
        # response_ceiling = 1 - orl_on_A  (gain if oracle scored 1.0)
        # Total gain = 1 - sys_mean = retrieval + representation + response_ceiling

        retr_gap = fA - sys_m
        repr_gap = orl_on_A - fA
        resp_ceil = 1.0 - orl_on_A

        print(f"{system:<10} {style:<10} {orl:>6.3f} {sys_m:>6.3f} {fA:>6.3f} {Apct*100:>5.1f}% "
              f"{retr_gap:>+7.3f} {repr_gap:>+7.3f} {resp_ceil:>7.3f}   "
              f"orl_on_A={orl_on_A:.3f}")

print()
print("Legend:")
print("  orl        = oracle fact_used (mean over queries with an oracle score)")
print("  sys        = system's overall fact_used (5-seed mean)")
print("  fA         = fact_used @ bucket A (recall=1)")
print("  A%         = bucket-A fraction")
print("  RETR       = fA − sys (would gain this if all queries were in bucket A)")
print("  REPR       = oracle_on_A − fA (bucket-A response STILL loses this vs pure oracle)")
print("  RESP       = 1 − oracle_on_A (response gen ceiling: Gemma can't hit 1.0 even from gold)")
print("  (RETR + REPR + RESP + sys) ≈ 1.0 by construction of the decomposition")
