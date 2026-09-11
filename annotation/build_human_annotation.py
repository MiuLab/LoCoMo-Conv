"""Build human-annotation packets for (1) silent grounding and (2) CoT vs Oracle.

Outputs under rebuttal/human_annotation/:
  silent_grounding_annotation.csv  — 332 cases; per case the query + THREE responses
                                     (oracle / no-memory / random) in randomized,
                                     blinded order (labels A/B/C).
  silent_grounding_key.json        — mapping (case → which label is which variant)
                                     + Opus per-dim scores. KEEP AWAY FROM ANNOTATORS.

  cot_vs_oracle_annotation.csv     — pairwise; per case query + two responses in
                                     randomized order (A/B). All 5 systems on the
                                     full implicit set (1536 × 5 rows).
  cot_vs_oracle_key.json           — case → (which of A/B is CoT), system, Opus verdict.

Annotator task mirrors the Opus rubric: score each response 0 / 0.5 / 1 on
faithfulness, relevance, engagement (columns provided empty in the CSV).
"""
import csv, json, os, random
from collections import defaultdict

ANCHOR = '.'
RE = f'{ANCHOR}/outputs_response_eval'
OUT = f'{ANCHOR}/rebuttal/human_annotation'
os.makedirs(OUT, exist_ok=True)
rng = random.Random(20260711)

# ---------------- shared: queries ----------------
dialog_data = json.load(open(f'{ANCHOR}/data/locomo10_dialog.json'))
def implicit_query(sample_idx, q_idx):
    qa = dialog_data[sample_idx]['qa'][q_idx]
    return qa.get('implicit_query') or ''

# ---------------- 1. Silent grounding ----------------
scores = json.load(open(f'{RE}/e3_silent/scores.json'))

# response lookups
oracle_resp = {}
for r in json.load(open(f'{RE}/full_convergent/responses.json')):
    if r.get('variant') == 'oracle' and r.get('style') == 'implicit':
        oracle_resp[(r['sample_idx'], r['q_idx'])] = r['response']
nomem_resp = {}
for r in json.load(open(f'{RE}/no_memory/responses.json')):
    if r.get('style') == 'implicit':
        nomem_resp[(r['sample_idx'], r['q_idx'])] = r['response']
random_resp = {}
for r in json.load(open(f'{RE}/random_mem/responses.json')):
    random_resp[(r['sample_idx'], r['q_idx'])] = r['response']

rows, key = [], {}
missing = 0
for rec in scores:
    k = (rec['sample_idx'], rec['q_idx'])
    q = implicit_query(*k)
    ro, rn, rr = oracle_resp.get(k), nomem_resp.get(k), random_resp.get(k)
    if not (q and ro and rn and rr):
        missing += 1
        continue
    variants = [('oracle', ro), ('no_memory', rn), ('random', rr)]
    rng.shuffle(variants)
    labels = ['A', 'B', 'C']
    case_id = f"sg_{k[0]}_{k[1]}"
    row = {'case_id': case_id, 'query': q}
    for lab, (vname, vtext) in zip(labels, variants):
        row[f'response_{lab}'] = vtext
        for dim in ('faithfulness', 'relevance', 'engagement'):
            row[f'{lab}_{dim}'] = ''  # annotator fills 0 / 0.5 / 1
    rows.append(row)
    key[case_id] = {
        'mapping': {lab: vname for lab, (vname, _) in zip(labels, variants)},
        'opus': {
            'oracle_dims': rec.get('oracle_dims'),
            'nomem_dims': rec.get('nomem_dims'),
            'random_dims': rec.get('random_dims'),
        },
    }

with open(f'{OUT}/silent_grounding_annotation.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)
json.dump(key, open(f'{OUT}/silent_grounding_key.json', 'w'), ensure_ascii=False, indent=1)
print(f'silent grounding: {len(rows)} cases written ({missing} missing joins)')

# ---------------- 2. CoT vs Oracle ----------------
E3_FILES = {
    'AnchorMem': 'anchormem_implicit_all.json',
    'A-mem': 'amem_implicit_all.json',
    'BM25': 'bm25_implicit_all.json',
    'Dense': 'dense_implicit_all.json',
    'mem0': 'mem0_implicit_all.json',
}
COT_DIRS = {
    'AnchorMem': 'cot_select_full',
    'A-mem': 'cot_select_bm25_amem',
    'BM25': 'cot_select_bm25_amem',
    'Dense': 'cot_select_dense_mem0',
    'mem0': 'cot_select_dense_mem0',
}

cot_resp = defaultdict(dict)
for system, d in COT_DIRS.items():
    for r in json.load(open(f'{RE}/{d}/responses.json')):
        if r.get('variant') == 'cot_select' and r.get('style') == 'implicit' and r.get('system') == system:
            # prefer parsed response_only (without thinking tags) for annotators
            cot_resp[system][(r['sample_idx'], r['q_idx'])] = r.get('response_only') or r.get('response')

rows2, key2 = [], {}
missing2 = 0
for system, fname in E3_FILES.items():
    for rec in json.load(open(f'{RE}/e3/{fname}')):
        k = (rec['sample_idx'], rec['q_idx'])
        q = implicit_query(*k)
        rc = cot_resp[system].get(k)
        ro = oracle_resp.get(k)
        if not (q and rc and ro):
            missing2 += 1
            continue
        pair = [('cot', rc), ('oracle', ro)]
        rng.shuffle(pair)
        case_id = f"cvo_{system}_{k[0]}_{k[1]}"
        row = {'case_id': case_id, 'query': q,
               'response_A': pair[0][1], 'response_B': pair[1][1]}
        for lab in ('A', 'B'):
            for dim in ('faithfulness', 'relevance', 'engagement'):
                row[f'{lab}_{dim}'] = ''
        rows2.append(row)
        key2[case_id] = {
            'system': system,
            'mapping': {'A': pair[0][0], 'B': pair[1][0]},
            'opus': {'cot_dims': rec.get('cot_dims'), 'oracle_dims': rec.get('oracle_dims'),
                     'winner': rec.get('winner')},
        }

with open(f'{OUT}/cot_vs_oracle_annotation.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows2[0].keys()))
    w.writeheader(); w.writerows(rows2)
json.dump(key2, open(f'{OUT}/cot_vs_oracle_key.json', 'w'), ensure_ascii=False, indent=1)
print(f'cot vs oracle: {len(rows2)} rows written ({missing2} missing joins)')
print(f'output dir: {OUT}')
