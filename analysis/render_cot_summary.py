"""Aggregate all CoT + E3 results across 4 systems into a paper-ready markdown summary."""
from __future__ import annotations
import json, os, sys
from collections import Counter, defaultdict

ROOT = "outputs_response_eval"

# Mapping: system -> CoT response dir
COT_DIRS = {
    "AnchorMem": "cot_select_anchormem",
    "Dense":     "cot_select_dense_mem0",
    "mem0":      "cot_select_dense_mem0",
    "BM25":      "cot_select_bm25_amem",
    "A-mem":     "cot_select_bm25_amem",
}
E3_LOWER = {"AnchorMem":"anchormem", "Dense":"dense", "mem0":"mem0", "BM25":"bm25", "A-mem":"amem"}

# Load existing top_k baselines
fp_conv = json.load(open(f"{ROOT}/fact_used_partial/convergent.json"))
cf_conv = json.load(open(f"{ROOT}/cf_3way/convergent.json"))
ca_full = json.load(open(f"{ROOT}/full_convergent/composed_atomic.json"))

def topk_mean(records, sys_name, style, score_field, valid_values):
    s, n = 0.0, 0
    for r in records:
        if r.get("system") != sys_name or r.get("style") != style: continue
        if r.get("variant") != "top_k": continue
        v = r.get(score_field)
        if v not in valid_values: continue
        s += float(v); n += 1
    return s/n if n else None

def cot_mean(sys_name, style, score_field, valid_values, score_file):
    p = f"{ROOT}/{COT_DIRS[sys_name]}/{score_file}"
    if not os.path.exists(p): return None
    recs = json.load(open(p))
    s, n = 0.0, 0
    for r in recs:
        if r.get("system") != sys_name or r.get("style") != style: continue
        v = r.get(score_field)
        if v not in valid_values: continue
        s += float(v); n += 1
    return s/n if n else None

def cot_atomic_mean(sys_name):
    p = f"{ROOT}/{COT_DIRS[sys_name]}/composed_atomic.json"
    if not os.path.exists(p): return None
    recs = json.load(open(p))
    s, n = 0.0, 0
    for r in recs:
        if r.get("system") != sys_name: continue
        if "coverage_score" not in r: continue
        s += r["coverage_score"]; n += 1
    return s/n if n else None

def topk_cf_mean(sys_name):
    s, n = 0.0, 0
    for r in cf_conv:
        if r.get("system") != sys_name: continue
        if r.get("variant") != "top_k": continue
        if r.get("cf_label") not in ("A","B","C"): continue
        s += r["cf_score"]; n += 1
    return s/n if n else None

def cot_cf_mean(sys_name):
    p = f"{ROOT}/{COT_DIRS[sys_name]}/cf_3way.json"
    if not os.path.exists(p): return None
    recs = json.load(open(p))
    s, n = 0.0, 0
    for r in recs:
        if r.get("system") != sys_name: continue
        if r.get("cf_label") not in ("A","B","C"): continue
        s += r["cf_score"]; n += 1
    return s/n if n else None

def topk_atomic_mean(sys_name):
    s, n = 0.0, 0
    for r in ca_full:
        if r.get("system") != sys_name: continue
        if r.get("variant") != "top_k": continue
        if "coverage_score" not in r: continue
        s += r["coverage_score"]; n += 1
    return s/n if n else None

systems = ["BM25","Dense","A-mem","AnchorMem","mem0"]
print("# CoT-selection full results\n")
print("## Per-system: top-K vs CoT (response quality metrics)\n")
print("| System | Metric | top-K | CoT | Δ |")
print("|---|---|---|---|---|")
for sys_name in systems:
    for style in ("dialog","implicit"):
        tk = topk_mean(fp_conv, sys_name, style, "fact_score", (0,0.5,1))
        co = cot_mean(sys_name, style, "fact_score", (0,0.5,1), "fact_used_partial.json")
        if tk is None or co is None: continue
        print(f"| {sys_name} | {style} partial | {tk:.3f} | {co:.3f} | {co-tk:+.3f} |")
    tk = topk_cf_mean(sys_name); co = cot_cf_mean(sys_name)
    if tk is not None and co is not None:
        print(f"| {sys_name} | counterfactual | {tk:.3f} | {co:.3f} | {co-tk:+.3f} |")
    tk = topk_atomic_mean(sys_name); co = cot_atomic_mean(sys_name)
    if tk is not None and co is not None:
        print(f"| {sys_name} | composed atomic | {tk:.3f} | {co:.3f} | {co-tk:+.3f} |")

# E3 win rates
print("\n## E3 pairwise: CoT response vs Oracle response (implicit)\n")
print("### (both fact_used = 0.0) — silent grounding cell\n")
print("| System | n | CoT win | Oracle win | Tie |")
print("|---|---|---|---|---|")
for sys_name in systems:
    p_new = f"{ROOT}/e3/{E3_LOWER[sys_name]}_implicit_both0.json"
    p_old = f"{ROOT}/e3/{E3_LOWER[sys_name]}_implicit.json"
    p = p_new if os.path.exists(p_new) else p_old
    if not os.path.exists(p): continue
    d = json.load(open(p))
    c = Counter(r.get("winner") for r in d if "winner" in r)
    n = c["cot"]+c["oracle"]+c["tie"]
    if n==0: continue
    print(f"| {sys_name} | {n} | {c['cot']} ({c['cot']/n:.1%}) | {c['oracle']} ({c['oracle']/n:.1%}) | {c['tie']} ({c['tie']/n:.1%}) |")

print("\n### (both fact_used = 1.0) — framing cell\n")
print("| System | n | CoT win | Oracle win | Tie |")
print("|---|---|---|---|---|")
for sys_name in systems:
    p = f"{ROOT}/e3/{E3_LOWER[sys_name]}_implicit_both1.json"
    if not os.path.exists(p): continue
    d = json.load(open(p))
    c = Counter(r.get("winner") for r in d if "winner" in r)
    n = c["cot"]+c["oracle"]+c["tie"]
    if n==0: continue
    print(f"| {sys_name} | {n} | {c['cot']} ({c['cot']/n:.1%}) | {c['oracle']} ({c['oracle']/n:.1%}) | {c['tie']} ({c['tie']/n:.1%}) |")

# Per-dim averages in (both=0)
print("\n### Per-dim score averages (both fact_used=0)\n")
print("| System | metric | CoT | Oracle | Δ |")
print("|---|---|---|---|---|")
for sys_name in systems:
    p_new = f"{ROOT}/e3/{E3_LOWER[sys_name]}_implicit_both0.json"
    p_old = f"{ROOT}/e3/{E3_LOWER[sys_name]}_implicit.json"
    p = p_new if os.path.exists(p_new) else p_old
    if not os.path.exists(p): continue
    d = json.load(open(p))
    dims = ("faithfulness","relevance","engagement")
    cot_d = {k:[] for k in dims}; or_d = {k:[] for k in dims}
    for r in d:
        if "cot_dims" not in r: continue
        for k in dims:
            cot_d[k].append(r["cot_dims"][k])
            or_d[k].append(r["oracle_dims"][k])
    for k in dims:
        if not cot_d[k]: continue
        cm = sum(cot_d[k])/len(cot_d[k]); om = sum(or_d[k])/len(or_d[k])
        print(f"| {sys_name} | {k} | {cm:.3f} | {om:.3f} | {cm-om:+.3f} |")
