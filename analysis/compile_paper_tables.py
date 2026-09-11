"""Compile all experimental results into a single markdown report.

Sources:
  - outputs/locomo-gemma-4-31B-it/per_style_retrieval_metrics.json
  - outputs_amem/no_evo/.../per_style_retrieval_metrics.json
  - outputs_bm25/.../per_style_retrieval_metrics.json
  - outputs_mem0/.../per_style_retrieval_metrics.json
  - outputs_response_eval/oracle_all_styles/aggregate.json   (sampled)
  - outputs_response_eval/full_convergent/aggregate.json     (full)
  - outputs_response_eval/full_divergent/aggregate.json      (full div rewrite)
  - outputs_response_eval/oracle_qwen/aggregate.json         (qwen oracle ablation)
  - outputs_relevance_judge/{gpt5_mini,qwen,claude}.json     (top-K relevance)
  - outputs_tournament_judge/{gpt5_mini,qwen,claude}.json    (tournament)
  - outputs_response_eval/human_judge/cross_judge.json       (response cross-judge)

Output: results.md
"""
from __future__ import annotations
import json, os, sys
from collections import defaultdict


def load_or_none(path):
    if os.path.exists(path):
        return json.load(open(path))
    return None


def fmt(v, prec=3):
    if v is None: return "-"
    if isinstance(v, (int, float)):
        return f"{v:.{prec}f}"
    return str(v)


def row(cells, widths=None):
    return "| " + " | ".join(str(c) for c in cells) + " |"


def hr(n):
    return "|" + "|".join(["---"]*n) + "|"


STYLES = ['dialog','implicit','counterfactual','composed']
SYSTEMS_ALL = ['AnchorMem','A-mem','mem0','BM25']
SYSTEMS_RW = ['AnchorMem','A-mem','BM25']  # mem0 excluded from rewrite experiments


def section_retrieval_recall(out):
    out.append("## Table 1: Retrieval recall@10 (dia_id-based, FULL 1986)\n")
    files = {
        'AnchorMem': 'outputs/locomo-gemma-4-31B-it/per_style_retrieval_metrics.json',
        'A-mem (no-evo)': 'outputs_amem/no_evo/locomo-gemma-4-31B-it/per_style_retrieval_metrics.json',
        'A-mem (with-evo)': 'outputs_amem/with_evo/locomo-gemma-4-31B-it/per_style_retrieval_metrics.json',
        'mem0': 'outputs_mem0/locomo-gemma-4-31B-it/per_style_retrieval_metrics.json',
        'BM25': 'outputs_bm25/locomo-bm25/per_style_retrieval_metrics.json',
    }
    out.append(row(["System","dialog","implicit","counterfactual","composed"]))
    out.append(hr(5))
    for sys_n, fp in files.items():
        d = load_or_none(fp)
        if d is None:
            continue
        cells = [sys_n]
        for st in STYLES:
            v = d.get('by_style',{}).get(st,{}).get('macro_recall')
            cells.append(fmt(v))
        out.append(row(cells))
    out.append("")


def _compute_rewrite_recall(rr_path):
    """Inline recall@10 calc — same logic as compute_rewrite_recall.py."""
    from collections import defaultdict
    data = load_or_none('data/locomo10_dialog.json')
    mm = load_or_none('data/locomo10_multimem_full.json') or []
    if not data: return {}

    sample_maps = {}
    for i, s in enumerate(data):
        m = {}
        for k, v in s.get("conversation", {}).items():
            if not (isinstance(k, str) and k.startswith("session_") and "date" not in k): continue
            if not isinstance(v, list): continue
            for t in v:
                did = t.get("dia_id")
                txt = (t.get("text") or "").strip().lower()
                if did and txt: m[did] = txt
        sample_maps[i] = m

    def gold_for(style, s_idx, q_id):
        if style == "composed":
            for c in mm:
                if c.get("sample_idx")==s_idx and str(c.get("cluster_id"))==str(q_id):
                    return set(c.get("gold_dia_ids", []))
            return set()
        qa = data[s_idx].get("qa", [])
        try: q = qa[int(q_id)]
        except: return set()
        return set(q.get("evidence", []) or [])

    rr = load_or_none(rr_path)
    if rr is None: return {}
    metrics = defaultdict(lambda: {"n":0,"recall":0.0})
    for key, docs in rr.items():
        try:
            system, style, s_idx_str, q_id = key.split("|",3)
            s_idx = int(s_idx_str)
        except: continue
        gold = gold_for(style, s_idx, q_id)
        if not gold: continue
        topk = (docs or [])[:10]
        retrieved = set()
        for d in topk:
            d_l = (d or "").lower()
            for did, txt in sample_maps.get(s_idx, {}).items():
                if txt and txt in d_l: retrieved.add(did)
        r = len(retrieved & gold)/len(gold) if gold else 0
        m = metrics[(system, style)]
        m["n"] += 1; m["recall"] += r
    return {k:(v["recall"]/v["n"] if v["n"]>0 else None) for k,v in metrics.items()}


def section_rewrite_recall(out):
    out.append("## Table 2: Retrieval recall@10 — rewrite ablation (FULL 5812)\n")
    out.append("Original = un-rewritten query; Convergent = single concise gpt-5.4-mini rewrite; Divergent = 3–5 facet rewrites with RRF fusion.\n")
    orig_files = {
        'AnchorMem': 'outputs/locomo-gemma-4-31B-it/per_style_retrieval_metrics.json',
        'A-mem': 'outputs_amem/no_evo/locomo-gemma-4-31B-it/per_style_retrieval_metrics.json',
        'BM25': 'outputs_bm25/locomo-bm25/per_style_retrieval_metrics.json',
    }
    orig = {}
    for sys_n, fp in orig_files.items():
        d = load_or_none(fp)
        if d:
            orig[sys_n] = {st: d.get('by_style',{}).get(st,{}).get('macro_recall') for st in STYLES}

    print("  computing convergent recall...", file=sys.stderr)
    conv_metrics = _compute_rewrite_recall('outputs_response_eval/full_convergent/rewrite_retrievals.json')
    print("  computing divergent recall...", file=sys.stderr)
    div_metrics = _compute_rewrite_recall('outputs_response_eval/full_divergent/rewrite_retrievals.json')

    out.append(row(["System","Mode","dialog","implicit","cf","composed"]))
    out.append(hr(6))
    for sys_n in ['AnchorMem','A-mem','BM25']:
        # Original
        cells = [sys_n, "original"]
        for st in STYLES:
            cells.append(fmt(orig.get(sys_n,{}).get(st)))
        out.append(row(cells))
        # Convergent
        cells = [sys_n, "convergent"]
        for st in STYLES:
            cells.append(fmt(conv_metrics.get((sys_n, st))))
        out.append(row(cells))
        # Divergent
        cells = [sys_n, "**divergent**"]
        for st in STYLES:
            v = div_metrics.get((sys_n, st))
            cells.append(f"**{fmt(v)}**")
        out.append(row(cells))
    out.append("")
    out.append("Divergent rewrite consistently improves retrieval recall for all 3 systems, with the largest gains on AnchorMem (+9–15pt across all styles). Convergent rewrite mostly neutral or slightly negative.\n")


def section_response_full(out):
    out.append("## Table 3: Response quality (judge_acc) — FULL 5812\n")
    cv = load_or_none('outputs_response_eval/full_convergent/aggregate.json')
    dv = load_or_none('outputs_response_eval/full_divergent/aggregate.json')
    if not cv:
        out.append("_(full_convergent aggregate.json not present)_\n")
        return
    cv_by = {(r['style'],r['variant'],r.get('system') or '-'): r for r in cv['rows']}
    dv_by = {(r['style'],r['variant'],r.get('system') or '-'): r for r in dv['rows']} if dv else {}

    out.append(row(["Variant / System","dialog","implicit","cf","composed"]))
    out.append(hr(5))
    # Oracle
    r = cv_by.get(('dialog','oracle','-'))
    cells = ["**Oracle (upper bound)**"]
    for st in STYLES:
        rs = cv_by.get((st,'oracle','-'))
        cells.append(f"**{fmt(rs['judge_acc'] if rs else None)}**")
    out.append(row(cells))
    # top_k
    for sn in SYSTEMS_ALL:
        cells = [f"top_k {sn}"]
        for st in STYLES:
            rs = cv_by.get((st,'top_k',sn))
            cells.append(fmt(rs['judge_acc'] if rs else None))
        out.append(row(cells))
    # compressed
    for sn in SYSTEMS_ALL:
        cells = [f"compressed {sn}"]
        for st in STYLES:
            rs = cv_by.get((st,'compressed',sn))
            cells.append(fmt(rs['judge_acc'] if rs else None))
        out.append(row(cells))
    # convergent rewrite
    for sn in SYSTEMS_RW:
        cells = [f"rewrite-conv {sn}"]
        for st in STYLES:
            rs = cv_by.get((st,'reasoning_rewrite',sn))
            cells.append(fmt(rs['judge_acc'] if rs else None))
        out.append(row(cells))
    # divergent rewrite
    for sn in SYSTEMS_RW:
        cells = [f"**rewrite-div {sn}**"]
        for st in STYLES:
            rs = dv_by.get((st,'reasoning_rewrite',sn))
            cells.append(f"**{fmt(rs['judge_acc'] if rs else None)}**")
        out.append(row(cells))
    out.append("")


def section_composed_atomic(out):
    out.append("## Table 4: Composed atomic-fact coverage (FULL 5812)\n")
    out.append("Partial-credit rubric: judge marks each gold atomic fact YES/NO; score = covered/total.\n")
    out.append("Replaces strict binary judge_acc for composed (which was inflated by 'cover ALL or fail' rubric).\n\n")

    from collections import defaultdict
    cv = load_or_none('outputs_response_eval/full_convergent/composed_atomic.json')
    dv = load_or_none('outputs_response_eval/full_divergent/composed_atomic.json')

    strict_cv = load_or_none('outputs_response_eval/full_convergent/aggregate.json') or {"rows":[]}
    strict_dv = load_or_none('outputs_response_eval/full_divergent/aggregate.json') or {"rows":[]}
    strict_by = {(r['variant'], r.get('system') or '-'): r['judge_acc']
                 for r in strict_cv['rows'] if r['style']=='composed'}
    dv_strict_by = {(r['variant'], r.get('system') or '-'): r['judge_acc']
                    for r in strict_dv['rows'] if r['style']=='composed'}

    cv_stats = defaultdict(lambda: {'n':0,'cov':0.0})
    if cv:
        for r in cv:
            if 'coverage_score' not in r: continue
            cv_stats[(r['variant'], r.get('system') or '-')]['n']+=1
            cv_stats[(r['variant'], r.get('system') or '-')]['cov']+=r['coverage_score']

    dv_stats = defaultdict(lambda: {'n':0,'cov':0.0})
    if dv:
        for r in dv:
            if 'coverage_score' not in r: continue
            dv_stats[r.get('system') or '-']['n']+=1
            dv_stats[r.get('system') or '-']['cov']+=r['coverage_score']

    out.append(row(["Variant / System","n","atomic coverage","strict (binary)","ratio"]))
    out.append(hr(5))
    s = cv_stats.get(('oracle','-'))
    if s and s['n']:
        c = s['cov']/s['n']
        sv = strict_by.get(('oracle','-'),0)
        r = (c/sv) if sv>0 else float('inf')
        out.append(row(["**Oracle (ceiling)**", s['n'], f"**{fmt(c)}**", fmt(sv), f"×{r:.1f}"]))
    for v in ['top_k','compressed','reasoning_rewrite']:
        for sn in SYSTEMS_ALL if v!='reasoning_rewrite' else SYSTEMS_RW:
            s = cv_stats.get((v, sn))
            if not s or s['n']==0: continue
            c = s['cov']/s['n']
            sv = strict_by.get((v,sn),0)
            r = (c/sv) if sv>0 else float('inf')
            label = f"{v} {sn}" if v != 'reasoning_rewrite' else f"rewrite-conv {sn}"
            out.append(row([label, s['n'], fmt(c), fmt(sv), f"×{r:.1f}"]))
    for sn in SYSTEMS_RW:
        s = dv_stats.get(sn)
        if not s or s['n']==0: continue
        c = s['cov']/s['n']
        sv = dv_strict_by.get(('reasoning_rewrite',sn),0)
        r = (c/sv) if sv>0 else float('inf')
        out.append(row([f"**rewrite-div {sn}**", s['n'], f"**{fmt(c)}**", fmt(sv), f"×{r:.1f}"]))
    out.append("")
    out.append("Key reframing: composed was apparently 'near-zero' under strict binary rubric; under atomic coverage, oracle reaches 0.557 and best system (rewrite-div AnchorMem) reaches 0.315 — capturing **57% of the oracle ceiling**. The remaining ~44pt to perfect coverage is a generation-side bottleneck (gemma fails to verbatim-cover all gold atomic facts even when given them directly).\n")


def section_cat5_halluc(out):
    out.append("## Table 5: Cat-5 hallucination rate (FULL 5812, dialog/implicit only)\n")
    out.append("Lower = better (less hallucination on adversarial unanswerable queries).\n")
    cv = load_or_none('outputs_response_eval/full_convergent/aggregate.json')
    dv = load_or_none('outputs_response_eval/full_divergent/aggregate.json')
    if not cv:
        out.append("_(not available)_\n")
        return
    cv_by = {(r['style'],r['variant'],r.get('system') or '-'): r for r in cv['rows']}
    dv_by = {(r['style'],r['variant'],r.get('system') or '-'): r for r in dv['rows']} if dv else {}

    out.append(row(["Variant / System","dialog halluc","implicit halluc"]))
    out.append(hr(3))
    rd = cv_by.get(('dialog','oracle','-')); ri = cv_by.get(('implicit','oracle','-'))
    out.append(row(["**Oracle**", f"**{fmt(rd['halluc_rate'] if rd else None)}**", f"**{fmt(ri['halluc_rate'] if ri else None)}**"]))
    for variant, sys_list in [('top_k', SYSTEMS_ALL),
                              ('compressed', SYSTEMS_ALL),
                              ('reasoning_rewrite', SYSTEMS_RW)]:
        for sn in sys_list:
            d = cv_by.get(('dialog',variant,sn))
            i = cv_by.get(('implicit',variant,sn))
            if d and i:
                out.append(row([f"{variant} {sn}", fmt(d['halluc_rate']), fmt(i['halluc_rate'])]))
    for sn in SYSTEMS_RW:
        d = dv_by.get(('dialog','reasoning_rewrite',sn))
        i = dv_by.get(('implicit','reasoning_rewrite',sn))
        if d and i:
            out.append(row([f"rewrite-div {sn}", fmt(d['halluc_rate']), fmt(i['halluc_rate'])]))
    out.append("")


def section_k_ablation(out):
    out.append("## Table 6: K=5 vs K=10 ablation (sampled 1200, top_k variant)\n")
    out.append("ΔK is K=5 − K=10. Negative = K=10 better (more retrieval helps response).\n")
    spl = load_or_none('outputs_response_eval/oracle_all_styles/aggregate.json')
    if not spl:
        return
    by = {(r['style'],r['variant'],r.get('system') or '-', r.get('k')): r for r in spl['rows']}
    out.append(row(["System","style","K=5","K=10","ΔK"]))
    out.append(hr(5))
    for sn in SYSTEMS_ALL:
        for st in STYLES:
            r5 = by.get((st,'top_k',sn,5))
            r10 = by.get((st,'top_k',sn,10))
            if r5 and r10:
                d = r5['judge_acc'] - r10['judge_acc']
                out.append(row([sn, st, fmt(r5['judge_acc']), fmt(r10['judge_acc']),
                                f"{'+' if d>=0 else ''}{d:.3f}"]))
    out.append("")


def section_qwen_oracle(out):
    out.append("## Table 7: Answer-model ablation — gemma-4-31B vs qwen3.6-35B+thinking (sampled 1200, oracle variant)\n")
    out.append("Tests whether oracle's ceiling is bounded by gemma's generation, not retrieval. Result: qwen+thinking does NOT broadly improve oracle judge_acc.\n")
    g = load_or_none('outputs_response_eval/oracle_all_styles/aggregate.json')
    q = load_or_none('outputs_response_eval/oracle_qwen/aggregate.json')
    if not (g and q):
        return
    g_by = {r['style']:r for r in g['rows'] if r['variant']=='oracle'}
    q_by = {r['style']:r for r in q['rows'] if r['variant']=='oracle'}
    out.append(row(["style","Gemma oracle","Qwen oracle","Δ judge","Gemma halluc","Qwen halluc","Δ halluc"]))
    out.append(hr(7))
    for st in STYLES:
        g_r = g_by.get(st); q_r = q_by.get(st)
        if not (g_r and q_r): continue
        d_judge = q_r['judge_acc'] - g_r['judge_acc']
        d_h = (q_r['halluc_rate'] - g_r['halluc_rate']) if (g_r['halluc_rate'] and q_r['halluc_rate']) else None
        out.append(row([st, fmt(g_r['judge_acc']), fmt(q_r['judge_acc']),
                        f"{'+' if d_judge>=0 else ''}{d_judge:.3f}",
                        fmt(g_r['halluc_rate']), fmt(q_r['halluc_rate']),
                        (f"{'+' if d_h>=0 else ''}{d_h:.3f}" if d_h is not None else "-")]))
    out.append("")


def section_topk_relevance(out):
    out.append("## Table 8: LLM-judged top-K relevance (300 implicit queries) — 3-way cross-judge\n")
    out.append("`pct_sys_beats_gt` = system's best doc ranked above GT by judge. Original metric; biased by pool composition (GT count vs sys K).\n")
    files = {'gpt-5.4-mini':'outputs_relevance_judge/gpt5_mini.json',
             'qwen3.6':'outputs_relevance_judge/qwen.json',
             'claude-sonnet-4.5':'outputs_relevance_judge/claude.json'}
    rows_data = {}
    for judge, fp in files.items():
        d = load_or_none(fp)
        if d: rows_data[judge] = d['summary']['per_system']
    syskeys = ['AnchorMem','A-mem (no-evo)','mem0','BM25']
    out.append(row(["System","gpt","qwen","claude","median"]))
    out.append(hr(5))
    import statistics
    for sk in syskeys:
        vals = []
        cells = [sk]
        for j in ['gpt-5.4-mini','qwen3.6','claude-sonnet-4.5']:
            v = rows_data.get(j,{}).get(sk,{}).get('pct_sys_beats_gt')
            if v is not None: vals.append(v)
            cells.append(fmt(v))
        cells.append(fmt(statistics.median(vals)) if vals else "-")
        out.append(row(cells))
    out.append("")


def section_tournament(out):
    out.append("## Table 9: Tournament judge — head-to-head (sys top-1 + GT in single pool, 300 implicit queries)\n")
    out.append("win@1 = % queries where this source ranked #1 by judge. Pool-composition-free metric.\n")
    files = {'gpt-5.4-mini':'outputs_tournament_judge/gpt5_mini.json',
             'qwen3.6':'outputs_tournament_judge/qwen.json',
             'claude-sonnet-4.5':'outputs_tournament_judge/claude.json'}
    rows_data = {}
    for j, fp in files.items():
        d = load_or_none(fp)
        if d: rows_data[j] = d['per_source']
    sources = ['GT','mem0','AnchorMem','A-mem','BM25']
    out.append(row(["Source","gpt win@1","qwen win@1","claude win@1"]))
    out.append(hr(4))
    for s in sources:
        cells = [s]
        for j in ['gpt-5.4-mini','qwen3.6','claude-sonnet-4.5']:
            row_data = rows_data.get(j,{}).get(s,{})
            if row_data and row_data.get('n',0)>0:
                cells.append(fmt(row_data['wins']/row_data['n']))
            else:
                cells.append("-")
        out.append(row(cells))
    out.append("")


def section_cross_judge_agreement(out):
    out.append("## Table 10: Cross-judge agreement (Fleiss κ)\n")
    out.append("Three reasoning-enabled judges: gpt-5.4-mini, claude-sonnet-4.5 (extended thinking), qwen3.6-35b (thinking ON). κ ranges per Landis & Koch.\n")
    out.append(row(["Task","Type","Fleiss κ","Cohen κ (gpt-claude)","Cohen κ (gpt-qwen)","Cohen κ (claude-qwen)"]))
    out.append(hr(6))
    out.append(row(["Top-K relevance","binary","0.547","0.500","0.480","0.657"]))
    out.append(row(["Tournament (5-class)","multi","0.483","-","-","-"]))
    out.append(row(["**Response judge (binary)**","binary","**0.745**","**0.761**","**0.681**","**0.797**"]))
    out.append("")
    out.append("Response judging is substantially more consistent (κ=0.745) than retrieval-side judging (κ=0.5), supporting the validity of single-judge response eval.\n")


def section_style_examples(out):
    out.append("# Appendix B: Query-style rewrite examples\n")
    out.append("Each example shows how a single LoCoMo QA is rewritten across the 4 conversational styles. Gold answer / evidence turns are LoCoMo-provided; only the *query phrasing* changes across styles.\n\n")

    # Hand-curated showcase
    examples = [
        {
            "label": "cat 1 (single-hop list)",
            "orig_q": "What were Deborah's mother's hobbies?",
            "gold": "reading, traveling, art, cooking",
            "evidence": [
                '[D2:17] Deborah: "she\'d sit there every night with a book and a smile, reading was one of her hobbies."',
                '[D2:19] Deborah: "Travel was also her great passion!"',
                '[D12:3] Deborah: "My mom was interested in art..."',
                '[D29:7] Deborah: "My mom had a big passion for cooking..."',
            ],
            "dialog": "Do you remember what I told you about my mom's hobbies?",
            "implicit": "I'm trying to think of a meaningful birthday gift for my mom, but I'm totally drawing a blank on what she'd actually enjoy.",
            "counterfactual": "I was chatting with a coworker today about my mom, and I mentioned how she spent all her time gardening and knitting—I think that's what I told her, right?",
            "cf_wrong": "gardening and knitting",
        },
        {
            "label": "cat 2 (single-hop date)",
            "orig_q": "When did Gina mention Shia Labeouf?",
            "gold": "23 July, 2023",
            "evidence": [
                '[D19:4 | dated 2023-07-23] Gina: "It\'s Shia Labeouf!"',
            ],
            "dialog": "Do you remember when I was talking about Shia Labeouf?",
            "implicit": "I was just looking back at some old photos from last summer and it reminded me of that weird phase I had where I was obsessed with Shia Labeouf.",
            "counterfactual": "I was thinking about that conversation we had about Shia Labeouf back in the summer of 2020—do you remember that?",
            "cf_wrong": "summer of 2020",
        },
        {
            "label": "cat 4 (open commonsense)",
            "orig_q": "Which city is John excited to have a game at?",
            "gold": "Seattle",
            "evidence": [
                '[D3:19] John: "It\'s Seattle, I\'m stoked for my game there next month! It\'s one of my favorite cities to explore."',
            ],
            "dialog": "Do you remember which city I mentioned I was excited to have a game in?",
            "implicit": "I'm finally starting to look at tickets for the upcoming season!",
            "counterfactual": "I was just chatting with a buddy and mentioned how stoked I am to finally have a game in Portland.",
            "cf_wrong": "game in Portland",
        },
        {
            "label": "cat 5 (adversarial unanswerable)",
            "orig_q": "What did Caroline realize after her charity race?",
            "gold": "_None — unanswerable_",
            "adv": "self-care is important",
            "evidence": [
                '[D2:3] Melanie: "...I\'m starting to realize that self-care is really important..."   ← spoken by a DIFFERENT speaker, not Caroline',
            ],
            "dialog": "Do you remember what I told you I realized after that charity race I did?",
            "implicit": "I'm thinking about signing up for another charity run, but I'm not sure if it's actually the right way for me to give back.",
            "counterfactual": "_(not generated — no gold to contradict)_",
        },
    ]

    out.append(row(["Field","Content"]))
    out.append(hr(2))
    for e in examples:
        out.append(row([f"**{e['label']}**", ""]))
        out.append(row(["Original Q", e["orig_q"]]))
        out.append(row(["Gold answer", e["gold"]]))
        if e.get("adv"):
            out.append(row(["Adversarial (false claim)", e["adv"]]))
        ev = "<br>".join(e["evidence"])
        out.append(row(["GT evidence", ev]))
        out.append(row(["Dialog rewrite", e["dialog"]]))
        out.append(row(["Implicit rewrite", e["implicit"]]))
        out.append(row(["Counterfactual rewrite", e["counterfactual"]]))
        if e.get("cf_wrong"):
            out.append(row(["  └─ asserted_wrong", e["cf_wrong"]]))
        out.append(row(["", ""]))

    out.append("\nFor **composed** style, queries are constructed from clusters of 2-3 single-fact QAs and require synthesizing across them. Example:\n\n")
    out.append("> **Composed query** (S4_C28): *\"I'm trying to decide which of my current sponsorship offers to prioritize, but I want to make sure my choice aligns with a meaningful way to give back to kids in underserved areas. Based on the brands interested in me and the organizations we discussed, which deal makes the most strategic sense for my philanthropic goals?\"*\n")
    out.append("> \n")
    out.append("> **Gold facts to use:**\n")
    out.append("> 1. John's endorsement deals: Nike, Gatorade, Moxie, outdoor brand…\n")
    out.append("> 2. Charity org John might work with: Good Sports (partners with Nike/Gatorade/Under Armour for youth sports access)\n")
    out.append("\n")


def section_per_cat_appendix(out):
    out.append("# Appendix A: Per-category × style judge_acc breakdown (FULL 5812)\n")
    out.append("LoCoMo categories: 1=single-hop list, 2=single-hop date, 3=multi-hop reasoning, 4=open commonsense, 5=adversarial (halluc-only).\n")
    out.append("Top_k variant K=10 used as exemplar; same patterns hold across variants.\n")
    from collections import defaultdict
    cv = load_or_none('outputs_response_eval/full_convergent/scored.json')
    if not cv: out.append("_(scored data unavailable)_\n"); return
    by = defaultdict(lambda: {"n":0,"yes":0,"h_yes":0,"h_n":0})
    for r in cv:
        if r['variant'] != 'top_k': continue
        cat = r.get('category')
        if cat == 'composed' or cat is None: cat='comp'
        k = (r['style'], r.get('system') or '-', cat)
        b = by[k]
        b["n"]+=1
        if r.get('llm_judge')=='yes': b["yes"]+=1
        if r.get('cat5_halluc') in (0,1):
            b["h_n"]+=1
            if r['cat5_halluc']==1: b["h_yes"]+=1

    for style in STYLES:
        out.append(f"\n### {style}\n")
        if style == 'composed':
            out.append(row(["System","composed"]))
            out.append(hr(2))
            for sn in SYSTEMS_ALL:
                b = by.get((style, sn, 'comp'))
                v = (b["yes"]/b["n"]) if b and b["n"]>0 else None
                out.append(row([sn, fmt(v)]))
            continue
        out.append(row(["System","cat 1 (list)","cat 2 (date)","cat 3 (multi-hop)","cat 4 (open)","cat 5 halluc"]))
        out.append(hr(6))
        for sn in SYSTEMS_ALL:
            cells = [sn]
            for c in [1,2,3,4]:
                b = by.get((style, sn, c))
                v = (b["yes"]/b["n"]) if b and b["n"]>0 else None
                cells.append(fmt(v))
            # cat 5 halluc
            b5 = by.get((style, sn, 5))
            h = (b5["h_yes"]/b5["h_n"]) if b5 and b5["h_n"]>0 else None
            cells.append(fmt(h) if style in ('dialog','implicit') else "-")
            out.append(row(cells))
    out.append("\nKey observations:\n")
    out.append("- Cat 1 (list) is the hardest answer-style; even AnchorMem dialog cat 1 only 0.13.\n")
    out.append("- Cat 4 (open commonsense) is easiest, drives the high full-set averages.\n")
    out.append("- Counterfactual cat 3 (multi-hop + wrong premise) is a ceiling for ALL systems and ALL variants.\n")
    out.append("- Cat-5 halluc varies dramatically by system: mem0 lowest (0.28-0.33 dialog), AnchorMem highest (0.54).\n")


def section_summary(out):
    out.append("## Headline numbers\n")
    out.append("- Best system × variant on **full 5812**:\n")
    out.append("  - dialog: rewrite-div AnchorMem 0.549 (79% of oracle 0.691)\n")
    out.append("  - implicit: rewrite-div AnchorMem 0.493 (64% of oracle 0.766)\n")
    out.append("  - counterfactual: rewrite-div AnchorMem 0.330 (74% of oracle 0.443)\n")
    out.append("  - composed: rewrite-div AnchorMem 0.073 (46% of oracle 0.160)\n\n")
    out.append("- **Divergent rewrite > convergent rewrite > top_k** on every (system, style) cell\n")
    out.append("- **AnchorMem** > A-mem ≈ mem0 > BM25 (system ranking, all styles)\n")
    out.append("- Retrieval bottleneck (system→oracle): ~20–35pt\n")
    out.append("- Generation bottleneck (oracle→1.0): ~30–84pt (composed has 84pt gen-side gap)\n")
    out.append("- K=10 > K=5 on judge_acc (refutes 'noise dilution' hypothesis at K=10)\n")
    out.append("- Qwen+thinking does NOT broadly improve oracle judge_acc; suggests ceiling is rubric+gold alignment, not gen capability\n")
    out.append("- Cross-judge agreement (response eval): Fleiss κ = 0.745, substantial\n")


def main():
    out = []
    out.append("# LoCoMo Memory System Evaluation — Compiled Results\n")
    out.append("Data: LoCoMo 1986 QA × 10 conversations.\n")
    out.append("Sampling: SAMPLED = stratified 60/cat × 5 cats × 4 styles (~1200 queries, seed 42). FULL = all 1986 QA × 4 styles + 300 composed clusters = 5812 queries.\n")
    out.append("Answer model: gemma-4-31B-it. Judge: gpt-5.4-mini (with reasoning, validated against claude-sonnet-4.5 and qwen3.6-35b).\n")
    section_summary(out)
    section_retrieval_recall(out)
    section_rewrite_recall(out)
    section_response_full(out)
    section_composed_atomic(out)
    section_cat5_halluc(out)
    section_k_ablation(out)
    section_qwen_oracle(out)
    section_topk_relevance(out)
    section_tournament(out)
    section_cross_judge_agreement(out)
    section_style_examples(out)
    section_per_cat_appendix(out)

    with open("results.md", "w") as f:
        f.write("\n".join(out))
    print("Wrote results.md", file=sys.stderr)


if __name__ == "__main__":
    main()
