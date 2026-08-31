"""E3 evaluation: score CoT-selection response vs Oracle response separately,
then compare scores for win/lose/tie.

For each (sample, q) where both CoT and Oracle have partial fact_used = 0.0
(both miss the gold fact), score each response on a 0-3 binary checklist:
  - faithfulness (uses any memory fact correctly)
  - relevance (addresses the user's situation)
  - engagement (engages with emotional/situational framing)
Each response is scored given its OWN memory context (no shared evidence
block, avoiding the bias that any one memory selection is "ground truth").

Output per (sample, q):
  { style, sample_idx, q_idx, cot_score:0-3, oracle_score:0-3, winner }
"""
from __future__ import annotations
import argparse, json, os, sys, threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List

import httpx
from anthropic import Anthropic
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_response_eval import all_speaker_turns, format_turns_block, SYSTEM_CACHE_DIR


SCORE_PROMPT = """You are evaluating how appropriately an AI assistant responded to a user, given the memory items it had access to.

User (named {user_name}) says:
{query}

Memory items the assistant could draw on (from prior conversations):
{memory_block}

Assistant's response:
{response}

Score the response on three independent criteria. For each, output one of: 1, 0.5, or 0.

(1) faithfulness — whether the response is well-grounded in the memory items above:
  1   = clearly draws on a specific memory item (paraphrasing is fine)
  0.5 = consistent with memory but does not actively use any specific item (neutral coexistence)
  0   = contradicts memory OR fabricates plausible-sounding specifics not present in the memory

(2) relevance — whether the response addresses what the user is asking about or describing:
  1   = directly addresses the user's question / situation
  0.5 = partially addresses; some of the response is on-topic and some is generic
  0   = off-topic / pure boilerplate / redirects to an unrelated subject

(3) engagement — how the response engages with the user's emotional / situational framing:
  1   = acknowledges the user's state AND offers something concrete (a fitting follow-up, an actionable suggestion, or genuine empathy)
  0.5 = polite, functional acknowledgment — neutral and on-topic but does not go beyond a generic "I see / I'm sorry to hear that / could you tell me more"
  0   = cold refusal, dismissive, pure list, or ignores the user's emotional/situational framing entirely

Output STRICT JSON ONLY:
{{
  "faithfulness": 1 | 0.5 | 0,
  "relevance":    1 | 0.5 | 0,
  "engagement":   1 | 0.5 | 0
}}
No preamble, no markdown."""


def _anthropic_client():
    kp = os.path.expanduser("~/.anthropic-key")
    key = open(kp).readline().strip() if os.path.exists(kp) else os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        raise RuntimeError("Anthropic API key missing")
    return Anthropic(api_key=key)


def _parse_score(raw: str):
    import re
    if not raw: return None
    # Strip code fences if present
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    body = m.group(1) if m else raw
    # Try to locate JSON object
    m = re.search(r"\{[^{}]*\}", body, re.DOTALL)
    if not m: return None
    try:
        d = json.loads(m.group(0))
    except Exception:
        return None
    keys = ("faithfulness", "relevance", "engagement")
    if not all(k in d for k in keys): return None
    total = 0.0
    norm = {}
    for k in keys:
        v = d[k]
        if isinstance(v, bool): v = int(v)
        # accept int, float, or numeric-string
        if isinstance(v, str):
            try: v = float(v)
            except ValueError: return None
        if not isinstance(v, (int, float)): return None
        if v not in (0, 0.5, 1, 0.0, 1.0): return None
        total += float(v)
        norm[k] = float(v)
    return total, norm


def score_one(client, model: str, query: str, memory_block: str, user_name: str, response: str):
    prompt = SCORE_PROMPT.format(
        user_name=user_name or "the user",
        query=query,
        memory_block=memory_block or "(no memory items)",
        response=response,
    )
    kw = {"model": model, "max_tokens": 200,
          "messages": [{"role": "user", "content": prompt}]}
    if not model.startswith("claude-opus-4-7"):
        kw["temperature"] = 0.0
    resp = client.messages.create(**kw)
    text = "".join(b.text for b in resp.content if hasattr(b, "text"))
    parsed = _parse_score(text)
    if parsed is None:
        return None, None, text
    total, dims = parsed
    return total, dims, text


def load_topk_docs(system: str, sample_idx: int, style: str):
    """Return top-K docs for (system, sample, style) using existing retrieval cache."""
    style_to_field = {
        "dialog": "dialog_query",
        "implicit": "implicit_query",
        "counterfactual": "counterfactual_query",
    }
    field = style_to_field.get(style)
    if not field:
        return []
    base = SYSTEM_CACHE_DIR[system]
    cache = os.path.join(base, f"sample_{sample_idx}", f"queries_solutions_{field}.json")
    if not os.path.exists(cache):
        return []
    sols = json.load(open(cache))
    return sols


def docs_for_query(system: str, sample_idx: int, style: str, q_idx: int, k: int = 10):
    sols = load_topk_docs(system, sample_idx, style)
    if not sols:
        return []
    for entry in sols:
        if entry.get("q_idx") == q_idx or entry.get("q_index") == q_idx:
            return (entry.get("docs") or [])[:k]
    # Fallback: positional
    if q_idx < len(sols):
        return (sols[q_idx].get("docs") or [])[:k]
    return []


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cot_responses_path", required=True,
                   help="responses.json containing cot_select responses (with cited_ids)")
    p.add_argument("--oracle_responses_path", required=True,
                   help="responses.json containing oracle responses")
    p.add_argument("--cot_fact_used_path", required=True,
                   help="partial fact_used scored file for cot_select")
    p.add_argument("--oracle_fact_used_path", required=True,
                   help="partial fact_used scored file for oracle")
    p.add_argument("--system", required=True,
                   help="memory system (e.g. AnchorMem)")
    p.add_argument("--styles", default="implicit")
    p.add_argument("--sample_path", default="data/locomo10_dialog.json")
    p.add_argument("--output", required=True)
    p.add_argument("--judge_model", default="claude-opus-4-7")
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--checkpoint_every", type=int, default=50)
    p.add_argument("--cot_score", type=float, default=0.0,
                   help="filter cot fact_score equal to this value (default 0.0)")
    p.add_argument("--oracle_score", type=float, default=0.0,
                   help="filter oracle fact_score equal to this value (default 0.0)")
    p.add_argument("--any_score", action="store_true",
                   help="if set, do not filter by fact_score — use all cases")
    args = p.parse_args()

    styles = set(args.styles.split(","))

    # 1. Build pool: (style, sample, q) where BOTH cot and oracle have fact_score == 0.0
    cot_fu = json.load(open(args.cot_fact_used_path))
    or_fu = json.load(open(args.oracle_fact_used_path))
    if args.any_score:
        # Take union of all (sample, q) where we have both cot and oracle records.
        cot_filt = {(r["style"], r["sample_idx"], r["q_idx"]) for r in cot_fu
                    if r.get("system") == args.system
                    and r.get("variant") == "cot_select" and r["style"] in styles}
        or_filt = {(r["style"], r["sample_idx"], r["q_idx"]) for r in or_fu
                   if r.get("variant") == "oracle" and r["style"] in styles}
        pool = cot_filt & or_filt
        print(f"Pool (any_score): cot={len(cot_filt)}, oracle={len(or_filt)}, intersect={len(pool)}", file=sys.stderr)
    else:
        cot_filt = {(r["style"], r["sample_idx"], r["q_idx"]) for r in cot_fu
                    if r.get("fact_score") == args.cot_score and r.get("system") == args.system
                    and r.get("variant") == "cot_select" and r["style"] in styles}
        or_filt = {(r["style"], r["sample_idx"], r["q_idx"]) for r in or_fu
                   if r.get("fact_score") == args.oracle_score
                   and r.get("variant") == "oracle" and r["style"] in styles}
        pool = cot_filt & or_filt
        print(f"Pool: cot(fact={args.cot_score})={len(cot_filt)}, oracle(fact={args.oracle_score})={len(or_filt)}, intersect={len(pool)}", file=sys.stderr)

    # 2. Load responses
    cot_resp = json.load(open(args.cot_responses_path))
    or_resp  = json.load(open(args.oracle_responses_path))
    cot_map = {(r["style"], r["sample_idx"], r["q_idx"]): r for r in cot_resp
               if r.get("variant") == "cot_select" and r.get("system") == args.system}
    or_map  = {(r["style"], r["sample_idx"], r["q_idx"]): r for r in or_resp
               if r.get("variant") == "oracle"}

    dialog_data = json.load(open(args.sample_path))

    # 3. Build jobs (cited memory for cot, gold turns for oracle)
    jobs = []
    for k in pool:
        cot = cot_map.get(k); ora = or_map.get(k)
        if not cot or not ora: continue
        if cot.get("error") or ora.get("error"): continue
        if not cot.get("response") or not ora.get("response"): continue
        # CoT cited memory: lookup top-K docs by 1-indexed cited_ids
        cited_ids = cot.get("cited_ids") or []
        docs = docs_for_query(args.system, k[1], k[0], k[2], k=10)
        cited_docs = []
        for cid in cited_ids:
            if isinstance(cid, int) and 1 <= cid <= len(docs):
                cited_docs.append(docs[cid - 1])
        cot_memory = "\n\n".join(f"- {d}" for d in cited_docs) if cited_docs else "(no cited memory)"
        # Oracle memory: gold evidence turns
        gold_ids = set(ora.get("gold_evidence") or [])
        sample_obj = dialog_data[k[1]]
        all_t = all_speaker_turns(sample_obj)
        gold_turns = [t for t in all_t if t[0] in gold_ids]
        oracle_memory = format_turns_block(gold_turns) if gold_turns else "(no evidence)"
        jobs.append({
            "style": k[0], "sample_idx": k[1], "q_idx": k[2],
            "query": cot["query"],
            "user_name": cot.get("subject_speaker_name") or "the user",
            "cot_response": cot["response_only"] or cot["response"],
            "cot_memory": cot_memory,
            "oracle_response": ora["response"],
            "oracle_memory": oracle_memory,
            "cited_ids": cited_ids,
        })
    print(f"Jobs (have both responses): {len(jobs)}", file=sys.stderr)

    # Resume
    out: List[Dict[str, Any]] = []
    done = set()
    if os.path.exists(args.output):
        out = json.load(open(args.output))
        for r in out:
            done.add((r["style"], r["sample_idx"], r["q_idx"]))
        print(f"Resuming, {len(done)} already done", file=sys.stderr)
    pending = [j for j in jobs if (j["style"], j["sample_idx"], j["q_idx"]) not in done]
    print(f"Pending: {len(pending)}", file=sys.stderr)

    client = _anthropic_client()
    lock = threading.Lock()

    def worker(j):
        try:
            cot_total, cot_dims, _ = score_one(client, args.judge_model,
                j["query"], j["cot_memory"], j["user_name"], j["cot_response"])
            or_total, or_dims, _ = score_one(client, args.judge_model,
                j["query"], j["oracle_memory"], j["user_name"], j["oracle_response"])
            if cot_total is None or or_total is None:
                raise ValueError("score parse failed")
            if cot_total > or_total: winner = "cot"
            elif or_total > cot_total: winner = "oracle"
            else: winner = "tie"
            entry = {
                "style": j["style"], "sample_idx": j["sample_idx"], "q_idx": j["q_idx"],
                "cot_score": cot_total, "cot_dims": cot_dims,
                "oracle_score": or_total, "oracle_dims": or_dims,
                "winner": winner,
                "cited_ids": j["cited_ids"],
            }
        except Exception as ex:
            entry = {
                "style": j["style"], "sample_idx": j["sample_idx"], "q_idx": j["q_idx"],
                "error": f"{type(ex).__name__}: {str(ex)[:150]}",
            }
        with lock:
            out.append(entry)
            if len(out) % args.checkpoint_every == 0:
                with open(args.output + ".tmp", "w") as f:
                    json.dump(out, f, ensure_ascii=False, indent=2)
                os.replace(args.output + ".tmp", args.output)

    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = [ex.submit(worker, j) for j in pending]
        for _ in tqdm(as_completed(futs), total=len(futs), desc="e3"):
            pass

    with open(args.output, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"Wrote {args.output}", file=sys.stderr)

    # Aggregate
    by_style = {}
    for r in out:
        if "winner" not in r: continue
        by_style.setdefault(r["style"], Counter())[r["winner"]] += 1
    print(f"\n{'style':<14} {'n':<5} {'cot':<14} {'oracle':<14} {'tie':<10}")
    for st, c in by_style.items():
        n = c["cot"] + c["oracle"] + c["tie"]
        if n == 0: continue
        print(f"{st:<14} {n:<5} {c['cot']} ({c['cot']/n:.1%})  {c['oracle']} ({c['oracle']/n:.1%})  {c['tie']} ({c['tie']/n:.1%})")


if __name__ == "__main__":
    main()
