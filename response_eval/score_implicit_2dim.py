"""Two-dimensional judge for implicit-style responses.

Current implicit judge bundles fact-coverage + situation-tie into one binary.
This script separately asks:
  1. fact_used: did the response surface the gold fact (substance present)?
  2. situation_tied: did the response acknowledge/respond to the user's stated situation?

Derived metrics per (variant, system):
  - full_success = both fact_used AND situation_tied → mirrors the original binary YES
  - fact_only / situation_only / neither: failure-mode distribution

Input:  outputs_response_eval/<run>/responses.json  (filter style=implicit)
Output: outputs_response_eval/<run>/implicit_2dim.json
"""
from __future__ import annotations
import argparse, json, os, re, sys, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict

import httpx
from openai import OpenAI
from tqdm import tqdm


JUDGE_PROMPT = """You are evaluating an AI assistant's response to an implicit-style memory-augmented query. The user shared a situation/concern; a relevant past fact is in the assistant's memory.

User's situational message:
{query}

Relevant past fact the assistant should use:
{gold}

Assistant's response:
{response}

Evaluate TWO independent dimensions:

(A) fact_used: Did the response surface the substance of the relevant past fact (paraphrasing OK; vague allusions DO NOT count)? YES or NO.

(B) situation_tied: Did the response acknowledge or respond to the user's stated situation/concern (rather than dumping the fact without context)? YES or NO.

These are independent — a response can use the fact without tying it to the situation, or tie to the situation without using the fact.

Output STRICT JSON ONLY:
{{
  "fact_used": "YES" or "NO",
  "fact_evidence": "<short quote or 'absent'>",
  "situation_tied": "YES" or "NO",
  "situation_evidence": "<short quote or 'generic'>"
}}
No preamble, no markdown."""


def _strip_fences(text: str) -> str:
    text = (text or "").strip()
    m = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    return m.group(1) if m else text


def _openai_client():
    kp = os.path.expanduser("~/.openai-key")
    api_key = open(kp).readline().strip() if os.path.exists(kp) else os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError("OpenAI API key missing")
    return OpenAI(api_key=api_key, http_client=httpx.Client(timeout=180.0))


def judge_one(client, model: str, query: str, gold: str, response: str) -> Dict[str, Any]:
    prompt = JUDGE_PROMPT.format(query=query, gold=gold, response=response)
    kwargs: Dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
    }
    is_gpt5 = model.startswith("gpt-5") or model.startswith("gpt-o") or model.startswith("o3")
    if is_gpt5:
        kwargs["max_completion_tokens"] = 2048
    else:
        kwargs["max_tokens"] = 1000
        kwargs["temperature"] = 0.0
    resp = client.chat.completions.create(**kwargs)
    return json.loads(_strip_fences(resp.choices[0].message.content or ""))


def parse_yn(v):
    s = str(v or "").strip().upper()
    if s.startswith("YES"): return "yes"
    if s.startswith("NO"): return "no"
    return ""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--responses_path", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--judge_model", default="gpt-5.4-mini")
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--checkpoint_every", type=int, default=200)
    p.add_argument("--style", default="implicit", help="Filter to one style (implicit/dialog/...)")
    p.add_argument("--variant", default=None, help="Optional: filter to one variant (e.g., oracle)")
    args = p.parse_args()

    responses = json.load(open(args.responses_path))
    selected = [r for r in responses
                if r['style'] == args.style
                and r.get('response')
                and r.get('gold_answer')
                and not r.get('error')]
    if args.variant:
        selected = [r for r in selected if r['variant'] == args.variant]
    # Exclude cat 5 (different rubric)
    selected = [r for r in selected if r.get('category') != 5]
    print(f"{args.style} responses to judge ({args.variant or 'all variants'}): {len(selected)}", file=sys.stderr)
    implicit = selected  # name kept for backward compat below

    # Resume
    out = []
    done = set()
    if os.path.exists(args.output):
        out = json.load(open(args.output))
        for r in out:
            done.add((r['variant'], r.get('system'), r['sample_idx'], r['q_idx']))
        print(f"Resuming: {len(done)} already scored", file=sys.stderr)

    todo = [r for r in implicit if (r['variant'], r.get('system'), r['sample_idx'], r['q_idx']) not in done]
    print(f"To score: {len(todo)}", file=sys.stderr)

    client = _openai_client()
    lock = threading.Lock()

    def worker(rec):
        try:
            j = judge_one(client, args.judge_model, rec['query'], rec['gold_answer'], rec['response'])
            f = parse_yn(j.get('fact_used'))
            s = parse_yn(j.get('situation_tied'))
            entry = {
                'variant': rec['variant'],
                'system': rec.get('system'),
                'sample_idx': rec['sample_idx'],
                'q_idx': rec['q_idx'],
                'category': rec.get('category'),
                'fact_used': f,
                'situation_tied': s,
                'fact_evidence': str(j.get('fact_evidence', ''))[:200],
                'situation_evidence': str(j.get('situation_evidence', ''))[:200],
            }
        except Exception as e:
            entry = {
                'variant': rec['variant'],
                'system': rec.get('system'),
                'sample_idx': rec['sample_idx'],
                'q_idx': rec['q_idx'],
                'error': f"{type(e).__name__}: {str(e)[:150]}",
            }
        with lock:
            out.append(entry)
            if len(out) % args.checkpoint_every == 0:
                with open(args.output + ".tmp", 'w') as f:
                    json.dump(out, f, ensure_ascii=False, indent=2)
                os.replace(args.output + ".tmp", args.output)

    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = [ex.submit(worker, r) for r in todo]
        for _ in tqdm(as_completed(futs), total=len(futs), desc="2dim"):
            pass

    with open(args.output, 'w') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"Wrote {args.output}", file=sys.stderr)

    # Aggregate
    from collections import defaultdict
    stats = defaultdict(lambda: {'n':0,'fact':0,'sit':0,'both':0,'neither':0})
    for r in out:
        if r.get('error') or r.get('fact_used') not in ('yes','no'): continue
        key = (r['variant'], r.get('system') or '-')
        s = stats[key]
        s['n']+=1
        f = r['fact_used']=='yes'; si = r['situation_tied']=='yes'
        if f: s['fact']+=1
        if si: s['sit']+=1
        if f and si: s['both']+=1
        if not f and not si: s['neither']+=1

    print(f"\n{'variant':<22} | {'system':<10} | {'n':<4} | {'fact_used':>9} | {'sit_tied':>9} | {'both(full)':>11} | {'neither':>8}")
    print('-'*80)
    for key, s in sorted(stats.items()):
        v, sn = key
        n = s['n']
        if n == 0: continue
        print(f"{v:<22} | {sn:<10} | {n:<4} | {s['fact']/n:>9.3f} | {s['sit']/n:>9.3f} | {s['both']/n:>11.3f} | {s['neither']/n:>8.3f}")


if __name__ == "__main__":
    main()
