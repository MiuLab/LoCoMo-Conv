"""Atomic-fact coverage scoring for composed queries.

For each composed cluster, the LLM judge checks EACH gold fact independently
and outputs YES/NO. Score = covered_facts / total_facts (continuous 0-1).

This replaces the strict "cover ALL or fail" binary judge for composed,
giving partial credit for responses that capture most but not all gold facts.

Output: outputs_response_eval/<run>/composed_atomic.json
        Per-record: per_fact_judgments (list of YES/NO), coverage_score, n_facts
"""
from __future__ import annotations
import argparse, json, os, re, sys, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List

import httpx
from openai import OpenAI
from tqdm import tqdm


JUDGE_PROMPT = """You are evaluating whether an AI assistant's response covers each of several atomic facts.

User's message:
{query}

Atomic gold facts the response should cover (one per line, numbered):
{facts_block}

Assistant's response:
{response}

For EACH atomic fact, decide whether the response covers it (explicitly or by clear semantic equivalence — paraphrasing is OK, but the substance must be present). Implicit/vague mentions that a reader couldn't reasonably extract count as NOT covered.

Output STRICT JSON ONLY:
{{
  "judgments": [
    {{"fact": 1, "covered": true, "evidence": "<short quote from response>"}},
    {{"fact": 2, "covered": false, "evidence": "<why not>"}},
    ...
  ]
}}
One entry per fact. No preamble, no markdown."""


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


def judge_one(client, model: str, query: str, facts: List[str], response: str) -> Dict[str, Any]:
    facts_block = "\n".join(f"{i+1}. {f}" for i, f in enumerate(facts))
    prompt = JUDGE_PROMPT.format(query=query, facts_block=facts_block, response=response)
    kwargs: Dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
    }
    is_gpt5 = model.startswith("gpt-5") or model.startswith("gpt-o") or model.startswith("o3") or model.startswith("o4")
    if is_gpt5:
        kwargs["max_completion_tokens"] = 2048
    else:
        kwargs["max_tokens"] = 1500
        kwargs["temperature"] = 0.0
    resp = client.chat.completions.create(**kwargs)
    raw = (resp.choices[0].message.content or "").strip()
    return json.loads(_strip_fences(raw))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--responses_path", default="outputs_response_eval/full_convergent/responses.json")
    p.add_argument("--multimem_path", default="data/locomo10_multimem_full.json")
    p.add_argument("--output", default="outputs_response_eval/full_convergent/composed_atomic.json")
    p.add_argument("--judge_model", default="gpt-5.4-mini")
    p.add_argument("--variants", default="oracle,top_k,compressed,reasoning_rewrite",
                   help="comma-sep variants to re-score")
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--checkpoint_every", type=int, default=100)
    args = p.parse_args()

    # Build cluster lookup: cluster_id -> list of atomic gold (question, answer) pairs as strings
    mm = json.load(open(args.multimem_path))
    cluster_facts: Dict[str, List[str]] = {}
    for c in mm:
        cid = str(c.get("cluster_id"))
        facts = []
        for mf in c.get("member_facts", []):
            q = mf.get("question") or ""
            ans = mf.get("answer")
            if isinstance(ans, list):
                ans_str = ", ".join(str(a) for a in ans)
            else:
                ans_str = str(ans) if ans else ""
            if q and ans_str:
                facts.append(f"{q.rstrip('?')}: {ans_str}")
        if facts:
            cluster_facts[cid] = facts

    # Load responses
    responses = json.load(open(args.responses_path))
    variants = set(v.strip() for v in args.variants.split(","))
    composed = [r for r in responses
                if r['style'] == 'composed'
                and r['variant'] in variants
                and r.get('response')
                and not r.get('error')]
    print(f"Composed responses to score: {len(composed)}", file=sys.stderr)

    # Resume
    out: List[Dict[str, Any]] = []
    done = set()
    if os.path.exists(args.output):
        out = json.load(open(args.output))
        for r in out:
            done.add((r.get('variant'), r.get('system'), r.get('sample_idx'), r.get('cluster_id')))
        print(f"Resuming: {len(done)} already scored", file=sys.stderr)

    todo = [r for r in composed
            if (r['variant'], r.get('system'), r.get('sample_idx'), r.get('cluster_id')) not in done]
    print(f"To score: {len(todo)}", file=sys.stderr)

    client = _openai_client()
    lock = threading.Lock()

    def worker(rec):
        cid = str(rec.get('cluster_id'))
        facts = cluster_facts.get(cid)
        if not facts:
            return None
        try:
            r = judge_one(client, args.judge_model, rec['query'], facts, rec['response'])
            judgments = r.get('judgments', [])
            # Make robust: pad/truncate to match n_facts
            covered = []
            for j in judgments:
                if isinstance(j, dict):
                    covered.append(bool(j.get('covered', False)))
            # Align to facts count
            if len(covered) < len(facts):
                covered += [False] * (len(facts) - len(covered))
            covered = covered[: len(facts)]
            score = sum(covered) / len(facts) if facts else 0.0
            entry = {
                'variant': rec['variant'],
                'system': rec.get('system'),
                'sample_idx': rec['sample_idx'],
                'cluster_id': cid,
                'n_facts': len(facts),
                'covered': sum(covered),
                'coverage_score': score,
                'per_fact_covered': covered,
                'judge_response': r,
            }
        except Exception as e:
            entry = {
                'variant': rec['variant'],
                'system': rec.get('system'),
                'sample_idx': rec['sample_idx'],
                'cluster_id': cid,
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
        for _ in tqdm(as_completed(futs), total=len(futs), desc="atomic-score"):
            pass

    with open(args.output, 'w') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"Wrote {args.output}", file=sys.stderr)

    # Quick aggregate
    from collections import defaultdict
    stats = defaultdict(lambda: {'n': 0, 'cov_sum': 0.0, 'full_yes': 0, 'zero_cov': 0})
    for r in out:
        if 'coverage_score' not in r:
            continue
        key = (r['variant'], r.get('system') or '-')
        s = stats[key]
        s['n'] += 1
        s['cov_sum'] += r['coverage_score']
        if r['coverage_score'] == 1.0:
            s['full_yes'] += 1
        if r['coverage_score'] == 0.0:
            s['zero_cov'] += 1

    print(f"\n{'variant':<22} | {'system':<12} | {'n':<4} | {'mean_cov':>8} | {'full_yes':>8} | {'zero_cov':>8}")
    print('-' * 70)
    for key, s in sorted(stats.items()):
        v, sn = key
        n = s['n']
        if n == 0: continue
        print(f"{v:<22} | {sn:<12} | {n:<4} | {s['cov_sum']/n:>8.3f} | {s['full_yes']/n:>8.3f} | {s['zero_cov']/n:>8.3f}")


if __name__ == "__main__":
    main()
