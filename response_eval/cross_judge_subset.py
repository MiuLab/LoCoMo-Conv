"""Cross-judge the response subset with claude + qwen, comparing to gpt-5.4-mini.

Reads outputs_response_eval/human_judge/subset.json
Writes outputs_response_eval/human_judge/cross_judge.json
       (each item augmented with claude_judge + qwen_judge + cat5_halluc from each)
"""
from __future__ import annotations
import argparse, json, os, sys, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict

import httpx
from openai import OpenAI
from tqdm import tqdm

# Reuse per-style prompts from score_responses
sys.path.insert(0, "scripts")
from score_responses import (
    JUDGE_PROMPTS, JUDGE_PROMPT_DIALOG, CAT5_PROMPT
)


SAFE_UA = {"User-Agent": "curl/8.0"}


def claude_client():
    import anthropic
    kp = os.path.expanduser("~/.anthropic-key")
    api_key = open(kp).readline().strip() if os.path.exists(kp) else os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key: raise RuntimeError("Anthropic key missing")
    return anthropic.Anthropic(api_key=api_key, timeout=180.0)


def qwen_client():
    http = httpx.Client(headers=SAFE_UA, timeout=180.0)
    return OpenAI(
        base_url="http://localhost:8000/v1",
        api_key="EMPTY", http_client=http, default_headers=SAFE_UA)


def parse_yn(raw: str) -> str:
    raw = (raw or "").strip().upper()
    if "YES" in raw and ("NO" not in raw.split("YES", 1)[0]):
        return "yes"
    if "NO" in raw and ("YES" not in raw.split("NO", 1)[0]):
        return "no"
    if raw.startswith("YES"): return "yes"
    if raw.startswith("NO"): return "no"
    return ""


def judge_claude(client, item):
    style = item['style']
    is_cat5 = item.get('category') == 5
    if is_cat5: return {"main": "", "halluc": -1}  # subset excludes cat5
    prompt_tmpl = JUDGE_PROMPTS.get(style, JUDGE_PROMPT_DIALOG)
    prompt = prompt_tmpl.format(query=item['query'], gold=item['gold_answer'], response=item['response'])
    resp = client.messages.create(
        model="claude-sonnet-4-5", max_tokens=4096,
        thinking={"type": "enabled", "budget_tokens": 2048},
        messages=[{"role": "user", "content": prompt}],
    )
    out = ""
    for b in resp.content:
        if getattr(b, "type", "") == "text":
            out += getattr(b, "text", "")
    return {"main": parse_yn(out), "halluc": -1}


def judge_qwen(client, item):
    style = item['style']
    if item.get('category') == 5: return {"main": "", "halluc": -1}
    prompt_tmpl = JUDGE_PROMPTS.get(style, JUDGE_PROMPT_DIALOG)
    prompt = prompt_tmpl.format(query=item['query'], gold=item['gold_answer'], response=item['response'])
    resp = client.chat.completions.create(
        model="qwen3.6-35b-a3b",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=4096, temperature=0,
    )
    raw = resp.choices[0].message.content or ""
    return {"main": parse_yn(raw), "halluc": -1}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="outputs_response_eval/human_judge/subset.json")
    p.add_argument("--output", default="outputs_response_eval/human_judge/cross_judge.json")
    p.add_argument("--judges", default="claude,qwen", help="comma-sep")
    p.add_argument("--concurrency", type=int, default=4)
    args = p.parse_args()

    items = json.load(open(args.input))
    print(f"Items: {len(items)}", file=sys.stderr)

    # Resume from existing output if any
    if os.path.exists(args.output):
        existing = json.load(open(args.output))
        existing_by_id = {x['item_id']: x for x in existing}
        for i, it in enumerate(items):
            if it['item_id'] in existing_by_id:
                items[i].update({k:v for k,v in existing_by_id[it['item_id']].items() if k.startswith('claude') or k.startswith('qwen')})

    judges = [j.strip() for j in args.judges.split(',')]
    clients = {}
    if 'claude' in judges: clients['claude'] = claude_client()
    if 'qwen' in judges: clients['qwen'] = qwen_client()
    lock = threading.Lock()

    def worker(idx):
        it = items[idx]
        if it.get('category') == 5: return
        for j in judges:
            key = f'{j}_judge'
            if it.get(key): continue  # already done
            try:
                if j == 'claude':
                    r = judge_claude(clients['claude'], it)
                else:
                    r = judge_qwen(clients['qwen'], it)
                with lock:
                    items[idx][key] = r['main']
            except Exception as e:
                with lock:
                    items[idx][key] = ""
                    items[idx][f'{j}_error'] = str(e)[:200]

    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = [ex.submit(worker, i) for i in range(len(items))]
        for i, _ in enumerate(tqdm(as_completed(futs), total=len(futs))):
            if (i+1) % 100 == 0:
                with open(args.output, 'w') as f:
                    json.dump(items, f, ensure_ascii=False, indent=2)

    with open(args.output, 'w') as f:
        json.dump(items, f, ensure_ascii=False, indent=2)
    print(f"\nWrote {args.output}", file=sys.stderr)

    # Quick agreement stats
    from collections import Counter
    ok = [r for r in items if r.get('gpt_judge') in ('yes','no')]
    print(f"\nAgreement on {len(ok)} items:", file=sys.stderr)
    if 'claude' in judges:
        cl = [r for r in ok if r.get('claude_judge') in ('yes','no')]
        same = sum(1 for r in cl if r['gpt_judge']==r['claude_judge'])
        print(f"  gpt vs claude: {same}/{len(cl)} = {same/len(cl):.3f}" if cl else "  gpt vs claude: no claude judgments", file=sys.stderr)
    if 'qwen' in judges:
        qw = [r for r in ok if r.get('qwen_judge') in ('yes','no')]
        same = sum(1 for r in qw if r['gpt_judge']==r['qwen_judge'])
        print(f"  gpt vs qwen: {same}/{len(qw)} = {same/len(qw):.3f}" if qw else "  gpt vs qwen: no qwen judgments", file=sys.stderr)
    if 'claude' in judges and 'qwen' in judges:
        both = [r for r in ok if r.get('claude_judge') in ('yes','no') and r.get('qwen_judge') in ('yes','no')]
        all3 = sum(1 for r in both if r['gpt_judge']==r['claude_judge']==r['qwen_judge'])
        print(f"  all 3 agree: {all3}/{len(both)} = {all3/len(both):.3f}" if both else "  no 3-way", file=sys.stderr)


if __name__ == "__main__":
    main()
