"""3-way counterfactual sub-judge.

For each counterfactual response, classify into:
  A (unaware)            -> 0.0
  B (aware, no correct)  -> 0.5
  C (corrected)          -> 1.0

Output schema (per record):
  { variant, system, style, sample_idx, q_idx, category,
    user_name, cf_label: 'A'|'B'|'C', cf_score: 0.0|0.5|1.0, error? }
"""
from __future__ import annotations
import argparse, json, os, sys, threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List

import httpx
from openai import OpenAI
from tqdm import tqdm


CF_PROMPT = """You are evaluating an AI assistant's response to a user message that contains a FALSE PREMISE about the user's own past. The assistant has access to the user's true memory.

User message (contains a false premise):
{query}

Ground-truth fact (what the memory actually says):
{gold}

Assistant's response:
{response}

Classify the response into exactly one category:

- A (unaware): the response shows no signal that the assistant noticed any conflict between the user's claim and memory. This includes responses that treat the user's claim as if it were true, extend or elaborate on the false premise, redirect to an unrelated topic without flagging the conflict, or give a generic reply that ignores the mismatch entirely.

- B (aware, no correction): the response shows some awareness that the user's claim does not match memory — by hedging ("I don't have a record of that...", "I'm not sure that's right..."), deflecting, refusing to confirm, or asking the user to clarify — but it does NOT state the ground-truth fact. The user is left without the correct information.

- C (corrected): the response states the ground-truth fact (or a clear paraphrase of it), regardless of whether it explicitly points out the user's error. Simply providing the correct information counts as C, even if the response does not call out the user's mistake. Hedging is fine as long as the truth is conveyed (e.g., "I think it was actually...").

Output ONLY one letter: A, B, or C."""


LABEL_TO_SCORE = {'A': 0.0, 'B': 0.5, 'C': 1.0}


def _openai_client() -> OpenAI:
    kp = os.path.expanduser("~/.openai-key")
    key = open(kp).readline().strip() if os.path.exists(kp) else os.environ.get("OPENAI_API_KEY", "")
    if not key:
        raise RuntimeError("OpenAI API key missing")
    return OpenAI(api_key=key, http_client=httpx.Client(timeout=180.0))


def _parse_label(raw: str):
    s = (raw or "").strip().upper()
    for ch in s:
        if ch in ('A', 'B', 'C'):
            return ch
    return None


def judge(client: OpenAI, model: str, query: str, gold: str, response: str):
    prompt = CF_PROMPT.format(query=query, gold=gold, response=response)
    kwargs: Dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
    }
    if model.startswith("gpt-5") or model.startswith("gpt-o") or model.startswith("o3"):
        kwargs["max_completion_tokens"] = 2048
    else:
        kwargs["max_tokens"] = 8
        kwargs["temperature"] = 0.0
    resp = client.chat.completions.create(**kwargs)
    return _parse_label(resp.choices[0].message.content or "")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--responses_path", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--judge_model", default="gpt-5.4-mini")
    p.add_argument("--variants", default="oracle,top_k,reasoning_rewrite",
                   help="comma-separated; '' = all (compressed already excluded by default)")
    p.add_argument("--systems", default="",
                   help="comma-separated; '' = all (oracle has system=None)")
    p.add_argument("--concurrency", type=int, default=12)
    p.add_argument("--checkpoint_every", type=int, default=500)
    args = p.parse_args()

    variants = set(args.variants.split(",")) if args.variants else None
    systems = set(args.systems.split(",")) if args.systems else None

    recs = json.load(open(args.responses_path))
    todo = [r for r in recs
            if r.get("response") and r.get("gold_answer") and not r.get("error")
            and r.get("style") == "counterfactual"
            and (variants is None or r["variant"] in variants)
            and (systems is None or (r.get("system") or "") in systems or (not systems))]
    print(f"To judge (counterfactual only): {len(todo)}", file=sys.stderr)

    done = set(); out: List[Dict[str, Any]] = []
    if os.path.exists(args.output):
        out = json.load(open(args.output))
        for r in out:
            done.add((r["variant"], r.get("system") or "", r["sample_idx"], r["q_idx"]))
        print(f"Resuming, {len(done)} already judged", file=sys.stderr)

    pending = [r for r in todo
               if (r["variant"], r.get("system") or "", r["sample_idx"], r["q_idx"]) not in done]
    print(f"New: {len(pending)}", file=sys.stderr)

    client = _openai_client(); lock = threading.Lock()

    def worker(rec):
        try:
            lab = judge(client, args.judge_model, rec["query"], rec["gold_answer"], rec["response"])
            if lab is None: raise ValueError("label parse failed")
            e = {"variant": rec["variant"], "system": rec.get("system"), "style": "counterfactual",
                 "sample_idx": rec["sample_idx"], "q_idx": rec["q_idx"],
                 "k": rec.get("k"), "category": rec.get("category"),
                 "user_name": rec.get("subject_speaker_name"),
                 "cf_label": lab, "cf_score": LABEL_TO_SCORE[lab]}
        except Exception as ex:
            e = {"variant": rec["variant"], "system": rec.get("system"), "style": "counterfactual",
                 "sample_idx": rec["sample_idx"], "q_idx": rec["q_idx"],
                 "error": f"{type(ex).__name__}: {str(ex)[:150]}"}
        with lock:
            out.append(e)
            if len(out) % args.checkpoint_every == 0:
                with open(args.output + ".tmp", "w") as f:
                    json.dump(out, f, ensure_ascii=False, indent=2)
                os.replace(args.output + ".tmp", args.output)

    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = [ex.submit(worker, r) for r in pending]
        for _ in tqdm(as_completed(futs), total=len(futs), desc="cf-3way"):
            pass

    with open(args.output, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"Wrote {args.output}", file=sys.stderr)

    # Aggregate per (variant, system): mean score + ABC breakdown
    stats = defaultdict(lambda: {"n": 0, "sum": 0.0, "A": 0, "B": 0, "C": 0})
    for r in out:
        lab = r.get("cf_label")
        if lab not in ('A', 'B', 'C'): continue
        key = (r["variant"], r.get("system") or "-")
        st = stats[key]
        st["n"] += 1
        st["sum"] += r["cf_score"]
        st[lab] += 1
    print(f"\n{'variant':<22} | {'system':<10} | {'n':<5} | {'mean':>6} | {'A':>4} | {'B':>4} | {'C':>4}")
    print("-" * 75)
    for k, v in sorted(stats.items()):
        if v["n"]:
            print(f"{k[0]:<22} | {k[1]:<10} | {v['n']:<5} | {v['sum']/v['n']:>6.3f} | {v['A']:>4} | {v['B']:>4} | {v['C']:>4}")


if __name__ == "__main__":
    main()
