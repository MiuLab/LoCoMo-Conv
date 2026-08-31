"""Partial-credit version of fact_used judge.

Three-level score (0.0 / 0.5 / 1.0) replaces the binary YES/NO of
score_fact_used.py. Designed to distinguish responses that convey the
central concept but miss specifics from those that completely omit
or contradict the fact.

Output schema (per record):
  { variant, system, style, sample_idx, q_idx, category,
    user_name, fact_score: 0.0|0.5|1.0, error? }
"""
from __future__ import annotations
import argparse, json, os, sys, threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List

import httpx
from openai import OpenAI
from tqdm import tqdm


PARTIAL_PROMPT = """You are evaluating whether an AI assistant's response correctly conveys a target fact from the user's past conversation memory.

User (named {user_name}): {query}

Target fact the response should convey:
{gold}

Assistant's response:
{response}

Score the response on a 3-level scale:

- 1.0 (full): the substance of the target fact is correctly conveyed. Paraphrasing is fine; verbatim not required. Hedging or qualifying language ("I think...", "if I recall...", "based on what we discussed...") is fine. Minor stylistic differences are fine.

- 0.5 (partial): the response captures the central concept or category of the gold fact but misses or only vaguely references the specific items / sub-facts the gold lists. Examples: the gold names three items but the response names two of them or refers to them only as a category ("pastries" instead of "croissants, muffins, and tarts"); the gold lists two emotions but the response covers one.

- 0.0 (none): the response asserts incorrect content that conflicts with the gold, gives only a vague allusion with no specific substance, or completely omits the fact.

Output ONLY one number: 1.0, 0.5, or 0.0."""


def _openai_client() -> OpenAI:
    kp = os.path.expanduser("~/.openai-key")
    key = open(kp).readline().strip() if os.path.exists(kp) else os.environ.get("OPENAI_API_KEY", "")
    if not key:
        raise RuntimeError("OpenAI API key missing")
    return OpenAI(api_key=key, http_client=httpx.Client(timeout=180.0))


def _parse_score(raw: str):
    s = (raw or "").strip()
    # Look for 1.0 / 0.5 / 0.0 anywhere in the output
    for token in ("1.0", "0.5", "0.0", "1", "0"):
        if token in s:
            return float(token)
    return None


def judge(client: OpenAI, model: str, query: str, gold: str, response: str, user_name: str):
    prompt = PARTIAL_PROMPT.format(
        user_name=user_name or "the user",
        query=query, gold=gold, response=response,
    )
    kwargs: Dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
    }
    if model.startswith("gpt-5") or model.startswith("gpt-o") or model.startswith("o3"):
        kwargs["max_completion_tokens"] = 2048
    else:
        kwargs["max_tokens"] = 16
        kwargs["temperature"] = 0.0
    resp = client.chat.completions.create(**kwargs)
    return _parse_score(resp.choices[0].message.content or "")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--responses_path", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--judge_model", default="gpt-5.4-mini")
    p.add_argument("--styles", default="dialog,implicit,counterfactual,composed")
    p.add_argument("--variants", default="oracle,top_k,reasoning_rewrite",
                   help="comma-separated; '' = all (compressed already excluded by default)")
    p.add_argument("--systems", default="",
                   help="comma-separated; '' = all (oracle has system=None)")
    p.add_argument("--concurrency", type=int, default=12)
    p.add_argument("--checkpoint_every", type=int, default=500)
    args = p.parse_args()

    styles = set(args.styles.split(",")) if args.styles else None
    variants = set(args.variants.split(",")) if args.variants else None
    systems = set(args.systems.split(",")) if args.systems else None

    recs = json.load(open(args.responses_path))
    todo = [r for r in recs
            if r.get("response") and r.get("gold_answer") and not r.get("error")
            and r.get("category") != 5
            and (styles is None or r["style"] in styles)
            and (variants is None or r["variant"] in variants)
            and (systems is None or (r.get("system") or "") in systems or (not systems))]
    print(f"To judge: {len(todo)}", file=sys.stderr)

    done = set(); out: List[Dict[str, Any]] = []
    if os.path.exists(args.output):
        out = json.load(open(args.output))
        for r in out:
            done.add((r["variant"], r.get("system") or "", r["style"], r["sample_idx"], r["q_idx"]))
        print(f"Resuming, {len(done)} already judged", file=sys.stderr)

    pending = [r for r in todo
               if (r["variant"], r.get("system") or "", r["style"], r["sample_idx"], r["q_idx"]) not in done]
    print(f"New: {len(pending)}", file=sys.stderr)

    client = _openai_client(); lock = threading.Lock()

    def worker(rec):
        try:
            sc = judge(client, args.judge_model, rec["query"], rec["gold_answer"], rec["response"],
                       rec.get("subject_speaker_name") or "the user")
            if sc is None: raise ValueError("score parse failed")
            e = {"variant": rec["variant"], "system": rec.get("system"), "style": rec["style"],
                 "sample_idx": rec["sample_idx"], "q_idx": rec["q_idx"],
                 "k": rec.get("k"), "category": rec.get("category"),
                 "user_name": rec.get("subject_speaker_name"),
                 "fact_score": sc}
        except Exception as ex:
            e = {"variant": rec["variant"], "system": rec.get("system"), "style": rec["style"],
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
        for _ in tqdm(as_completed(futs), total=len(futs), desc="partial"):
            pass

    with open(args.output, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"Wrote {args.output}", file=sys.stderr)

    # Aggregate per (variant, system, style): mean score + breakdown
    stats = defaultdict(lambda: {"n": 0, "sum": 0.0, "full": 0, "partial": 0, "none": 0})
    for r in out:
        sc = r.get("fact_score")
        if sc not in (0.0, 0.5, 1.0): continue
        key = (r["variant"], r.get("system") or "-", r["style"])
        st = stats[key]
        st["n"] += 1
        st["sum"] += sc
        if sc == 1.0: st["full"] += 1
        elif sc == 0.5: st["partial"] += 1
        else: st["none"] += 1
    print(f"\n{'variant':<20} | {'system':<10} | {'style':<14} | {'n':<5} | {'mean':>6} | {'F':>4} | {'P':>4} | {'N':>4}")
    print("-" * 80)
    for k, v in sorted(stats.items()):
        if v["n"]:
            print(f"{k[0]:<20} | {k[1]:<10} | {k[2]:<14} | {v['n']:<5} | {v['sum']/v['n']:>6.3f} | {v['full']:>4} | {v['partial']:>4} | {v['none']:>4}")


if __name__ == "__main__":
    main()
