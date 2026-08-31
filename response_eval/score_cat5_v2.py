"""Cat-5 hallucination judge v2.

Differences vs the original CAT5_PROMPT in score_responses.py:
  - Includes the user's name in the judge prompt.
  - Does NOT reference adversarial_answer. Instead, shows the assistant's
    retrieved memory and asks the judge to decide whether the response
    asserts memory-grounded facts not supported by that memory.

Inputs:
  --responses_path : a responses.json (e.g., outputs_response_eval/full_convergent/responses.json
                     or the qwen-cat5 run).
  --system         : which system's retrieve cache to reconstruct memory_block from
                     (default AnchorMem).
  --variant        : 'top_k' (default). Memory_block = top-K retrieved docs.

Output: <responses_path-dir>/cat5_halluc_v2_<answer_model>.json
        list of {variant,system,style,sample_idx,q_idx,query,response,
                 memory_block,user_name,halluc:yes|no,evidence}
"""
from __future__ import annotations
import argparse, json, os, re, sys, threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List

import httpx
from openai import OpenAI
from tqdm import tqdm


CAT5_V2_PROMPT = """You are evaluating whether an AI assistant hallucinated a memory-grounded answer when the retrieved memory does not support one.

User (named {user_name}): {query}

Retrieved memory shown to the assistant:
{memory_block}

Assistant's response:
{response}

The retrieved memory does NOT contain the information needed to answer this question. Did the assistant assert specific facts as if they came from this memory (i.e., misuse the memory to fabricate a memory-grounded answer)?

- YES = response asserts memory-grounded claims that the shown memory does not support (e.g., "based on our chats you mentioned X" when no such turn is in the memory).
- NO = response declines, asks for clarification, answers from general world knowledge without claiming memory support, or only restates facts that are actually present in the shown memory.

Output ONLY one word: YES or NO."""


SYSTEM_CACHE_DIR = {
    "AnchorMem": "outputs/locomo-gemma-4-31B-it",
    "A-mem":     "outputs_amem/no_evo/locomo-gemma-4-31B-it",
    "mem0":      "outputs_mem0/locomo-gemma-4-31B-it",
    "BM25":      "outputs_bm25/locomo-bm25",
    "Dense":     "outputs_dense/locomo-dense",
}

FIELD_FOR_STYLE = {"dialog": "dialog_query", "implicit": "implicit_query"}


def load_retrieve_cache(system: str, sample_idx: int, style: str):
    base = SYSTEM_CACHE_DIR[system]
    field = FIELD_FOR_STYLE[style]
    p = os.path.join(base, f"sample_{sample_idx}", f"queries_solutions_{field}.json")
    if not os.path.exists(p):
        return []
    return json.load(open(p))


def get_topk_docs(system: str, sample_idx: int, style: str, q_idx: int, k: int, query_text: str):
    sols = load_retrieve_cache(system, sample_idx, style)
    # Try q_idx field
    for entry in sols:
        if entry.get("q_idx") == q_idx:
            return (entry.get("docs") or [])[:k]
    # AnchorMem fallback: match by question text
    for entry in sols:
        if entry.get("question") == query_text or entry.get("query") == query_text:
            return (entry.get("docs") or [])[:k]
    return []


def format_memory_block(docs: List[str]) -> str:
    if not docs:
        return "(no memories retrieved)"
    return "\n\n".join(f"- {d}" for d in docs)


def _openai_client():
    kp = os.path.expanduser("~/.openai-key")
    api_key = open(kp).readline().strip() if os.path.exists(kp) else os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError("OpenAI API key missing")
    return OpenAI(api_key=api_key, http_client=httpx.Client(timeout=180.0))


def parse_yn(raw: str) -> str:
    s = (raw or "").strip().upper()
    if "YES" in s and "NO" not in s.split("YES", 1)[0]:
        return "yes"
    if "NO" in s and "YES" not in s.split("NO", 1)[0]:
        return "no"
    return ""


def judge_one(client, model: str, query: str, memory_block: str, response: str, user_name: str):
    prompt = CAT5_V2_PROMPT.format(
        user_name=user_name or "the user",
        query=query, memory_block=memory_block, response=response,
    )
    kwargs: Dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
    }
    is_gpt5 = model.startswith("gpt-5") or model.startswith("gpt-o") or model.startswith("o3")
    if is_gpt5:
        kwargs["max_completion_tokens"] = 2048
    else:
        kwargs["max_tokens"] = 32
        kwargs["temperature"] = 0.0
    resp = client.chat.completions.create(**kwargs)
    return parse_yn(resp.choices[0].message.content or "")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--responses_path", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--system", default="AnchorMem")
    p.add_argument("--variant", default="top_k")
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--judge_model", default="gpt-5.4-mini")
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--checkpoint_every", type=int, default=100)
    p.add_argument("--label", default=None,
                   help="optional label (e.g., 'gemma'/'qwen') to tag entries")
    args = p.parse_args()

    responses = json.load(open(args.responses_path))
    todo_recs = [r for r in responses
                 if r.get("category") == 5
                 and r["variant"] == args.variant
                 and r.get("system") == args.system
                 and r["style"] in FIELD_FOR_STYLE
                 and r.get("response")
                 and not r.get("error")]
    print(f"To judge: {len(todo_recs)} cat-5 responses ({args.system}/{args.variant})", file=sys.stderr)

    out: List[Dict[str, Any]] = []
    done = set()
    if os.path.exists(args.output):
        out = json.load(open(args.output))
        for r in out:
            done.add((r["variant"], r["system"], r["style"], r["sample_idx"], r["q_idx"]))
        print(f"Resuming, {len(done)} already judged", file=sys.stderr)

    pending = [r for r in todo_recs if (r["variant"], r["system"], r["style"], r["sample_idx"], r["q_idx"]) not in done]
    print(f"New to judge: {len(pending)}", file=sys.stderr)

    client = _openai_client()
    lock = threading.Lock()

    def worker(rec):
        try:
            docs = get_topk_docs(args.system, rec["sample_idx"], rec["style"], rec["q_idx"], args.k, rec["query"])
            mem = format_memory_block(docs)
            user_name = rec.get("subject_speaker_name") or "the user"
            verdict = judge_one(client, args.judge_model, rec["query"], mem, rec["response"], user_name)
            entry = {
                "label": args.label,
                "variant": rec["variant"],
                "system": rec["system"],
                "style": rec["style"],
                "sample_idx": rec["sample_idx"],
                "q_idx": rec["q_idx"],
                "user_name": user_name,
                "query": rec["query"],
                "response": rec["response"][:400],
                "memory_block_chars": len(mem),
                "halluc": verdict,
            }
        except Exception as e:
            entry = {
                "label": args.label,
                "variant": rec["variant"],
                "system": rec["system"],
                "style": rec["style"],
                "sample_idx": rec["sample_idx"],
                "q_idx": rec["q_idx"],
                "error": f"{type(e).__name__}: {str(e)[:150]}",
            }
        with lock:
            out.append(entry)
            if len(out) % args.checkpoint_every == 0:
                with open(args.output + ".tmp", "w") as f:
                    json.dump(out, f, ensure_ascii=False, indent=2)
                os.replace(args.output + ".tmp", args.output)

    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = [ex.submit(worker, r) for r in pending]
        for _ in tqdm(as_completed(futs), total=len(futs), desc="judge"):
            pass

    with open(args.output, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"Wrote {args.output}", file=sys.stderr)

    # Aggregate
    stats = defaultdict(lambda: {"n": 0, "h": 0})
    for r in out:
        if r.get("halluc") not in ("yes", "no"):
            continue
        key = (r.get("label") or "-", r["style"])
        stats[key]["n"] += 1
        if r["halluc"] == "yes":
            stats[key]["h"] += 1

    print(f"\n{'label':<10} | {'style':<10} | {'n':<5} | {'halluc_rate':>12}")
    print("-" * 50)
    for (lbl, sty), v in sorted(stats.items()):
        n = v["n"]
        print(f"{lbl:<10} | {sty:<10} | {n:<5} | {v['h']/n if n else 0:>12.3f}")


if __name__ == "__main__":
    main()
