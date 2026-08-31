"""Generate reasoning-model rewrites for response-eval queries.

For each unique query in data/response_eval_sample.json, ask gpt-5.4-mini to
rewrite into a concise search query that captures topic+entity+intent.

Output:
  outputs_response_eval/<run>/rewrites.json
  { "<original_query>": "<rewritten_query>", ... }

The driver consumes this file when --variants reasoning_rewrite is used.
"""
from __future__ import annotations

import argparse, json, os, sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict

import httpx
from openai import OpenAI
from tqdm import tqdm


REWRITE_PROMPT = """A user has said the following in conversation with an AI assistant. The AI needs to retrieve relevant memories from past conversations to respond well. Rewrite the user's message into a CONCISE search query that captures the topic, entity, and intent in a form a retrieval system can match (use third-person if helpful, mention specific named entities, dates, and topics).

User message:
"{query}"

Output ONLY the search query string (under 30 words). No quotes, no preamble."""


def _openai_client() -> OpenAI:
    kp = os.path.expanduser("~/.openai-key")
    if os.path.exists(kp):
        api_key = open(kp).readline().strip()
    else:
        api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError("OpenAI API key missing")
    return OpenAI(api_key=api_key, http_client=httpx.Client(timeout=120.0))


def _rewrite_one(client: OpenAI, model: str, query: str) -> str:
    is_gpt5 = model.startswith("gpt-5") or model.startswith("gpt-o") or model.startswith("o3") or model.startswith("o4")
    kwargs: Dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": REWRITE_PROMPT.format(query=query)}],
    }
    if is_gpt5:
        kwargs["max_completion_tokens"] = 200
    else:
        kwargs["max_tokens"] = 80
        kwargs["temperature"] = 0.0
    resp = client.chat.completions.create(**kwargs)
    return (resp.choices[0].message.content or "").strip().strip('"').strip("'")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sample_path", default="data/response_eval_sample.json")
    p.add_argument("--output_dir", default="outputs_response_eval/oracle_all_styles")
    p.add_argument("--model", default="gpt-5.4-mini")
    p.add_argument("--concurrency", type=int, default=8)
    args = p.parse_args()

    with open(args.sample_path) as f:
        sampled = json.load(f)

    # Gather unique queries across styles
    unique: set[str] = set()
    for style, items in sampled["samples"].items():
        if style == "composed":
            for it in items:
                q = it.get("composed_query")
                if q:
                    unique.add(q)
        else:
            for it in items:
                q = it.get("query")
                if q:
                    unique.add(q)
    print(f"Unique queries: {len(unique)}", file=sys.stderr)

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "rewrites.json")
    cache: Dict[str, str] = {}
    if os.path.exists(out_path):
        cache = json.load(open(out_path))
        print(f"Loaded {len(cache)} cached", file=sys.stderr)

    todo = [q for q in unique if q not in cache]
    print(f"To rewrite: {len(todo)}", file=sys.stderr)
    if not todo:
        print("Nothing to do.", file=sys.stderr)
        return

    client = _openai_client()
    import threading
    lock = threading.Lock()

    def work(q: str):
        try:
            rw = _rewrite_one(client, args.model, q)
        except Exception:
            rw = q  # fallback to original
        with lock:
            cache[q] = rw
            if len(cache) % 50 == 0:
                json.dump(cache, open(out_path, "w"), ensure_ascii=False, indent=2)

    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = [ex.submit(work, q) for q in todo]
        for _ in tqdm(as_completed(futs), total=len(futs), desc="rewrites"):
            pass

    json.dump(cache, open(out_path, "w"), ensure_ascii=False, indent=2)
    print(f"Wrote {out_path} (n={len(cache)})", file=sys.stderr)


if __name__ == "__main__":
    main()
