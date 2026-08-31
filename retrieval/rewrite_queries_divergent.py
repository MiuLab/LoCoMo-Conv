"""Generate DIVERGENT (multi-facet) query rewrites for response-eval queries.

For each unique query, ask gpt-5.4-mini to produce 3-5 short search queries
covering DIFFERENT facets (entities, time periods, themes, related angles).
The retrieval helper then runs each facet separately and fuses results via RRF.

Output:
  outputs_response_eval/<run>/rewrites_divergent.json
  { "<original_query>": ["facet1", "facet2", ...], ... }
"""
from __future__ import annotations

import argparse, json, os, sys, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List

import httpx
from openai import OpenAI
from tqdm import tqdm


REWRITE_PROMPT = """A user has said the following in conversation with an AI assistant. The AI needs to retrieve relevant memories from past conversations to respond well.

Generate 3 to 5 SHORT search queries, each capturing a DIFFERENT facet of what would be relevant: different entities, time periods, themes, or related angles. Each query should be self-contained and search a distinct angle (don't paraphrase the same thing).

User message:
"{query}"

Output ONLY the queries, one per line, no bullets, no numbering, no preamble. Each under 15 words."""


def _openai_client() -> OpenAI:
    kp = os.path.expanduser("~/.openai-key")
    if os.path.exists(kp):
        api_key = open(kp).readline().strip()
    else:
        api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError("OpenAI API key missing")
    return OpenAI(api_key=api_key, http_client=httpx.Client(timeout=120.0))


def _rewrite_one(client: OpenAI, model: str, query: str) -> List[str]:
    is_gpt5 = model.startswith("gpt-5") or model.startswith("gpt-o") or model.startswith("o3") or model.startswith("o4")
    kwargs: Dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": REWRITE_PROMPT.format(query=query)}],
    }
    if is_gpt5:
        kwargs["max_completion_tokens"] = 400
    else:
        kwargs["max_tokens"] = 300
        kwargs["temperature"] = 0.3
    resp = client.chat.completions.create(**kwargs)
    raw = (resp.choices[0].message.content or "").strip()
    facets = []
    for line in raw.splitlines():
        line = line.strip().lstrip("-*•").lstrip("0123456789. )").strip().strip('"').strip("'")
        if line and len(line) > 2:
            facets.append(line)
    if not facets:
        return [query]  # fallback: use original
    return facets[:5]  # cap at 5


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sample_path", default="data/response_eval_sample.json")
    p.add_argument("--output_dir", default="outputs_response_eval/divergent_rewrite")
    p.add_argument("--model", default="gpt-5.4-mini")
    p.add_argument("--concurrency", type=int, default=8)
    args = p.parse_args()

    with open(args.sample_path) as f:
        sampled = json.load(f)

    unique: set[str] = set()
    for style, items in sampled["samples"].items():
        for it in items:
            q = it.get("composed_query") if style == "composed" else it.get("query")
            if q:
                unique.add(q)
    print(f"Unique queries: {len(unique)}", file=sys.stderr)

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "rewrites_divergent.json")
    cache: Dict[str, List[str]] = {}
    if os.path.exists(out_path):
        cache = json.load(open(out_path))
        print(f"Loaded {len(cache)} cached", file=sys.stderr)

    todo = [q for q in unique if q not in cache]
    print(f"To rewrite: {len(todo)}", file=sys.stderr)
    if not todo:
        return

    client = _openai_client()
    lock = threading.Lock()

    def work(q: str):
        try:
            facets = _rewrite_one(client, args.model, q)
        except Exception:
            facets = [q]
        with lock:
            cache[q] = facets
            if len(cache) % 50 == 0:
                json.dump(cache, open(out_path, "w"), ensure_ascii=False, indent=2)

    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = [ex.submit(work, q) for q in todo]
        for _ in tqdm(as_completed(futs), total=len(futs), desc="divergent"):
            pass

    json.dump(cache, open(out_path, "w"), ensure_ascii=False, indent=2)
    print(f"Wrote {out_path} (n={len(cache)})", file=sys.stderr)

    # Stats: avg facets per query
    sizes = [len(v) for v in cache.values()]
    print(f"avg facets/query: {sum(sizes)/len(sizes):.2f}  min={min(sizes)} max={max(sizes)}", file=sys.stderr)


if __name__ == "__main__":
    main()
