"""Generate response with RANDOM memory turns as control for silent grounding.

For each (sample_idx, q_idx) where oracle's partial fact_used = 0.0 on
implicit, sample N turns from the same sample's conversation
(excluding gold evidence turns) and generate a response using the same
answer prompt. Output mimics oracle responses.json schema so it can
plug into score_pairwise_silent.py.
"""
from __future__ import annotations
import argparse, json, os, random, sys, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List

import httpx
from openai import OpenAI
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_response_eval import all_speaker_turns, format_turns_block, ANSWER_PROMPT


def _speaker_name(sample, subj_key):
    conv = sample.get("conversation", {}) or {}
    return conv.get("speaker_a") if subj_key == "speaker_a" else conv.get("speaker_b")


def _chat(client: OpenAI, model: str, prompt: str, max_tokens: int = 800) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens, temperature=0.0,
    )
    return resp.choices[0].message.content or ""


_SAFE_UA = {"User-Agent": "curl/8.0"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pool_keys_path", required=True,
                   help="JSON list of {style, sample_idx, q_idx} cases to generate")
    p.add_argument("--base_responses_path", required=True,
                   help="oracle responses.json source for query/gold_evidence/speaker info")
    p.add_argument("--sample_path", default="data/locomo10_dialog.json")
    p.add_argument("--output", required=True)
    p.add_argument("--n_random", type=int, default=3, help="# of random turns to sample")
    p.add_argument("--answer_model", default="./gemma-4-31B-it")
    p.add_argument("--base_url", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--max_tokens", type=int, default=800)
    p.add_argument("--checkpoint_every", type=int, default=100)
    args = p.parse_args()

    keys_in = json.load(open(args.pool_keys_path))
    keyset = {(k["style"], k["sample_idx"], k["q_idx"]) for k in keys_in}
    print(f"Pool: {len(keyset)} cases", file=sys.stderr)

    base = json.load(open(args.base_responses_path))
    base_map = {}
    for r in base:
        if r.get("variant") != "oracle":
            continue
        base_map[(r["style"], r["sample_idx"], r["q_idx"])] = r

    dialog_data = json.load(open(args.sample_path))

    jobs = []
    for k in keyset:
        rec = base_map.get(k)
        if not rec:
            continue
        jobs.append(rec)
    print(f"Jobs: {len(jobs)}", file=sys.stderr)

    out: List[Dict[str, Any]] = []
    done = set()
    if os.path.exists(args.output):
        out = json.load(open(args.output))
        for r in out:
            done.add((r["style"], r["sample_idx"], r["q_idx"]))
        print(f"Resuming, {len(done)} done", file=sys.stderr)
    pending = [j for j in jobs if (j["style"], j["sample_idx"], j["q_idx"]) not in done]
    print(f"Pending: {len(pending)}", file=sys.stderr)

    client = OpenAI(api_key="EMPTY", base_url=args.base_url,
                    http_client=httpx.Client(headers=_SAFE_UA, timeout=180.0),
                    default_headers=_SAFE_UA)
    lock = threading.Lock()

    # Deterministic per-case random sampler (reproducible)
    base_rng = random.Random(args.seed)
    job_seeds = {}
    for j in jobs:
        job_seeds[(j["style"], j["sample_idx"], j["q_idx"])] = base_rng.randint(0, 2**31 - 1)

    def worker(rec):
        try:
            sample_obj = dialog_data[rec["sample_idx"]]
            subj = rec.get("subject_speaker_name") or _speaker_name(
                sample_obj, rec.get("subject_speaker", "speaker_a"))
            all_t = all_speaker_turns(sample_obj)
            gold_ids = set(rec.get("gold_evidence") or [])
            # Pool of candidate turns (exclude gold evidence)
            candidates = [t for t in all_t if t[0] not in gold_ids]
            if not candidates:
                raise ValueError("no candidate turns")
            sub_rng = random.Random(job_seeds[(rec["style"], rec["sample_idx"], rec["q_idx"])])
            picked = sub_rng.sample(candidates, k=min(args.n_random, len(candidates)))
            # Preserve conversation order
            picked.sort(key=lambda t: all_t.index(t))
            mem = format_turns_block(picked) if picked else "(no memory)"
            prompt = ANSWER_PROMPT.format(memory_block=mem, query=rec["query"], speaker=subj)
            ans = _chat(client, args.answer_model, prompt, max_tokens=args.max_tokens)
            entry = {**rec, "variant": "random_memory",
                     "response": ans, "prompt_len": len(prompt), "error": None,
                     "random_turn_ids": [t[0] for t in picked]}
        except Exception as e:
            entry = {**rec, "variant": "random_memory",
                     "response": "", "error": f"{type(e).__name__}: {e}"}
        with lock:
            out.append(entry)
            if len(out) % args.checkpoint_every == 0:
                with open(args.output + ".tmp", "w") as f:
                    json.dump(out, f, ensure_ascii=False, indent=2)
                os.replace(args.output + ".tmp", args.output)

    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = [ex.submit(worker, r) for r in pending]
        for _ in tqdm(as_completed(futs), total=len(futs), desc="random_mem"):
            pass

    with open(args.output, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"Wrote {args.output}  n={len(out)}", file=sys.stderr)


if __name__ == "__main__":
    main()
