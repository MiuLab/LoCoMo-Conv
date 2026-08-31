"""Regenerate oracle responses with surrounding ±N turns of context.

For each gold dia_id in `gold_evidence`, include the gold turn plus ±N
turns from the same sample (clipped at session boundaries — turns whose
session prefix `D<n>:` matches). De-duplicated and ordered.

Used to test whether oracle's failures are due to context starvation.
"""
from __future__ import annotations
import argparse, json, os, sys, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Set, Tuple

import httpx
from openai import OpenAI
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_response_eval import all_speaker_turns, format_turns_block, ANSWER_PROMPT


def _speaker_name(sample, subj_key):
    conv = sample.get("conversation", {}) or {}
    return conv.get("speaker_a") if subj_key == "speaker_a" else conv.get("speaker_b")

_SAFE_UA = {"User-Agent": "curl/8.0"}


def gold_evidence_turns_ctx(sample: Dict[str, Any], dia_ids: List[str], window: int = 5):
    """Return gold turns ± window neighbors, clipped at session boundary (same D<n>: prefix)."""
    all_t = all_speaker_turns(sample)
    if not all_t:
        return []
    idx_by_id = {t[0]: i for i, t in enumerate(all_t)}
    keep_idx: Set[int] = set()
    for did in dia_ids:
        if did not in idx_by_id:
            continue
        center = idx_by_id[did]
        center_session = did.split(":")[0]
        # Walk left
        for j in range(center - 1, max(-1, center - 1 - window), -1):
            if j < 0:
                break
            if all_t[j][0].split(":")[0] != center_session:
                break
            keep_idx.add(j)
        keep_idx.add(center)
        # Walk right
        for j in range(center + 1, min(len(all_t), center + 1 + window)):
            if all_t[j][0].split(":")[0] != center_session:
                break
            keep_idx.add(j)
    return [all_t[i] for i in sorted(keep_idx)]


def _chat(client: OpenAI, model: str, prompt: str, max_tokens: int = 800) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens, temperature=0.0,
    )
    return resp.choices[0].message.content or ""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--keys_path", required=True,
                   help="JSON file with list of {style, sample_idx, q_idx} to regen")
    p.add_argument("--base_responses_path", required=True,
                   help="responses.json to source gold_evidence/query/etc.")
    p.add_argument("--sample_path", default="data/locomo10_dialog.json")
    p.add_argument("--output", required=True)
    p.add_argument("--window", type=int, default=5)
    p.add_argument("--answer_model", default="./gemma-4-31B-it")
    p.add_argument("--base_url", required=True)
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--max_tokens", type=int, default=800)
    p.add_argument("--checkpoint_every", type=int, default=100)
    args = p.parse_args()

    keys_in = json.load(open(args.keys_path))
    keyset = {(k["style"], k["sample_idx"], k["q_idx"]) for k in keys_in}
    print(f"Keys to regen: {len(keyset)}", file=sys.stderr)

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
    print(f"Jobs (found in base): {len(jobs)}", file=sys.stderr)

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

    def worker(rec):
        try:
            sample_obj = dialog_data[rec["sample_idx"]]
            subj = rec.get("subject_speaker_name") or _speaker_name(
                sample_obj, rec.get("subject_speaker", "speaker_a"))
            turns = gold_evidence_turns_ctx(sample_obj, rec.get("gold_evidence") or [], window=args.window)
            mem = format_turns_block(turns) if turns else "(no evidence available)"
            prompt = ANSWER_PROMPT.format(memory_block=mem, query=rec["query"], speaker=subj)
            ans = _chat(client, args.answer_model, prompt, max_tokens=args.max_tokens)
            entry = {**rec, "variant": "oracle_ctx", "window": args.window,
                     "response": ans, "prompt_len": len(prompt), "error": None,
                     "n_turns_given": len(turns)}
        except Exception as e:
            entry = {**rec, "variant": "oracle_ctx", "window": args.window,
                     "response": "", "error": f"{type(e).__name__}: {e}"}
        with lock:
            out.append(entry)
            if len(out) % args.checkpoint_every == 0:
                with open(args.output + ".tmp", "w") as f:
                    json.dump(out, f, ensure_ascii=False, indent=2)
                os.replace(args.output + ".tmp", args.output)

    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = [ex.submit(worker, r) for r in pending]
        for _ in tqdm(as_completed(futs), total=len(futs), desc="oracle_ctx"):
            pass

    with open(args.output, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"Wrote {args.output}  n={len(out)}", file=sys.stderr)


if __name__ == "__main__":
    main()
