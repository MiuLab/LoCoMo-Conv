"""E3-style per-dimension scoring for silent-grounding analysis.

For each (sample_idx, q_idx) in the silent-grounding pool (Oracle's
implicit fact_used = 0.0), score each of three response variants
independently on three binary-ish criteria:

  - faithfulness, relevance, engagement (each 0 / 0.5 / 1)

Variants:
  - oracle  : gold evidence turns
  - no_mem  : no memory items at all
  - random  : 3 random non-gold turns from same conversation

Output records carry the per-dim and total score for all three
variants, so pairwise margins (Oracle vs no_mem, Oracle vs random,
random vs no_mem) can be computed downstream.
"""
from __future__ import annotations
import argparse, json, os, sys, threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List

from anthropic import Anthropic
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_response_eval import all_speaker_turns, format_turns_block
from score_e3_cot_vs_oracle import SCORE_PROMPT, _parse_score


def _anthropic_client():
    kp = os.path.expanduser("~/.anthropic-key")
    key = open(kp).readline().strip() if os.path.exists(kp) else os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        raise RuntimeError("Anthropic API key missing")
    return Anthropic(api_key=key)


def score_one(client, model: str, query: str, memory_block: str, user_name: str, response: str):
    prompt = SCORE_PROMPT.format(
        user_name=user_name or "the user",
        query=query,
        memory_block=memory_block or "(no memory items)",
        response=response,
    )
    kw = {"model": model, "max_tokens": 200,
          "messages": [{"role": "user", "content": prompt}]}
    if not model.startswith("claude-opus-4-7"):
        kw["temperature"] = 0.0
    resp = client.messages.create(**kw)
    text = "".join(b.text for b in resp.content if hasattr(b, "text"))
    parsed = _parse_score(text)
    if parsed is None:
        return None, None
    total, dims = parsed
    return total, dims


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--oracle_responses_path", required=True)
    p.add_argument("--nomem_responses_path", required=True)
    p.add_argument("--random_responses_path", required=True)
    p.add_argument("--fact_used_path", required=True)
    p.add_argument("--sample_path", default="data/locomo10_dialog.json")
    p.add_argument("--output", required=True)
    p.add_argument("--judge_model", default="claude-opus-4-7")
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--checkpoint_every", type=int, default=50)
    args = p.parse_args()

    # 1. Build pool: implicit Oracle fact_used == 0.0
    fu = json.load(open(args.fact_used_path))
    pool = {(r["style"], r["sample_idx"], r["q_idx"]) for r in fu
            if r.get("fact_score") == 0.0 and r.get("variant") == "oracle"
            and r.get("style") == "implicit"}
    print(f"Pool (oracle implicit fact=0): {len(pool)}", file=sys.stderr)

    # 2. Load responses for each variant
    oracle_resp = json.load(open(args.oracle_responses_path))
    nomem_resp = json.load(open(args.nomem_responses_path))
    random_resp = json.load(open(args.random_responses_path))
    o_map = {(r["style"], r["sample_idx"], r["q_idx"]): r
             for r in oracle_resp if r.get("variant") == "oracle"}
    n_map = {(r["style"], r["sample_idx"], r["q_idx"]): r
             for r in nomem_resp if r.get("variant") == "no_memory"}
    r_map = {(r["style"], r["sample_idx"], r["q_idx"]): r
             for r in random_resp}

    dialog_data = json.load(open(args.sample_path))

    # 3. Build jobs
    jobs = []
    for k in pool:
        o = o_map.get(k); n = n_map.get(k); rd = r_map.get(k)
        if not o or not n or not rd: continue
        if not o.get("response") or not n.get("response") or not rd.get("response"): continue
        if rd.get("error"): continue
        # Oracle memory: gold evidence turns
        gold_ids = set(o.get("gold_evidence") or [])
        sample_obj = dialog_data[k[1]]
        all_t = all_speaker_turns(sample_obj)
        gold_turns = [t for t in all_t if t[0] in gold_ids]
        oracle_mem = format_turns_block(gold_turns) if gold_turns else "(no memory items)"
        # Random memory: random_turn_ids
        rand_ids = set(rd.get("random_turn_ids") or [])
        rand_turns = [t for t in all_t if t[0] in rand_ids]
        random_mem = format_turns_block(rand_turns) if rand_turns else "(no memory items)"
        # No memory
        nomem_mem = "(no memory items)"
        jobs.append({
            "style": k[0], "sample_idx": k[1], "q_idx": k[2],
            "query": o["query"],
            "user_name": o.get("subject_speaker_name") or "the user",
            "oracle_response": o["response"], "oracle_memory": oracle_mem,
            "nomem_response": n["response"], "nomem_memory": nomem_mem,
            "random_response": rd["response"], "random_memory": random_mem,
        })
    print(f"Jobs: {len(jobs)}", file=sys.stderr)

    # Resume
    out: List[Dict[str, Any]] = []
    done = set()
    if os.path.exists(args.output):
        out = json.load(open(args.output))
        for r in out:
            done.add((r["style"], r["sample_idx"], r["q_idx"]))
        print(f"Resuming, {len(done)} done", file=sys.stderr)
    pending = [j for j in jobs if (j["style"], j["sample_idx"], j["q_idx"]) not in done]
    print(f"Pending: {len(pending)}", file=sys.stderr)

    client = _anthropic_client()
    lock = threading.Lock()

    def worker(j):
        try:
            o_tot, o_dims = score_one(client, args.judge_model,
                j["query"], j["oracle_memory"], j["user_name"], j["oracle_response"])
            n_tot, n_dims = score_one(client, args.judge_model,
                j["query"], j["nomem_memory"], j["user_name"], j["nomem_response"])
            r_tot, r_dims = score_one(client, args.judge_model,
                j["query"], j["random_memory"], j["user_name"], j["random_response"])
            if o_tot is None or n_tot is None or r_tot is None:
                raise ValueError("score parse failed")
            entry = {
                "style": j["style"], "sample_idx": j["sample_idx"], "q_idx": j["q_idx"],
                "oracle_score": o_tot, "oracle_dims": o_dims,
                "nomem_score": n_tot,  "nomem_dims": n_dims,
                "random_score": r_tot, "random_dims": r_dims,
            }
        except Exception as ex:
            entry = {
                "style": j["style"], "sample_idx": j["sample_idx"], "q_idx": j["q_idx"],
                "error": f"{type(ex).__name__}: {str(ex)[:150]}",
            }
        with lock:
            out.append(entry)
            if len(out) % args.checkpoint_every == 0:
                with open(args.output + ".tmp", "w") as f:
                    json.dump(out, f, ensure_ascii=False, indent=2)
                os.replace(args.output + ".tmp", args.output)

    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = [ex.submit(worker, j) for j in pending]
        for _ in tqdm(as_completed(futs), total=len(futs), desc="e3-silent"):
            pass

    with open(args.output, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"Wrote {args.output} n={len(out)}", file=sys.stderr)

    # Aggregate: pairwise win/lose/tie per pair, per-dim margins
    dims = ("faithfulness", "relevance", "engagement")
    def margin(a_field, b_field):
        win = lose = tie = 0
        for r in out:
            if "error" in r: continue
            if a_field not in r or b_field not in r: continue
            if r[a_field] > r[b_field]: win += 1
            elif r[a_field] < r[b_field]: lose += 1
            else: tie += 1
        n = win + lose + tie
        return win, lose, tie, n
    def dim_margin(a_field, b_field, dim):
        win = lose = tie = 0
        for r in out:
            if "error" in r: continue
            a_dims = r.get(a_field); b_dims = r.get(b_field)
            if a_dims is None or b_dims is None: continue
            if a_dims[dim] > b_dims[dim]: win += 1
            elif a_dims[dim] < b_dims[dim]: lose += 1
            else: tie += 1
        n = win + lose + tie
        return win, lose, tie, n

    print()
    for a, b, label in [("oracle_score","nomem_score","Oracle vs no_mem"),
                        ("oracle_score","random_score","Oracle vs random"),
                        ("random_score","nomem_score","Random vs no_mem")]:
        w, l, t, n = margin(a, b)
        if n == 0: continue
        print(f"{label}: n={n}  A={w}({w/n:.1%})  B={l}({l/n:.1%})  T={t}({t/n:.1%})  margin={(w-l)/n*100:+.1f}pt")

    print()
    for a_field, b_field, label in [("oracle_dims","nomem_dims","Oracle - no_mem"),
                                    ("oracle_dims","random_dims","Oracle - random"),
                                    ("random_dims","nomem_dims","Random - no_mem")]:
        print(f"\n[{label}] per-dim win/lose/tie:")
        for dim in dims:
            w, l, t, n = dim_margin(a_field, b_field, dim)
            if n == 0: continue
            print(f"  {dim:<14} A={w/n:.1%}  B={l/n:.1%}  T={t/n:.1%}  margin={(w-l)/n*100:+.1f}pt")


if __name__ == "__main__":
    main()
