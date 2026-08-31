"""Iteratively re-compose leaky extension clusters until answer-token leakage
matches the original 300's distribution.

Leak metric: fraction of gold-answer tokens (len>3) appearing in the composed
query. Original 300 mean = 0.032; we regenerate any ext cluster with
leak > --threshold, feeding the model the exact forbidden words. Up to
--rounds attempts per cluster; keeps the best (lowest-leak) attempt.
"""
from __future__ import annotations
import argparse, importlib.util, json, os, sys, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx
from openai import OpenAI
from tqdm import tqdm


def _import_orig():
    p = Path(__file__).parent / "build_multimemory_clusters.py"
    spec = importlib.util.spec_from_file_location("build_mm_orig2", p)
    m = importlib.util.module_from_spec(spec)
    sys.modules["build_mm_orig2"] = m
    spec.loader.exec_module(m)
    return m


def answer_tokens(cluster):
    toks = set()
    for mf in cluster.get("member_facts", []):
        a = mf.get("answer")
        s = " ".join(str(x) for x in a) if isinstance(a, list) else str(a or "")
        toks.update(w for w in s.lower().split() if len(w) > 3)
    return toks


def leak_of(query, toks):
    if not toks:
        return 0.0
    q = (query or "").lower()
    return sum(1 for w in toks if w in q) / len(toks)


def recompose(client, model, orig_mod, sample, cluster, forbidden):
    conv = sample.get("conversation", {})
    speaker_a = conv.get("speaker_a", "Speaker A")
    speaker_b = conv.get("speaker_b", "Speaker B")
    members_block_parts = []
    for mf in cluster["member_facts"]:
        a = mf.get("answer")
        ans_str = ", ".join(str(x) for x in a) if isinstance(a, list) else str(a)
        members_block_parts.append(
            f"- Q: {mf.get('question','')}\n  A: {ans_str}\n  evidence: {mf.get('evidence', [])}"
        )
    members_block = "\n".join(members_block_parts)
    user_msg = orig_mod.CLUSTER_PROMPT.format(
        members_block=members_block, speaker_a=speaker_a, speaker_b=speaker_b
    )
    user_msg += (
        "\n\nADDITIONAL HARD CONSTRAINT: the composed_query must NOT contain any of these "
        "words (they reveal the gold answers): " + ", ".join(sorted(forbidden)[:60]) +
        ". Refer to things indirectly (e.g., 'that event we discussed', 'the plan I mentioned') "
        "instead of naming them."
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": user_msg}],
        max_completion_tokens=2000,
    )
    raw = (resp.choices[0].message.content or "").strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:].strip()
    parsed = json.loads(raw)
    return parsed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="baselines/AnchorMem/data/locomo10_multimem_ext.json")
    ap.add_argument("--locomo", default="baselines/AnchorMem/data/locomo10.json")
    ap.add_argument("--output", default="baselines/AnchorMem/data/locomo10_multimem_ext.json")
    ap.add_argument("--model", default="gpt-5.4-mini")
    ap.add_argument("--threshold", type=float, default=0.10)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--concurrency", type=int, default=12)
    args = ap.parse_args()

    orig_mod = _import_orig()
    data = json.load(open(args.locomo))
    clusters = json.load(open(args.input))

    kp = os.path.expanduser("~/.openai-key")
    api_key = open(kp).readline().strip() if os.path.exists(kp) else os.environ.get("OPENAI_API_KEY", "")
    client = OpenAI(api_key=api_key, http_client=httpx.Client(timeout=180.0))

    leaky_idx = []
    for i, c in enumerate(clusters):
        if not c.get("composed_query"): continue
        toks = answer_tokens(c)
        if leak_of(c["composed_query"], toks) > args.threshold:
            leaky_idx.append(i)
    print(f"Clusters over threshold {args.threshold}: {len(leaky_idx)} / {len(clusters)}", file=sys.stderr)

    lock = threading.Lock()
    fixed = {"n": 0}

    def worker(i):
        c = clusters[i]
        sample = data[c["sample_idx"]]
        toks = answer_tokens(c)
        best_q = c["composed_query"]
        best_leak = leak_of(best_q, toks)
        best_parsed = None
        for _ in range(args.rounds):
            forbidden = {w for w in toks if w in best_q.lower()} or toks
            try:
                parsed = recompose(client, args.model, orig_mod, sample, c, forbidden)
            except Exception:
                continue
            q = str(parsed.get("composed_query", "")).strip()
            lk = leak_of(q, toks)
            if q and lk < best_leak:
                best_q, best_leak, best_parsed = q, lk, parsed
            if best_leak <= args.threshold:
                break
        with lock:
            if best_parsed is not None:
                c["composed_query"] = best_q
                c["expected_memory_use"] = str(best_parsed.get("expected_memory_use", c.get("expected_memory_use", ""))).strip()
                c["rewrite_reason"] = str(best_parsed.get("reason", c.get("rewrite_reason", ""))).strip()
                subj = best_parsed.get("subject_speaker", c.get("subject_speaker"))
                if subj in ("speaker_a", "speaker_b"):
                    conv = sample.get("conversation", {})
                    c["subject_speaker"] = subj
                    c["subject_speaker_name"] = conv.get(subj, c.get("subject_speaker_name"))
                fixed["n"] += 1

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futs = [pool.submit(worker, i) for i in leaky_idx]
        for _ in tqdm(as_completed(futs), total=len(futs)):
            pass

    # Final stats
    leaks = []
    for c in clusters:
        if not c.get("composed_query"): continue
        leaks.append(leak_of(c["composed_query"], answer_tokens(c)))
    import statistics as st
    print(f"Re-composed: {fixed['n']}. Final leak mean: {st.mean(leaks):.3f}, "
          f"over-threshold remaining: {sum(1 for x in leaks if x > args.threshold)}", file=sys.stderr)

    json.dump(clusters, open(args.output, "w"), ensure_ascii=False, indent=2)
    print(f"Wrote {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
