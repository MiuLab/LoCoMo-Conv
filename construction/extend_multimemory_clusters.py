"""Extend the composed multi-memory cluster set by lifting the max_per_sample cap.

- Discovers ALL valid clusters (same constraints as the original build:
  cat 1-4, evidence overlap >=1, not identical, union >= 2).
- Skips clusters already present in data/locomo10_multimem_full.json (matched by
  (sample_idx, sorted member_q_idxs)) — the original 300 stay untouched.
- Composes queries for the new clusters with gpt-5.4-mini (same CLUSTER_PROMPT).
- New cluster_ids continue per-sample numbering after the existing ones.

Outputs:
  data/locomo10_multimem_ext.json   — new clusters only
  data/locomo10_multimem_full.json  — original 300 + new (merged)
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
    spec = importlib.util.spec_from_file_location("build_mm_orig", p)
    m = importlib.util.module_from_spec(spec)
    sys.modules["build_mm_orig"] = m
    spec.loader.exec_module(m)
    return m


def compose_one_gpt5(client, model, sample, member_q_idxs, orig):
    """Same as orig.compose_one but uses max_completion_tokens for gpt-5 models."""
    conv = sample.get("conversation", {})
    speaker_a = conv.get("speaker_a", "Speaker A")
    speaker_b = conv.get("speaker_b", "Speaker B")

    members_block_parts = []
    member_facts = []
    for q_idx in member_q_idxs:
        qa = sample["qa"][q_idx]
        ans = qa.get("answer")
        ans_str = ", ".join(str(a) for a in ans) if isinstance(ans, list) else str(ans)
        members_block_parts.append(
            f"- Q: {qa.get('question','')}\n  A: {ans_str}\n  evidence: {qa.get('evidence', [])}"
        )
        member_facts.append({
            "q_idx": q_idx,
            "question": qa.get("question", ""),
            "answer": ans,
            "evidence": qa.get("evidence", []),
            "category": qa.get("category"),
        })
    members_block = "\n".join(members_block_parts)
    user_msg = orig.CLUSTER_PROMPT.format(
        members_block=members_block, speaker_a=speaker_a, speaker_b=speaker_b
    )

    kwargs = {
        "model": model,
        "messages": [{"role": "user", "content": user_msg}],
    }
    if model.startswith("gpt-5") or model.startswith("o"):
        kwargs["max_completion_tokens"] = 2000
    else:
        kwargs["max_tokens"] = 500
        kwargs["temperature"] = 0.3

    resp = client.chat.completions.create(**kwargs)
    raw = resp.choices[0].message.content or ""
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].strip()
    parsed = json.loads(text)

    subj_key = parsed.get("subject_speaker", "speaker_a")
    if subj_key not in {"speaker_a", "speaker_b"}:
        subj_key = "speaker_a"
    subj_name = speaker_a if subj_key == "speaker_a" else speaker_b
    union_evidence = sorted(set().union(*[set(mf["evidence"]) for mf in member_facts]))

    return {
        "member_q_idxs": list(member_q_idxs),
        "gold_dia_ids": union_evidence,
        "member_facts": member_facts,
        "composed_query": str(parsed.get("composed_query", "")).strip(),
        "subject_speaker": subj_key,
        "subject_speaker_name": subj_name,
        "expected_memory_use": str(parsed.get("expected_memory_use", "")).strip(),
        "rewrite_reason": str(parsed.get("reason", "")).strip(),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="baselines/AnchorMem/data/locomo10.json")
    p.add_argument("--existing", default="data/locomo10_multimem_full.json")
    p.add_argument("--output_ext", default="baselines/AnchorMem/data/locomo10_multimem_ext.json")
    p.add_argument("--output_full", default="data/locomo10_multimem_full.json")
    p.add_argument("--model", default="gpt-5.4-mini")
    p.add_argument("--concurrency", type=int, default=12)
    p.add_argument("--limit", type=int, default=0, help="cap NEW clusters (0 = all) for smoke tests")
    args = p.parse_args()

    orig = _import_orig()

    data = json.load(open(args.input))
    existing = json.load(open(args.existing))
    existing_keys = {(e["sample_idx"], tuple(sorted(e["member_q_idxs"]))) for e in existing}
    # per-sample max existing cluster number so new ids don't collide
    next_cid = {}
    for e in existing:
        n = int(e["cluster_id"].split("_C")[1])
        next_cid[e["sample_idx"]] = max(next_cid.get(e["sample_idx"], -1), n)

    # Discover full pool
    new_clusters = []
    for s_idx, sample in enumerate(data):
        idx_tuples = orig.find_clusters_in_sample(sample.get("qa", []),
                                                  cluster_size=2,
                                                  max_per_sample=10**9)
        for idxs in idx_tuples:
            key = (s_idx, tuple(sorted(idxs)))
            if key in existing_keys:
                continue
            next_cid[s_idx] = next_cid.get(s_idx, -1) + 1
            new_clusters.append({
                "sample_idx": s_idx,
                "cluster_id": f"S{s_idx}_C{next_cid[s_idx]}",
                "member_q_idxs": list(idxs),
                "_sample_ref": sample,
            })
    print(f"Existing: {len(existing)}   New candidates: {len(new_clusters)}", file=sys.stderr)
    if args.limit:
        new_clusters = new_clusters[: args.limit]
        print(f"Limited to {len(new_clusters)} for this run", file=sys.stderr)

    # OpenAI client
    kp = os.path.expanduser("~/.openai-key")
    api_key = open(kp).readline().strip() if os.path.exists(kp) else os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError("OpenAI API key missing")
    client = OpenAI(api_key=api_key, http_client=httpx.Client(timeout=180.0))

    lock = threading.Lock()
    out = [None] * len(new_clusters)

    def worker(i):
        c = new_clusters[i]
        try:
            r = compose_one_gpt5(client, args.model, c["_sample_ref"], tuple(c["member_q_idxs"]), orig)
            entry = {"sample_idx": c["sample_idx"], "cluster_id": c["cluster_id"], **r, "rewrite_error": None}
        except Exception as e:
            entry = {"sample_idx": c["sample_idx"], "cluster_id": c["cluster_id"],
                     "member_q_idxs": c["member_q_idxs"], "gold_dia_ids": [], "member_facts": [],
                     "composed_query": "", "subject_speaker": "", "subject_speaker_name": "",
                     "expected_memory_use": "", "rewrite_reason": "",
                     "rewrite_error": f"{type(e).__name__}: {e}"}
        with lock:
            out[i] = entry

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futs = [pool.submit(worker, i) for i in range(len(new_clusters))]
        for _ in tqdm(as_completed(futs), total=len(futs)):
            pass

    ok = [e for e in out if e and not e.get("rewrite_error") and e.get("composed_query")]
    err = [e for e in out if e and (e.get("rewrite_error") or not e.get("composed_query"))]
    print(f"Composed OK: {len(ok)}   errors: {len(err)}", file=sys.stderr)

    json.dump(out, open(args.output_ext, "w"), ensure_ascii=False, indent=2)
    merged = existing + [e for e in out if e]
    json.dump(merged, open(args.output_full, "w"), ensure_ascii=False, indent=2)
    print(f"Wrote {args.output_ext} ({len(out)}) and {args.output_full} ({len(merged)})", file=sys.stderr)


if __name__ == "__main__":
    main()
