"""Build multi-memory clusters from locomo10 evidence overlap.

For each sample, find pairs/triples of cat-1-4 QA items whose evidence dia_id
sets are RELATED (share at least 1 dia_id, but are not identical). The union
of their evidence is the gold cluster — a query that activates all of these
memories should retrieve all of them.

Then use an LLM to compose a single first-person utterance that requires the
union of memories to answer well.

Output: data/locomo10_multimem.json
  [
    {
      "sample_idx": int,
      "cluster_id": str,
      "member_q_idxs": [int, ...],   # indices into sample.qa
      "gold_dia_ids": [str, ...],    # union of evidence
      "member_facts": [               # for LLM rewrite + judge
        {"q_idx": ..., "question": ..., "answer": ..., "evidence": [...], "category": ...},
        ...
      ],
      "composed_query": str,
      "subject_speaker": str,
      "expected_memory_use": str,
      "rewrite_reason": str,
    },
    ...
  ]

Usage:
  python scripts/build_multimemory_clusters.py \
      --input data/locomo10.json \
      --output data/locomo10_multimem.json \
      --base_url <vllm> --model <name> \
      --cluster_size 2 --max_per_sample 30
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import combinations
from typing import Any, Dict, List, Set, Tuple

import httpx
from openai import OpenAI
from tqdm import tqdm


CLUSTER_PROMPT = """You compose a single first-person utterance that requires MULTIPLE memories from a user's past conversations to answer well.

Setting: The subject speaker is opening a fresh chat with an AI assistant that has memory of all past conversations. They send ONE message that should make the assistant draw on EVERY memory listed below to construct a good response.

Hard rules:
- The utterance must come from one speaker (speaker_a or speaker_b — pick whoever fits the memories), in first person.
- The utterance must NOT directly name the gold answers or quote evidence text.
- It should sound natural — a real situation, plan, decision, or reflection where ALL of the listed memories are relevant.
- It should NOT be a generic question that any memory could satisfy — only the listed memories together should fully address it.

Member memories (each is a Q+A from a past evaluation; the assistant should "use" the answer when responding):
{members_block}

speaker_a: {speaker_a}
speaker_b: {speaker_b}

Return STRICT JSON with keys:
  - subject_speaker ("speaker_a" or "speaker_b")
  - composed_query (the utterance)
  - expected_memory_use (one paragraph: how an ideal response would use ALL listed memories together)
  - reason (one sentence: why this utterance requires the memory union)
No preamble, no code fences."""


def find_clusters_in_sample(
    sample_qa: List[Dict[str, Any]],
    cluster_size: int = 2,
    max_per_sample: int = 30,
) -> List[Tuple[int, ...]]:
    """Find clusters of QA indices whose evidence sets overlap but differ.

    Returns list of tuples of qa_indices (length cluster_size).
    Filters: cat in 1..4; evidence list non-empty.
    """
    candidates: List[Tuple[int, Set[str]]] = []
    for i, qa in enumerate(sample_qa):
        try:
            cat = int(qa.get("category", 0))
        except (TypeError, ValueError):
            continue
        if cat not in (1, 2, 3, 4):
            continue
        ev = set(qa.get("evidence") or [])
        if not ev:
            continue
        candidates.append((i, ev))

    seen: Set[Tuple[int, ...]] = set()
    out: List[Tuple[int, ...]] = []
    for combo in combinations(candidates, cluster_size):
        idxs = tuple(sorted(c[0] for c in combo))
        if idxs in seen:
            continue
        ev_sets = [c[1] for c in combo]
        union = set().union(*ev_sets)
        # Require: pairwise overlap (each pair shares >=1 dia_id) BUT not all identical
        ok = True
        identical = True
        for a, b in combinations(ev_sets, 2):
            if a == b:
                continue
            identical = False
            if not (a & b):
                ok = False
                break
        # If all identical, useless cluster
        if identical:
            ok = False
        if not ok:
            continue
        # Require union size >= 2 for cluster to be "multi-memory"
        if len(union) < 2:
            continue
        seen.add(idxs)
        out.append(idxs)
        if len(out) >= max_per_sample:
            break
    return out


def compose_one(
    client: OpenAI,
    model: str,
    sample: Dict[str, Any],
    member_q_idxs: Tuple[int, ...],
) -> Dict[str, Any]:
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

    user_msg = CLUSTER_PROMPT.format(
        members_block=members_block, speaker_a=speaker_a, speaker_b=speaker_b
    )

    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": user_msg}],
        temperature=0.3,
        max_tokens=500,
    )
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
    p.add_argument("--input", default="data/locomo10.json")
    p.add_argument("--output", default="data/locomo10_multimem.json")
    p.add_argument("--base_url", default=os.environ.get(
        "LLM_BASE_URL",
        "http://localhost:8000/v1",
    ))
    p.add_argument("--model", default=os.environ.get("LLM_MODEL", "./gemma-4-31B-it"))
    p.add_argument("--api_key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    p.add_argument("--cluster_size", type=int, default=2,
                   help="number of QAs per cluster (2 = pairs, 3 = triples)")
    p.add_argument("--max_per_sample", type=int, default=30,
                   help="cap clusters per sample to keep runtime bounded")
    p.add_argument("--concurrency", type=int, default=2)
    args = p.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    safe_ua = {"User-Agent": "curl/8.0"}
    http_client = httpx.Client(headers=safe_ua, timeout=120.0)
    client = OpenAI(
        base_url=args.base_url,
        api_key=args.api_key,
        http_client=http_client,
        default_headers=safe_ua,
    )

    # Step 1: find clusters
    all_clusters: List[Dict[str, Any]] = []
    for s_idx, sample in enumerate(data):
        clusters = find_clusters_in_sample(sample.get("qa", []),
                                           cluster_size=args.cluster_size,
                                           max_per_sample=args.max_per_sample)
        for c_id, idxs in enumerate(clusters):
            all_clusters.append({
                "sample_idx": s_idx,
                "cluster_id": f"S{s_idx}_C{c_id}",
                "member_q_idxs": list(idxs),
                "_sample_ref": sample,
            })

    print(f"Discovered {len(all_clusters)} clusters across {len(data)} samples", file=sys.stderr)

    # Step 2: generate composed query per cluster
    results_lock = threading.Lock()
    out: List[Dict[str, Any]] = [None] * len(all_clusters)  # type: ignore[list-item]

    def worker(idx: int):
        c = all_clusters[idx]
        try:
            r = compose_one(client, args.model, c["_sample_ref"], tuple(c["member_q_idxs"]))
            entry = {
                "sample_idx": c["sample_idx"],
                "cluster_id": c["cluster_id"],
                **r,
                "rewrite_error": None,
            }
        except Exception as e:
            entry = {
                "sample_idx": c["sample_idx"],
                "cluster_id": c["cluster_id"],
                "member_q_idxs": c["member_q_idxs"],
                "gold_dia_ids": [],
                "member_facts": [],
                "composed_query": "",
                "subject_speaker": "",
                "subject_speaker_name": "",
                "expected_memory_use": "",
                "rewrite_reason": "",
                "rewrite_error": f"{type(e).__name__}: {e}",
            }
        with results_lock:
            out[idx] = entry

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [pool.submit(worker, i) for i in range(len(all_clusters))]
        for _ in tqdm(as_completed(futures), total=len(futures)):
            pass

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    n_ok = sum(1 for x in out if x and not x.get("rewrite_error"))
    n_err = sum(1 for x in out if x and x.get("rewrite_error"))
    by_sample: Dict[int, int] = {}
    for x in out:
        if x:
            by_sample[x["sample_idx"]] = by_sample.get(x["sample_idx"], 0) + 1
    print(f"Wrote {args.output}. ok={n_ok} err={n_err}", file=sys.stderr)
    print(f"  per-sample counts: {dict(sorted(by_sample.items()))}", file=sys.stderr)


if __name__ == "__main__":
    main()
