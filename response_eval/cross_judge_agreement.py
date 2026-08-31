"""Cross-judge agreement: replay validate_rewrites.py Stage-A prompts on a
random sample using a SECOND judge (gpt-4.1-mini) and compute agreement vs
the qwen verdict stored in *_validated.json.

Output:
  - per-style raw agreement
  - per-style Cohen's kappa
  - confusion matrices
  - per-item details written to cross_judge_results.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import threading
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Set

import httpx
from openai import OpenAI
from tqdm import tqdm

# Reuse the prompts from validate_rewrites.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from validate_rewrites import JUDGE_PROMPTS, _strip_fences, _evidence_text_for


def make_openai_client() -> OpenAI:
    key_path = os.path.expanduser("~/.openai-key")
    if os.path.exists(key_path):
        with open(key_path) as f:
            api_key = f.readline().strip()
    else:
        api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY missing (env or ~/.openai-key)")
    return OpenAI(api_key=api_key)


def judge_one(client: OpenAI, model: str, style: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    prompt = JUDGE_PROMPTS[style].format(**payload)
    # gpt-5.* "reasoning" models reject temperature != 1 and max_tokens; use
    # max_completion_tokens and default temperature.
    is_gpt5 = model.startswith("gpt-5") or model.startswith("o1") or model.startswith("o3")
    kwargs: Dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
    }
    if is_gpt5:
        kwargs["max_completion_tokens"] = 1024  # reasoning models need more headroom
    else:
        kwargs["temperature"] = 0.0
        kwargs["max_tokens"] = 300
    resp = client.chat.completions.create(**kwargs)
    raw = resp.choices[0].message.content or ""
    parsed = json.loads(_strip_fences(raw))
    verdict = str(parsed.get("verdict", "")).lower().strip()
    if verdict not in {"yes", "partial", "no"}:
        verdict = "partial"
    return {"verdict": verdict, "reason": str(parsed.get("reason", "")).strip()}


def cohen_kappa(a: List[str], b: List[str]) -> float:
    """Cohen's kappa for two equal-length sequences of categorical labels."""
    assert len(a) == len(b)
    n = len(a)
    if n == 0:
        return 0.0
    labels = sorted(set(a) | set(b))
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    pe = 0.0
    for lab in labels:
        p_a = a.count(lab) / n
        p_b = b.count(lab) / n
        pe += p_a * p_b
    if pe >= 1.0:
        return 1.0 if po == 1.0 else 0.0
    return (po - pe) / (1 - pe)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="data/locomo10_dialog_validated.json")
    p.add_argument("--multimem_input", default="data/locomo10_multimem_validated.json")
    p.add_argument("--output", default="data/cross_judge_results.json")
    p.add_argument("--model", default="gpt-4.1-mini")
    p.add_argument("--sample_per_style", type=int, default=150,
                   help="Sampled items per style for agreement test")
    p.add_argument("--styles", default="dialog,implicit,counterfactual,composed")
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    rng = random.Random(args.seed)
    styles = [s.strip() for s in args.styles.split(",") if s.strip()]

    client = make_openai_client()

    # Build pool of items per style with both rewrite and qwen verdict
    pool: Dict[str, List[Dict[str, Any]]] = {s: [] for s in styles}

    with open(args.input) as f:
        data = json.load(f)
    for s_idx, sample in enumerate(data):
        conv = sample.get("conversation", {})
        speaker_a = conv.get("speaker_a", "Speaker A")
        speaker_b = conv.get("speaker_b", "Speaker B")
        for q_idx, qa in enumerate(sample.get("qa", [])):
            cat = int(qa.get("category", 0))
            for style in ("dialog", "implicit", "counterfactual"):
                if style not in styles:
                    continue
                v = qa.get(f"validation_{style}")
                if not v:
                    continue
                rewrite_field = {"dialog": "dialog_query", "implicit": "implicit_query",
                                 "counterfactual": "counterfactual_query"}[style]
                rewrite = qa.get(rewrite_field)
                if not rewrite:
                    continue
                qwen_verdict = v.get("stage_a", {}).get("verdict")
                if not qwen_verdict:
                    continue
                answer_str = ", ".join(str(a) for a in qa["answer"]) if isinstance(qa.get("answer"), list) else str(qa.get("answer") or "")

                # Determine subject speaker name per style
                if style == "implicit":
                    subj_key = qa.get("implicit_subject_speaker") or qa.get("subject_speaker") or "speaker_a"
                elif style == "counterfactual":
                    subj_key = qa.get("counterfactual_subject_speaker") or qa.get("subject_speaker") or "speaker_a"
                else:
                    subj_key = qa.get("subject_speaker") or "speaker_a"
                subj_name = speaker_a if subj_key == "speaker_a" else speaker_b

                # cat 5 implicit uses different prompt + payload
                effective_style = style
                if style == "implicit" and cat == 5:
                    effective_style = "implicit_cat5"

                payload: Dict[str, Any] = {
                    "question": qa.get("question", ""),
                    "answer": answer_str,
                    "rewrite": rewrite,
                    "subject_speaker_name": subj_name,
                }
                if style == "counterfactual":
                    payload["wrong"] = qa.get("asserted_wrong", "")
                if effective_style == "implicit_cat5":
                    adv = qa.get("adversarial_answer")
                    payload["adversarial_answer"] = ", ".join(str(a) for a in adv) if isinstance(adv, list) else str(adv or "")

                pool[style].append({
                    "s_idx": s_idx, "q_idx": q_idx,
                    "style": style, "effective_style": effective_style,
                    "category": cat,
                    "qwen_verdict": qwen_verdict,
                    "payload": payload,
                })

    if "composed" in styles and os.path.exists(args.multimem_input):
        with open(args.multimem_input) as f:
            mm = json.load(f)
        for c_idx, cluster in enumerate(mm):
            v = cluster.get("validation_composed")
            if not v:
                continue
            if not cluster.get("composed_query"):
                continue
            members_block = "\n".join(
                f"- Q: {mf.get('question','')}\n  A: " +
                (", ".join(str(a) for a in mf.get('answer')) if isinstance(mf.get('answer'), list) else str(mf.get('answer') or ''))
                for mf in cluster.get("member_facts", [])
            )
            payload = {"members_block": members_block, "rewrite": cluster["composed_query"]}
            pool["composed"].append({
                "c_idx": c_idx,
                "style": "composed",
                "category": None,
                "qwen_verdict": v.get("stage_a", {}).get("verdict"),
                "payload": payload,
            })

    # Sample
    sampled: List[Dict[str, Any]] = []
    for style in styles:
        items = pool[style]
        n = min(args.sample_per_style, len(items))
        sampled.extend(rng.sample(items, n))
        print(f"  {style}: pool={len(items)}, sampled={n}", file=sys.stderr)

    print(f"Total cross-judge jobs: {len(sampled)} on {args.model}", file=sys.stderr)

    lock = threading.Lock()
    results: List[Dict[str, Any]] = [None] * len(sampled)  # type: ignore[list-item]

    def worker(idx: int):
        item = sampled[idx]
        # Use effective_style (handles cat 5 implicit → implicit_cat5)
        eff_style = item.get("effective_style", item["style"])
        try:
            r = judge_one(client, args.model, eff_style, item["payload"])
            gpt_verdict = r["verdict"]
            err = None
        except Exception as e:
            gpt_verdict = None
            r = {"verdict": None, "reason": ""}
            err = f"{type(e).__name__}: {e}"
        entry = {
            "style": item["style"],
            "category": item["category"],
            "qwen_verdict": item["qwen_verdict"],
            "gpt_verdict": gpt_verdict,
            "gpt_reason": r.get("reason", ""),
            "error": err,
        }
        # Keep identifying keys
        for k in ("s_idx", "q_idx", "c_idx"):
            if k in item:
                entry[k] = item[k]
        with lock:
            results[idx] = entry

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool_exec:
        futures = [pool_exec.submit(worker, i) for i in range(len(sampled))]
        for _ in tqdm(as_completed(futures), total=len(futures)):
            pass

    # Compute agreement per style
    summary: Dict[str, Any] = {"per_style": {}}
    for style in styles:
        rs = [r for r in results if r and r["style"] == style and r["gpt_verdict"]]
        if not rs:
            continue
        qwen_seq = [r["qwen_verdict"] for r in rs]
        gpt_seq = [r["gpt_verdict"] for r in rs]
        n = len(rs)
        agree = sum(1 for q, g in zip(qwen_seq, gpt_seq) if q == g) / n
        kappa = cohen_kappa(qwen_seq, gpt_seq)
        # Confusion matrix qwen -> gpt
        conf: Dict[str, Dict[str, int]] = {}
        for q, g in zip(qwen_seq, gpt_seq):
            conf.setdefault(q, {}).setdefault(g, 0)
            conf[q][g] += 1
        # Marginal
        qwen_dist = Counter(qwen_seq)
        gpt_dist = Counter(gpt_seq)
        summary["per_style"][style] = {
            "n": n,
            "raw_agreement": round(agree, 4),
            "cohen_kappa": round(kappa, 4),
            "qwen_distribution": dict(qwen_dist),
            "gpt_distribution": dict(gpt_dist),
            "confusion_qwen_to_gpt": conf,
        }

    out = {"summary": summary, "results": results, "judge_model": args.model,
           "sample_per_style": args.sample_per_style, "seed": args.seed}
    with open(args.output, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print("\n=== Cross-judge agreement summary ===", file=sys.stderr)
    print(json.dumps(summary, indent=2), file=sys.stderr)
    print(f"\nSaved details to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
