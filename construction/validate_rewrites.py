"""Validate rewrites with two stages:

Stage A — LLM faithfulness judge (qwen, separate from generator gemma):
  Given (original Q, gold answer, rewrite, style), ask whether responding to
  the rewrite still REQUIRES the gold memory. Output: yes / partial / no.

Stage B — Programmatic checks (no LLM):
  - Answer-token leakage: how many of the gold-answer tokens appear in the rewrite
  - Evidence-text leakage: how many evidence-turn tokens appear in the rewrite
  - Length sanity (>=5 words for non-trivial framing)

Per-style thresholds:
  - dialog (Style 1): paraphrase, can mention topic; leak threshold lenient.
  - implicit (Style 2): must NOT mention answer; strict leak threshold.
  - counterfactual (Style 4): must include topic anchor (some leak OK) but NOT
    the actual gold; we look for the WRONG assertion to be present.
  - composed (Style 5, multimem): must NOT mention any of the gold facts.

Usage:
  python scripts/validate_rewrites.py \
      --input data/locomo10_dialog.json \
      --output data/locomo10_dialog_validated.json \
      --multimem_input data/locomo10_multimem.json \
      --multimem_output data/locomo10_multimem_validated.json \
      --judge_base_url <qwen> --judge_model qwen3.6-35b-a3b \
      --concurrency 2 \
      --styles dialog,implicit,counterfactual,composed
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Set

import httpx
from openai import OpenAI
from tqdm import tqdm


STOPWORDS = set(
    "a an the of in on at to for with by from is are was were be been being am has have had "
    "do does did doing this that these those it its and or but if then so as not no me my mine "
    "you your yours we us our ours they them their theirs he him his she her hers about "
    "what when where who whom whose how why which there here just very much more most some any "
    "would could should can will may might shall i'm you're we're they're it's i've i'll i'd".split()
)


def _tokenize(text: str) -> Set[str]:
    if not text:
        return set()
    text = text.lower()
    tokens = re.findall(r"[a-z0-9]+", text)
    return {t for t in tokens if t not in STOPWORDS and len(t) > 2}


def _overlap(rewrite_tokens: Set[str], target_tokens: Set[str]) -> float:
    """Fraction of target tokens that appear in rewrite."""
    if not target_tokens:
        return 0.0
    return len(rewrite_tokens & target_tokens) / len(target_tokens)


def _evidence_text_for(sample: Dict[str, Any], dia_ids: List[str]) -> str:
    """Concatenate text of the cited evidence turns, lowercased."""
    out = []
    for k, v in sample.get("conversation", {}).items():
        if not (isinstance(k, str) and k.startswith("session_") and "date" not in k):
            continue
        if not isinstance(v, list):
            continue
        for turn in v:
            if turn.get("dia_id") in dia_ids:
                out.append((turn.get("text") or "").strip())
    return " ".join(out)


# ---------- Stage A: LLM judge ----------

JUDGE_PROMPTS = {
    "dialog": """You are validating a paraphrased query.

Original question (third-person QA): {question}
Gold answer: {answer}
Subject speaker (the person the original question asks ABOUT, and who is now speaking the rewrite in first person): {subject_speaker_name}
Paraphrased query — note: 'I' / 'my' here refers to {subject_speaker_name}: {rewrite}

QUESTION FOR YOU (BINARY):
Reading 'I' as {subject_speaker_name}, would the gold answer be the RELEVANT memory to recall when responding to this paraphrase?

Respond STRICT JSON: {{"verdict": "yes" | "no", "reason": "<one short sentence>"}}.

- "yes" = the gold answer is the right memory to recall for this paraphrase.
- "no"  = the paraphrase does NOT ask for the gold; a different memory (or none) would be needed.""",

    "implicit": """You are validating an IMPLICIT query rewrite.

Original third-person QA: {question}
Gold memory the assistant might recall: {answer}
Subject speaker (the person making the utterance): {subject_speaker_name}
Rewrite — a situational utterance from {subject_speaker_name} (use 'I' / 'my' as {subject_speaker_name}): {rewrite}

QUESTION FOR YOU (BINARY):
Given this utterance from {subject_speaker_name}, would a memory-aware assistant naturally CONSIDER and DRAW ON the gold memory when crafting its response? Other memories may also be relevant; what we ask is whether the gold memory ADDS VALUE or is CONTEXTUALLY APPROPRIATE.

Respond STRICT JSON: {{"verdict": "yes" | "no", "reason": "<one short sentence>"}}.

- "yes" = the gold memory should be drawn on; it is relevant to the situation.
- "no"  = the gold memory is irrelevant or contradicts the situation; assistant should NOT bring it up.""",

    "implicit_cat5": """You are validating an IMPLICIT query for an UNANSWERABLE (cat-5) item.

Context: For this item, the speaker has NEVER previously mentioned the specific topic the question asks about. However, an "adversarial" memory exists in the corpus — a similar topic but about a DIFFERENT person — which a careless assistant might mistakenly attribute to the speaker.

Original third-person QA (about the speaker, but unanswerable): {question}
Adversarial memory (real corpus memory about someone ELSE, NOT the speaker): {adversarial_answer}
Subject speaker (the person making the rewrite utterance): {subject_speaker_name}
Rewrite — a situational utterance from {subject_speaker_name}: {rewrite}

QUESTION FOR YOU (BINARY):
Does this utterance set up a PLAUSIBLE TEMPTATION for the assistant to mistakenly attribute the adversarial memory to {subject_speaker_name}? In other words, would a less careful assistant likely fabricate a memory about {subject_speaker_name} based on the adversarial fact when responding to this utterance?

Respond STRICT JSON: {{"verdict": "yes" | "no", "reason": "<one short sentence>"}}.

- "yes" = the utterance creates plausible hallucination temptation (assistant might wrongly say "you mentioned X before").
- "no"  = the utterance is unrelated to the adversarial topic; no real hallucination test.""",

    "counterfactual": """You are validating a COUNTERFACTUAL query rewrite.

Original third-person QA: {question}
Gold answer (the truth): {answer}
Subject speaker (whose memory the assistant should recall to push back): {subject_speaker_name}
Rewrite — the speaker asserts a WRONG version, in first person ('I' = {subject_speaker_name}): {rewrite}
Asserted wrong fact (per generator): {wrong}

CHECK ALL THREE:
1. Does the rewrite contain a wrong assertion that CONTRADICTS the gold?
2. Does the rewrite preserve enough context (topic anchor, time reference, speaker disambiguation) for the assistant to recognize the inconsistency using stored memory?
3. Does the rewrite AVOID revealing the gold (correct) answer?

Respond STRICT JSON: {{"verdict": "yes" | "no", "contains_wrong_assertion": true|false, "preserves_context": true|false, "leaks_correct_answer": true|false, "reason": "<one short sentence>"}}.

- "yes" = all three conditions are met (wrong present, context preserved, no gold leak).
- "no"  = any condition fails.""",

    "composed": """You are validating a MULTI-MEMORY composed query.

The query is meant to need ALL of the following memories together. An ideal response should draw on every single one.

Listed memories:
{members_block}

Composed query: {rewrite}

QUESTION FOR YOU (BINARY):
Is responding to this query well NOT POSSIBLE without surfacing ALL of the listed memories? In other words, would a response missing any single one of these memories be clearly incomplete?

Would an ideal response to this query DRAW ON EVERY one of the listed memories? (A response that ignores any listed memory would be missing something.)

Respond STRICT JSON: {{"verdict": "yes" | "no", "uniquely_needs_all": true|false, "reason": "<one short sentence>"}}.

- "yes" = every listed memory is contextually needed; response is incomplete without any one.
- "no"  = at least one listed memory is unnecessary; the query could be answered using a strict subset.""",
}


def _strip_fences(text: str) -> str:
    text = (text or "").strip()
    m = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    return m.group(1) if m else text


def stage_a_judge(
    client: OpenAI,
    model: str,
    style: str,
    extra_body: Dict[str, Any],
    payload: Dict[str, Any],
) -> Dict[str, Any]:
    prompt = JUDGE_PROMPTS[style].format(**payload)
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=200,
        extra_body=extra_body,
    )
    raw = resp.choices[0].message.content or ""
    parsed = json.loads(_strip_fences(raw))
    verdict = str(parsed.get("verdict", "")).lower().strip()
    # Binary scheme: collapse anything that isn't "yes" to "no".
    if verdict != "yes":
        verdict = "no"
    out: Dict[str, Any] = {"verdict": verdict, "reason": str(parsed.get("reason", "")).strip()}
    # Style-specific extra fields
    for k in ("contains_wrong_assertion", "preserves_context", "leaks_correct_answer", "uniquely_needs_all"):
        if k in parsed:
            out[k] = bool(parsed[k])
    return out


# ---------- Stage B: programmatic checks ----------

def stage_b_check(
    style: str,
    rewrite: str,
    answer: str,
    evidence_text: str,
) -> Dict[str, Any]:
    rt = _tokenize(rewrite)
    at = _tokenize(answer)
    et = _tokenize(evidence_text)
    answer_overlap = _overlap(rt, at)
    evidence_overlap = _overlap(rt, et)
    word_count = len(re.findall(r"\b\w+\b", rewrite or ""))

    # Per-style thresholds
    if style == "dialog":
        leak_flag = answer_overlap > 0.6  # paraphrase, allow topic words; flag only blatant copy
        length_flag = word_count < 4
    elif style == "implicit":
        leak_flag = answer_overlap > 0.2  # strict — implicit must not name answer
        length_flag = word_count < 5
    elif style == "counterfactual":
        # CF should NOT contain gold answer tokens (they're trying to assert wrong)
        leak_flag = answer_overlap > 0.3
        length_flag = word_count < 6
    elif style == "composed":
        leak_flag = answer_overlap > 0.25  # strict — implicit-style multi
        length_flag = word_count < 8
    else:
        leak_flag = answer_overlap > 0.3
        length_flag = word_count < 5

    return {
        "answer_overlap": round(answer_overlap, 3),
        "evidence_overlap": round(evidence_overlap, 3),
        "word_count": word_count,
        "leak_flag": leak_flag,
        "length_flag": length_flag,
        "style_b_pass": (not leak_flag) and (not length_flag),
    }


# ---------- Pipeline ----------

def _make_client(base_url: str, api_key: str) -> OpenAI:
    safe_ua = {"User-Agent": "curl/8.0"}
    http = httpx.Client(headers=safe_ua, timeout=120.0)
    return OpenAI(base_url=base_url, api_key=api_key, http_client=http, default_headers=safe_ua)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="data/locomo10_dialog.json")
    p.add_argument("--output", default="data/locomo10_dialog_validated.json")
    p.add_argument("--multimem_input", default="data/locomo10_multimem.json")
    p.add_argument("--multimem_output", default="data/locomo10_multimem_validated.json")
    p.add_argument("--judge_base_url", default=os.environ.get(
        "JUDGE_BASE_URL",
        "http://localhost:8000/v1",
    ))
    p.add_argument("--judge_model", default=os.environ.get("JUDGE_MODEL", "qwen3.6-35b-a3b"))
    p.add_argument("--api_key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    p.add_argument("--concurrency", type=int, default=2)
    p.add_argument("--styles", default="dialog,implicit,counterfactual,composed")
    p.add_argument("--max_per_sample", type=int, default=0,
                   help="Cap items per sample (0 = no cap; pilot use only)")
    args = p.parse_args()

    styles = [s.strip() for s in args.styles.split(",") if s.strip()]

    client = _make_client(args.judge_base_url, args.api_key)
    extra_body = {"chat_template_kwargs": {"enable_thinking": False}}

    # ---- Build job list across data file (styles 1/2/4) ----
    with open(args.input) as f:
        data = json.load(f)

    jobs: List[Dict[str, Any]] = []
    for s_idx, sample in enumerate(data):
        qa_list = sample.get("qa", [])
        if args.max_per_sample > 0:
            qa_list = qa_list[: args.max_per_sample]
        for q_idx, qa in enumerate(qa_list):
            for style in styles:
                if style == "composed":
                    continue  # handled separately from multimem file
                if style == "dialog" and qa.get("dialog_query"):
                    rewrite = qa["dialog_query"]
                elif style == "implicit" and qa.get("implicit_query"):
                    rewrite = qa["implicit_query"]
                elif style == "counterfactual" and qa.get("counterfactual_query"):
                    rewrite = qa["counterfactual_query"]
                else:
                    continue
                if int(qa.get("category", 0)) == 5 and style == "counterfactual":
                    continue
                jobs.append({
                    "kind": "qa",
                    "s_idx": s_idx,
                    "q_idx": q_idx,
                    "style": style,
                    "rewrite": rewrite,
                    "qa": qa,
                    "sample": sample,
                })

    # ---- Multimem file ----
    mm_data: List[Dict[str, Any]] = []
    if "composed" in styles and os.path.exists(args.multimem_input):
        with open(args.multimem_input) as f:
            mm_data = json.load(f)
        for c_idx, cluster in enumerate(mm_data):
            if not cluster or not cluster.get("composed_query"):
                continue
            jobs.append({
                "kind": "cluster",
                "c_idx": c_idx,
                "style": "composed",
                "rewrite": cluster["composed_query"],
                "cluster": cluster,
                "sample": data[cluster["sample_idx"]],
            })

    print(f"Total validation jobs: {len(jobs)}", file=sys.stderr)

    # ---- Run ----
    results_lock = threading.Lock()
    qa_validations: Dict[str, Dict[str, Any]] = {}   # key "s:q:style" -> validation dict
    cluster_validations: Dict[int, Dict[str, Any]] = {}  # c_idx -> validation dict

    def worker(job):
        style = job["style"]
        if job["kind"] == "qa":
            qa = job["qa"]
            sample = job["sample"]
            cat = int(qa.get("category", 0))
            answer_str = ", ".join(str(a) for a in qa["answer"]) if isinstance(qa.get("answer"), list) else str(qa.get("answer") or "")
            evidence_text = _evidence_text_for(sample, qa.get("evidence", []))

            # Derive subject_speaker_name (the speaker assumed to be making the
            # rewrite utterance in first person).
            conv = sample.get("conversation", {})
            speaker_a = conv.get("speaker_a", "Speaker A")
            speaker_b = conv.get("speaker_b", "Speaker B")
            # Each style stores its own subject_speaker; fall back to dialog's.
            if style == "implicit":
                subj_key = qa.get("implicit_subject_speaker") or qa.get("subject_speaker") or "speaker_a"
            elif style == "counterfactual":
                subj_key = qa.get("counterfactual_subject_speaker") or qa.get("subject_speaker") or "speaker_a"
            else:
                subj_key = qa.get("subject_speaker") or "speaker_a"
            subj_name = speaker_a if subj_key == "speaker_a" else speaker_b

            # Choose effective style for prompt (cat 5 implicit uses a different rubric).
            effective_style = style
            if style == "implicit" and cat == 5:
                effective_style = "implicit_cat5"

            payload: Dict[str, Any] = {
                "question": qa.get("question", ""),
                "answer": answer_str,
                "rewrite": job["rewrite"],
                "subject_speaker_name": subj_name,
            }
            if style == "counterfactual":
                payload["wrong"] = qa.get("asserted_wrong", "")
            if effective_style == "implicit_cat5":
                adv = qa.get("adversarial_answer")
                payload["adversarial_answer"] = ", ".join(str(a) for a in adv) if isinstance(adv, list) else str(adv or "")

            try:
                a = stage_a_judge(client, args.judge_model, effective_style, extra_body, payload)
                a_err = None
            except Exception as e:
                a = {"verdict": "no", "reason": f"judge_error: {e}"}
                a_err = f"{type(e).__name__}: {e}"

            b = stage_b_check(style, job["rewrite"], answer_str, evidence_text)
            v = {"stage_a": a, "stage_b": b, "stage_a_error": a_err}
            v["overall_pass"] = (a["verdict"] == "yes") and b["style_b_pass"]
            with results_lock:
                qa_validations[f"{job['s_idx']}:{job['q_idx']}:{style}"] = v
        else:  # cluster
            cluster = job["cluster"]
            sample = job["sample"]
            members_block_parts = []
            all_answer_tokens: Set[str] = set()
            all_evidence_text_parts = []
            for mf in cluster.get("member_facts", []):
                ans = mf.get("answer")
                ans_str = ", ".join(str(a) for a in ans) if isinstance(ans, list) else str(ans or "")
                members_block_parts.append(f"- Q: {mf.get('question','')}\n  A: {ans_str}")
                all_answer_tokens |= _tokenize(ans_str)
                all_evidence_text_parts.append(_evidence_text_for(sample, mf.get("evidence", [])))

            members_block = "\n".join(members_block_parts)
            payload = {"members_block": members_block, "rewrite": job["rewrite"]}

            try:
                a = stage_a_judge(client, args.judge_model, "composed", extra_body, payload)
                a_err = None
            except Exception as e:
                a = {"verdict": "no", "reason": f"judge_error: {e}"}
                a_err = f"{type(e).__name__}: {e}"

            # Stage B for composed: combine all answers and all evidence
            combined_answer = " ".join(
                ", ".join(str(a) for a in (mf.get("answer") if isinstance(mf.get("answer"), list) else [mf.get("answer")]))
                for mf in cluster.get("member_facts", [])
            )
            combined_evidence = " ".join(all_evidence_text_parts)
            b = stage_b_check("composed", job["rewrite"], combined_answer, combined_evidence)
            v = {"stage_a": a, "stage_b": b, "stage_a_error": a_err}
            v["overall_pass"] = (a["verdict"] == "yes") and b["style_b_pass"]
            with results_lock:
                cluster_validations[job["c_idx"]] = v

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [pool.submit(worker, j) for j in jobs]
        for _ in tqdm(as_completed(futures), total=len(futures)):
            pass

    # ---- Annotate output files ----
    for s_idx, sample in enumerate(data):
        for q_idx, qa in enumerate(sample.get("qa", [])):
            for style in styles:
                key = f"{s_idx}:{q_idx}:{style}"
                if key in qa_validations:
                    qa[f"validation_{style}"] = qa_validations[key]
    with open(args.output, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Wrote {args.output}  (qa validations={len(qa_validations)})", file=sys.stderr)

    if mm_data:
        for c_idx, c in enumerate(mm_data):
            if c_idx in cluster_validations:
                c["validation_composed"] = cluster_validations[c_idx]
        with open(args.multimem_output, "w") as f:
            json.dump(mm_data, f, ensure_ascii=False, indent=2)
        print(f"Wrote {args.multimem_output}  (cluster validations={len(cluster_validations)})", file=sys.stderr)

    # Summary
    summary = {"by_style": {}}
    for key, v in qa_validations.items():
        style = key.split(":")[-1]
        d = summary["by_style"].setdefault(style, {"n": 0, "stage_a": {"yes": 0, "partial": 0, "no": 0}, "stage_b_pass": 0, "overall_pass": 0})
        d["n"] += 1
        d["stage_a"][v["stage_a"]["verdict"]] += 1
        d["stage_b_pass"] += int(v["stage_b"]["style_b_pass"])
        d["overall_pass"] += int(v["overall_pass"])
    if cluster_validations:
        d = summary["by_style"].setdefault("composed", {"n": 0, "stage_a": {"yes": 0, "partial": 0, "no": 0}, "stage_b_pass": 0, "overall_pass": 0})
        for v in cluster_validations.values():
            d["n"] += 1
            d["stage_a"][v["stage_a"]["verdict"]] += 1
            d["stage_b_pass"] += int(v["stage_b"]["style_b_pass"])
            d["overall_pass"] += int(v["overall_pass"])
    print("\n=== Validation summary ===", file=sys.stderr)
    print(json.dumps(summary, indent=2), file=sys.stderr)


if __name__ == "__main__":
    main()
