"""Rewrite LoCoMo QA pairs into first-person dialog turns.

For each QA, the LLM is asked to:
  1. Pick the subject speaker (the participant the question is about).
  2. Rewrite the question as a natural utterance that speaker would say to an AI
     assistant who has been their long-term chat partner, prompting the assistant
     to retrieve the relevant memory and answer.
  3. Rate naturalness (high/medium/low). Multi-hop / inference questions that
     would sound forced as a single dialog turn should be rated 'low'.

Output preserves the original locomo10.json shape; each `qa` entry gains:
  - dialog_query          : str, the rewritten utterance
  - subject_speaker       : str, "speaker_a" | "speaker_b"
  - subject_speaker_name  : str, actual speaker name from conversation
  - naturalness           : "high" | "medium" | "low"
  - rewrite_reason        : str, short justification (LLM)
  - rewrite_error         : str | None, populated if the call failed

Usage:
  python scripts/rewrite_qa_to_dialog.py \
      --input data/locomo10.json \
      --output data/locomo10_dialog.json \
      --base_url http://localhost:5000/v1 \
      --model qwen3.6-35b-a3b \
      --concurrency 8
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List

import httpx
from openai import OpenAI
from tqdm import tqdm


SYSTEM_PROMPT = """You convert third-person QA pairs into natural first-person dialog turns (EXPLICIT memory queries).

Setting: The two people in `conversation` (speaker_a, speaker_b) have been chatting with an AI assistant for a long time. The assistant has memories of everything they've shared. Now one of them opens a fresh chat with the assistant and says ONE message that should naturally make the assistant look up the right memory and answer the original question.

Your job, for one QA at a time:
1. Choose the subject speaker — the participant the question is asking ABOUT (e.g. "What did Caroline research?" -> Caroline). The dialog turn must come from THAT speaker, addressed to the assistant in first person ("I", "me", "my").
2. Write a single natural utterance that this speaker would actually send as a DIRECT memory query (e.g. "Hey, do you remember when I painted that sunrise?", "Remind me what I was researching last spring?"). The utterance must NOT contain the answer, dates, or evidence specifics.
3. Rate naturalness:
   - "high"   = sounds like a real thing someone would say in chat
   - "medium" = slightly contrived but plausible
   - "low"    = a real person would NOT directly ask this. The information is the kind that comes up implicitly through context (advice-seeking, sharing a feeling, describing a situation), not through a direct "do you remember X" question. Examples that should be "low":
        * Self-knowledge questions like "How do I prioritize self-care?" — people don't ask AI to recite their own habits; they describe a situation and let the AI surface the routine.
        * Identity / values / preferences that one would share through context, not query.
        * Open-ended multi-hop reflections that feel like therapy prompts when asked directly.
4. Reason field:
   - For high/medium: one short sentence justifying naturalness.
   - For low: explain WHY the explicit phrasing fails AND hint at what implicit context would naturally surface this memory (e.g. "Asking the AI to list one's own self-care habits is unnatural; in real chat the speaker would describe feeling burnt out and the assistant would bring up their routine.").

Return STRICT JSON with keys: subject_speaker ("speaker_a" or "speaker_b"), dialog_query, naturalness, reason. No preamble, no code fences."""


COUNTERFACTUAL_PROMPT = """You convert third-person QA pairs into COUNTERFACTUAL first-person dialog turns.

Setting: The subject speaker is talking to an AI assistant that has memory of past conversations. The speaker now asserts a FACTUALLY WRONG version of something they previously shared, and either asks for confirmation or states it casually in passing. An ideal assistant would catch the inconsistency with stored memory and gently push back. We use this to test whether memory systems proactively correct user errors instead of going along.

Hard rules:
1. Choose the subject speaker (speaker_a or speaker_b) — the participant the original question is asking ABOUT. The utterance must come from THAT speaker, in first person.
2. Generate a PLAUSIBLE-WRONG counterfactual:
   - Cat 1 (single-hop fact): replace the gold fact with a plausible alternative of the same type.
       e.g. gold = "adoption agencies" → wrong = "law schools"
   - Cat 2 (temporal): shift the date by 2–5 years or change the month/season noticeably.
       e.g. gold = "May 2023" → wrong = "around 2018"
   - Cat 3 (inference / counterfactual): assert the OPPOSITE inference.
       e.g. gold = "would likely pursue counseling" → wrong "I think I'd actually go into engineering eventually"
   - Cat 4 (multi-hop fact): substitute one element of the chain with a plausible wrong (keep enough context).
3. Preserve enough CONTEXT so the assistant can reason:
   - Keep the topic anchor named (e.g. "that sunrise piece I painted", not just "that thing").
   - If the original question is temporal, include some time reference like "today" or "this week" so the assistant has a "now" anchor.
   - Keep first-person speaker pronouns ("I", "my").
4. Phrase the wrong assertion in a casual, conversational way — NOT a quiz. Examples:
   - "I was telling someone today I painted that sunrise like 6 years ago — that sounds about right yeah?"
   - "I told a friend yesterday I've been researching law schools — she thought that was random."
   - "Just told mom I'd probably end up in engineering eventually."
5. Do NOT mention the gold (correct) answer in the utterance. The whole point is the assistant must supply it from memory.

Return STRICT JSON with keys:
  - subject_speaker ("speaker_a" or "speaker_b")
  - counterfactual_query (the speaker's utterance)
  - asserted_wrong (a short phrase describing the wrong fact embedded in the query)
  - reason (one sentence on what makes this counterfactual plausible-yet-wrong)
No preamble, no code fences."""


IMPLICIT_PROMPT = """You convert third-person QA pairs into IMPLICIT first-person dialog turns.

Setting: The two people in `conversation` (speaker_a, speaker_b) have been chatting with an AI assistant for a long time. The assistant has memory of everything they've shared. Now the subject speaker opens a fresh chat and says ONE message. The message presents a real-life CONTEXT (a feeling, situation, problem, plan, decision they're facing) where the gold memory below would be the relevant thing for the assistant to recall, surface, or apply on its own initiative.

Hard rules:
- The utterance must NOT directly ask "do you remember…", "remind me…", "what did I say about…". It must NOT mention the answer or any evidence detail.
- The utterance should sound like the start of a normal conversation — a vent, a plan, a decision, a question about life — that a memory-aware assistant could respond to better by drawing on the gold memory.
- The expected_memory_use field describes what an ideal assistant response would look like: which memory it should surface, and how it would apply it (advise / remind / suggest / personalize). Be concrete.

Examples:
- Memory: "Melanie does running, reading, violin for self-care"
  implicit_query: "Ugh, I've been so stressed this week, I literally can't unwind."
  expected_memory_use: "Suggest the user try the self-care routines she's mentioned before — going for a run, reading, or playing violin — instead of giving generic stress tips."

- Memory: "Caroline is researching adoption agencies for the summer"
  implicit_query: "Trying to plan out what I'm doing the next few months and I feel kind of stuck."
  expected_memory_use: "Bring up the adoption-agency research she said was her summer focus, and help her break it into steps."

Return STRICT JSON with keys: subject_speaker ("speaker_a" or "speaker_b"), implicit_query, expected_memory_use, reason. No preamble, no code fences."""


USER_TEMPLATE = """speaker_a: {speaker_a}
speaker_b: {speaker_b}

Category: {category} ({category_desc})
Question: {question}
Gold answer: {answer}

Rewrite this as one first-person utterance from the subject speaker."""


CATEGORY_DESC = {
    1: "single-hop fact about the speaker",
    2: "temporal (when did X happen)",
    3: "inference / counterfactual about what the speaker would do",
    4: "open-ended multi-hop fact",
    5: "unanswerable — never mentioned in the conversation",
}


def _to_str(x: Any) -> str:
    if isinstance(x, list):
        return ", ".join(str(i) for i in x)
    return str(x)


def _chat_json(client: OpenAI, model: str, system_prompt: str, user_msg: str) -> Dict[str, Any]:
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": system_prompt + "\n\n---\n\n" + user_msg},
        ],
        temperature=0.3,
        max_tokens=400,
    )
    raw = resp.choices[0].message.content or ""
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].strip()
    return json.loads(text)


def rewrite_one(client: OpenAI, model: str, sample: Dict[str, Any], qa: Dict[str, Any]) -> Dict[str, Any]:
    conv = sample.get("conversation", {})
    speaker_a = conv.get("speaker_a", "Speaker A")
    speaker_b = conv.get("speaker_b", "Speaker B")
    try:
        cat: int = int(qa.get("category", 0))
    except (TypeError, ValueError):
        cat = 0
    user_msg = USER_TEMPLATE.format(
        speaker_a=speaker_a,
        speaker_b=speaker_b,
        category=cat,
        category_desc=CATEGORY_DESC.get(cat, "unknown"),
        question=qa.get("question", ""),
        answer=_to_str(qa.get("answer", "")),
    )

    parsed = _chat_json(client, model, SYSTEM_PROMPT, user_msg)

    subj_key = parsed.get("subject_speaker", "speaker_a")
    if subj_key not in {"speaker_a", "speaker_b"}:
        subj_key = "speaker_a"
    subj_name = speaker_a if subj_key == "speaker_a" else speaker_b

    return {
        "dialog_query": str(parsed.get("dialog_query", "")).strip(),
        "subject_speaker": subj_key,
        "subject_speaker_name": subj_name,
        "naturalness": parsed.get("naturalness", "medium"),
        "rewrite_reason": str(parsed.get("reason", "")).strip(),
        "rewrite_error": None,
    }


def rewrite_implicit_one(client: OpenAI, model: str, sample: Dict[str, Any], qa: Dict[str, Any]) -> Dict[str, Any]:
    conv = sample.get("conversation", {})
    speaker_a = conv.get("speaker_a", "Speaker A")
    speaker_b = conv.get("speaker_b", "Speaker B")
    try:
        cat: int = int(qa.get("category", 0))
    except (TypeError, ValueError):
        cat = 0
    user_msg = USER_TEMPLATE.format(
        speaker_a=speaker_a,
        speaker_b=speaker_b,
        category=cat,
        category_desc=CATEGORY_DESC.get(cat, "unknown"),
        question=qa.get("question", ""),
        answer=_to_str(qa.get("answer", "")),
    )
    # Pass through the explicit-pass reason so the model knows what was wrong.
    if qa.get("rewrite_reason"):
        user_msg += f"\n\nExplicit-pass note (why a direct ask was unnatural): {qa['rewrite_reason']}"
    parsed = _chat_json(client, model, IMPLICIT_PROMPT, user_msg)

    subj_key = parsed.get("subject_speaker") or qa.get("subject_speaker") or "speaker_a"
    if subj_key not in {"speaker_a", "speaker_b"}:
        subj_key = "speaker_a"
    subj_name = speaker_a if subj_key == "speaker_a" else speaker_b

    return {
        "implicit_query": str(parsed.get("implicit_query", "")).strip(),
        "implicit_subject_speaker": subj_key,
        "implicit_subject_speaker_name": subj_name,
        "expected_memory_use": str(parsed.get("expected_memory_use", "")).strip(),
        "implicit_reason": str(parsed.get("reason", "")).strip(),
        "implicit_error": None,
    }


def rewrite_counterfactual_one(client: OpenAI, model: str, sample: Dict[str, Any], qa: Dict[str, Any]) -> Dict[str, Any]:
    conv = sample.get("conversation", {})
    speaker_a = conv.get("speaker_a", "Speaker A")
    speaker_b = conv.get("speaker_b", "Speaker B")
    try:
        cat: int = int(qa.get("category", 0))
    except (TypeError, ValueError):
        cat = 0
    user_msg = USER_TEMPLATE.format(
        speaker_a=speaker_a,
        speaker_b=speaker_b,
        category=cat,
        category_desc=CATEGORY_DESC.get(cat, "unknown"),
        question=qa.get("question", ""),
        answer=_to_str(qa.get("answer", "")),
    )
    parsed = _chat_json(client, model, COUNTERFACTUAL_PROMPT, user_msg)

    subj_key = parsed.get("subject_speaker") or qa.get("subject_speaker") or "speaker_a"
    if subj_key not in {"speaker_a", "speaker_b"}:
        subj_key = "speaker_a"
    subj_name = speaker_a if subj_key == "speaker_a" else speaker_b

    return {
        "counterfactual_query": str(parsed.get("counterfactual_query", "")).strip(),
        "counterfactual_subject_speaker": subj_key,
        "counterfactual_subject_speaker_name": subj_name,
        "asserted_wrong": str(parsed.get("asserted_wrong", "")).strip(),
        "counterfactual_reason": str(parsed.get("reason", "")).strip(),
        "counterfactual_error": None,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="data/locomo10.json")
    p.add_argument("--output", default="data/locomo10_dialog.json")
    p.add_argument("--base_url", default=os.environ.get("LLM_BASE_URL", "http://localhost:8000/v1"))
    p.add_argument("--model", default=os.environ.get("LLM_MODEL", "./gemma-4-31B-it"))
    p.add_argument("--api_key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--samples", type=str, default="all", help="comma-separated sample indices, or 'all'")
    p.add_argument("--limit_per_sample", type=int, default=0, help="if >0, cap QAs per sample (for piloting)")
    p.add_argument("--resume", action="store_true", help="reuse already-rewritten entries from output file")
    p.add_argument("--implicit_pass", action="store_true",
                   help="Skip the explicit pass; instead, read --input as an already-rewritten file "
                        "and add implicit_query for every qa whose naturalness is 'low'.")
    p.add_argument("--implicit_naturalness", default="low",
                   help="Comma-separated naturalness values that trigger an implicit rewrite (default: low).")
    p.add_argument("--counterfactual_pass", action="store_true",
                   help="Generate counterfactual_query for every qa (skip cat 5).")
    args = p.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    existing: Dict[str, Dict[str, Any]] = {}
    if args.resume and os.path.exists(args.output):
        with open(args.output, "r", encoding="utf-8") as f:
            prior = json.load(f)
        for s_idx, sample in enumerate(prior):
            for q_idx, qa in enumerate(sample.get("qa", [])):
                if qa.get("dialog_query") and not qa.get("rewrite_error"):
                    existing[f"{s_idx}:{q_idx}"] = qa

    if args.samples == "all":
        sample_indices = list(range(len(data)))
    else:
        sample_indices = [int(x) for x in args.samples.split(",") if x.strip()]

    # Cloudflare WAF in front of the vLLM proxy blocks the OpenAI SDK User-Agent;
    # override it so requests aren't 403'd.
    safe_ua = {"User-Agent": "curl/8.0"}
    http_client = httpx.Client(headers=safe_ua, timeout=120.0)
    client = OpenAI(
        base_url=args.base_url,
        api_key=args.api_key,
        http_client=http_client,
        default_headers=safe_ua,
    )

    trigger_set = {x.strip() for x in args.implicit_naturalness.split(",") if x.strip()}

    # Build job list
    jobs: List[Dict[str, Any]] = []
    for s_idx in sample_indices:
        sample = data[s_idx]
        qa_list = sample.get("qa", [])
        if args.limit_per_sample > 0:
            qa_list_iter = qa_list[: args.limit_per_sample]
        else:
            qa_list_iter = qa_list
        for q_idx, qa in enumerate(qa_list_iter):
            if args.implicit_pass:
                if qa.get("naturalness") not in trigger_set:
                    continue
                if qa.get("implicit_query") and not qa.get("implicit_error"):
                    continue
            elif args.counterfactual_pass:
                # Cat 5 has no gold to be wrong about; skip.
                if int(qa.get("category", 0)) == 5:
                    continue
                if qa.get("counterfactual_query") and not qa.get("counterfactual_error"):
                    continue
            jobs.append({"s_idx": s_idx, "q_idx": q_idx, "sample": sample, "qa": qa})

    if args.implicit_pass:
        mode = "implicit"
    elif args.counterfactual_pass:
        mode = "counterfactual"
    else:
        mode = "explicit"
    print(f"Total jobs: {len(jobs)} (resume reuses {len(existing)}) mode={mode}", file=sys.stderr)

    results_lock = threading.Lock()
    # Mutable copy of data we will write into
    out_data = json.loads(json.dumps(data))  # deep copy

    def worker(job):
        key = f"{job['s_idx']}:{job['q_idx']}"
        if (not args.implicit_pass) and (not args.counterfactual_pass) and key in existing:
            return key, existing[key], None
        try:
            if args.implicit_pass:
                r = rewrite_implicit_one(client, args.model, job["sample"], job["qa"])
            elif args.counterfactual_pass:
                r = rewrite_counterfactual_one(client, args.model, job["sample"], job["qa"])
            else:
                r = rewrite_one(client, args.model, job["sample"], job["qa"])
        except Exception as e:
            if args.implicit_pass:
                r = {
                    "implicit_query": "",
                    "implicit_subject_speaker": "",
                    "implicit_subject_speaker_name": "",
                    "expected_memory_use": "",
                    "implicit_reason": "",
                    "implicit_error": f"{type(e).__name__}: {e}",
                }
            elif args.counterfactual_pass:
                r = {
                    "counterfactual_query": "",
                    "counterfactual_subject_speaker": "",
                    "counterfactual_subject_speaker_name": "",
                    "asserted_wrong": "",
                    "counterfactual_reason": "",
                    "counterfactual_error": f"{type(e).__name__}: {e}",
                }
            else:
                r = {
                    "dialog_query": "",
                    "subject_speaker": "",
                    "subject_speaker_name": "",
                    "naturalness": "low",
                    "rewrite_reason": "",
                    "rewrite_error": f"{type(e).__name__}: {e}",
                }
        # Merge with original qa fields
        merged = {**job["qa"], **r}
        with results_lock:
            out_data[job["s_idx"]]["qa"][job["q_idx"]] = merged
        return key, merged, None

    # Periodic checkpoint to disk so partial progress survives crashes / vLLM
    # outages. Set CHECKPOINT_EVERY env var to override.
    checkpoint_every = int(os.environ.get("CHECKPOINT_EVERY", "50"))
    completed_count = [0]

    def _flush_to_disk():
        with results_lock:
            tmp = args.output + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(out_data, f, ensure_ascii=False, indent=2)
            os.replace(tmp, args.output)

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [pool.submit(worker, j) for j in jobs]
        for _ in tqdm(as_completed(futures), total=len(futures)):
            completed_count[0] += 1
            if checkpoint_every > 0 and completed_count[0] % checkpoint_every == 0:
                _flush_to_disk()

    # Persist after each batch (also at end)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out_data, f, ensure_ascii=False, indent=2)

    # Quick stats
    if args.implicit_pass:
        n_impl = sum(1 for s in out_data for q in s.get("qa", []) if q.get("implicit_query"))
        n_err = sum(1 for s in out_data for q in s.get("qa", []) if q.get("implicit_error"))
        print(f"Wrote {args.output}. Implicit rewrites={n_impl}, errors={n_err}", file=sys.stderr)
    elif args.counterfactual_pass:
        n_cf = sum(1 for s in out_data for q in s.get("qa", []) if q.get("counterfactual_query"))
        n_err = sum(1 for s in out_data for q in s.get("qa", []) if q.get("counterfactual_error"))
        print(f"Wrote {args.output}. Counterfactual rewrites={n_cf}, errors={n_err}", file=sys.stderr)
    else:
        nat_counts = {"high": 0, "medium": 0, "low": 0, "error": 0}
        total = 0
        for sample in out_data:
            for qa in sample.get("qa", []):
                if "dialog_query" not in qa:
                    continue
                total += 1
                if qa.get("rewrite_error"):
                    nat_counts["error"] += 1
                else:
                    nat_counts[qa.get("naturalness", "medium")] = nat_counts.get(qa.get("naturalness", "medium"), 0) + 1
        print(f"Wrote {args.output}. Rewritten={total}. Naturalness: {nat_counts}", file=sys.stderr)


if __name__ == "__main__":
    main()
