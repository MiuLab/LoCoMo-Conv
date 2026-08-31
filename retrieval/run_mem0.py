"""Run mem0 retrieval on all 5 styles + composed clusters, following mem0's
official locomo evaluation protocol (mem0/evaluation/src/memzero/add.py).

mem0 native locomo usage:
  - TWO user_ids per sample: speaker_a_<idx>, speaker_b_<idx>
  - Each speaker maintains their OWN memory of the entire conversation, from
    that speaker's POV (mirrored roles: my-messages=user, other=assistant)
  - Ingest in batches of 2 messages per add() call (mem0's default)
  - At retrieval, route the query to the SUBJECT speaker's user_id

We track each ingest batch's source dia_ids in metadata so retrieval can
recover dia_id provenance.

Outputs:
  outputs_mem0/locomo-<llm>/sample_<i>/queries_solutions_<style>.json
"""
from __future__ import annotations

import sys
import os
sys.path.append(".")

import multiprocessing as _mp
try:
    _mp.set_start_method("fork", force=True)
except RuntimeError:
    pass

import argparse
import json
import logging
from typing import Any, Dict, List, Tuple

import httpx


# Patch OpenAI SDK with curl/8.0 UA for Cloudflare WAF.
def _patch_openai_for_cloudflare():
    import openai
    safe_ua = {"User-Agent": "curl/8.0"}

    class _Patched(openai.OpenAI):
        def __init__(self, *a, **kw):
            http = httpx.Client(headers=safe_ua, timeout=120.0)
            kw.setdefault("http_client", http)
            kw["default_headers"] = {**(kw.get("default_headers") or {}), **safe_ua}
            super().__init__(*a, **kw)

    openai.OpenAI = _Patched


_patch_openai_for_cloudflare()

from mem0 import Memory  # noqa: E402


# Throttle add() calls to avoid Cloudflare sustained-rate limits.
# Override via MEM0_ADD_SLEEP_SEC env. Default 1.0 s between add() calls.
ADD_SLEEP_SEC = float(os.environ.get("MEM0_ADD_SLEEP_SEC", "1.0"))


# --- Session-date grounding -------------------------------------------------
# mem0 builds its extraction prompt with an "## Observation Date" section, and
# generate_additive_extraction_prompt() takes a `timestamp` argument for it —
# but Memory.add() never passes one, so _resolve_dates() falls back to
# datetime.now(). Every relative expression ("yesterday", "last year") is then
# resolved against the wall-clock date of the indexing run rather than the date
# the conversation took place, which corrupts every temporal memory.
#
# We plumb the session date through mem0's own parameter instead of altering
# its prompt: a module-level slot holds the current batch's date, and a thin
# wrapper injects it. No vendored mem0 source is modified.
_OBS_DATE: str | None = None

_MONTHS = {m: i for i, m in enumerate(
    ["January", "February", "March", "April", "May", "June",
     "July", "August", "September", "October", "November", "December"], 1)}


def _locomo_ts_to_iso(timestamp: str) -> str | None:
    """'1:56 pm on 8 May, 2023' -> '2023-05-08'."""
    import re
    m = re.search(r"(\d{1,2})\s+([A-Za-z]+),?\s+(\d{4})", timestamp or "")
    if not m:
        return None
    day, month, year = m.group(1), m.group(2), m.group(3)
    if month not in _MONTHS:
        return None
    return f"{year}-{_MONTHS[month]:02d}-{int(day):02d}"


def patch_mem0_observation_date():
    """Make mem0 resolve relative dates against the session date."""
    import mem0.memory.main as _main
    _orig = _main.generate_additive_extraction_prompt

    def _wrapped(*a, **kw):
        if _OBS_DATE and kw.get("timestamp") is None:
            kw["timestamp"] = _OBS_DATE
        return _orig(*a, **kw)

    _main.generate_additive_extraction_prompt = _wrapped


def _build_mem0_config(llm_base_url: str, llm_model: str, chroma_path: str,
                      embed_model: str = "all-MiniLM-L6-v2",
                      temperature: float = 0.1) -> dict:
    return {
        "llm": {
            "provider": "openai",
            "config": {
                "model": llm_model,
                "openai_base_url": llm_base_url,
                "api_key": "EMPTY",
                "temperature": temperature,
                "max_tokens": 1024,
            },
        },
        "embedder": {
            "provider": "huggingface",
            "config": {"model": embed_model},
        },
        "vector_store": {
            "provider": "chroma",
            "config": {"collection_name": "locomo_mem0", "path": chroma_path},
        },
    }


def build_speaker_views(sample: Dict[str, Any]):
    """Build two mirrored views of the conversation, one per speaker.

    Returns:
      speaker_a_name, speaker_b_name,
      List of (session_id, timestamp, messages_a_view, messages_b_view, batch_dia_ids)
      where messages_X_view is [{role,content}] with X's turns as 'user', the other's as 'assistant'.
      batch_dia_ids is the parallel list of dia_ids per message in original order.
    """
    conv = sample.get("conversation", {})
    speaker_a = conv.get("speaker_a", "Speaker A")
    speaker_b = conv.get("speaker_b", "Speaker B")
    sessions: List[Tuple[str, str, List[Dict[str, str]], List[Dict[str, str]], List[str]]] = []

    referenced_sessions = set()
    for qa in sample.get("qa", []):
        for ev in qa.get("evidence") or []:
            if isinstance(ev, str) and ":" in ev and ev.startswith("D"):
                referenced_sessions.add(ev.split(":", 1)[0])

    for k, v in conv.items():
        if not (isinstance(k, str) and k.startswith("session_") and "date" not in k):
            continue
        if not isinstance(v, list) or not v:
            continue
        first_dia = str(v[0].get("dia_id", ""))
        sid = first_dia.split(":", 1)[0] if ":" in first_dia else f"D{k.split('_',1)[1]}"
        if sid not in referenced_sessions:
            continue
        timestamp = conv.get(k + "_date_time", "")
        msgs_a, msgs_b, dia_ids = [], [], []
        for turn in v:
            spk = str(turn.get("speaker", "")).strip()
            txt = str(turn.get("text", "")).strip()
            did = turn.get("dia_id")
            if not (spk and txt and did):
                continue
            if spk == speaker_a:
                msgs_a.append({"role": "user", "content": f"{spk}: {txt}"})
                msgs_b.append({"role": "assistant", "content": f"{spk}: {txt}"})
            elif spk == speaker_b:
                msgs_a.append({"role": "assistant", "content": f"{spk}: {txt}"})
                msgs_b.append({"role": "user", "content": f"{spk}: {txt}"})
            else:
                # Unknown speaker — skip
                continue
            dia_ids.append(did)
        if msgs_a:
            sessions.append((sid, timestamp, msgs_a, msgs_b, dia_ids))
    return speaker_a, speaker_b, sessions


def add_for_speaker(memory: Memory, user_id: str, messages: List[Dict[str, str]],
                    dia_ids: List[str], timestamp: str, session_id: str,
                    batch_size: int = 2, sleep_sec: float = 0.0):
    import time as _time
    global _OBS_DATE
    _OBS_DATE = _locomo_ts_to_iso(timestamp)
    for i in range(0, len(messages), batch_size):
        batch_msgs = messages[i : i + batch_size]
        batch_dias = dia_ids[i : i + batch_size]
        meta = {
            "timestamp": timestamp,
            "session_id": session_id,
            "dia_ids": ",".join(batch_dias),
        }
        try:
            memory.add(messages=batch_msgs, user_id=user_id, metadata=meta)
        except Exception as e:
            logging.getLogger("mem0_runner").warning(
                f"  add failed user_id={user_id} session={session_id} batch_start={i}: {e}"
            )
        if sleep_sec > 0:
            _time.sleep(sleep_sec)


def select_qas_in_main_order(sample_qa: List[Dict[str, Any]], query_field: str):
    selected = []
    for q_idx, qa in enumerate(sample_qa):
        try:
            cat = int(qa.get("category", 0))
        except (TypeError, ValueError):
            cat = 0
        if cat == 5 and query_field == "question":
            continue
        if cat == 5 and query_field == "counterfactual_query":
            continue
        if query_field == "question":
            qstr = qa.get("question") or ""
        elif query_field == "dialog_query":
            qstr = qa.get("dialog_query") or ""
        elif query_field == "implicit_query":
            qstr = qa.get("implicit_query") or ""
        elif query_field == "counterfactual_query":
            qstr = qa.get("counterfactual_query") or ""
        else:
            qstr = ""
        if not qstr:
            continue
        selected.append((q_idx, qa))
    return selected


def _subject_user_id(qa: Dict[str, Any], sample_idx: int, speaker_a: str, speaker_b: str,
                     style_field: str) -> str:
    """Pick which speaker's memory to search based on the QA's subject_speaker."""
    # styles store their subject_speaker separately:
    #   dialog_query → subject_speaker
    #   implicit_query → implicit_subject_speaker
    #   counterfactual_query → counterfactual_subject_speaker
    # question style: no subject info; default to speaker_a (matches mem0 official: ingests both anyway)
    key_map = {
        "dialog_query": "subject_speaker",
        "implicit_query": "implicit_subject_speaker",
        "counterfactual_query": "counterfactual_subject_speaker",
        "question": "subject_speaker",
    }
    subj_key = qa.get(key_map.get(style_field, "subject_speaker")) or "speaker_a"
    name = speaker_a if subj_key == "speaker_a" else speaker_b
    return f"{name}_{sample_idx}"


def run_sample(memory: Memory, sample: Dict[str, Any], sample_idx: int,
               styles: List[str], sample_save_dir: str,
               multimem_clusters: List[Dict[str, Any]],
               top_k: int = 10, search_only: bool = False) -> None:
    logger = logging.getLogger("mem0_runner")
    speaker_a, speaker_b, sessions = build_speaker_views(sample)

    user_id_a = f"{speaker_a}_{sample_idx}"
    user_id_b = f"{speaker_b}_{sample_idx}"

    if not search_only:
        # Reset both
        for uid in [user_id_a, user_id_b]:
            try:
                memory.delete_all(user_id=uid)
            except Exception:
                pass

        total_msgs = sum(len(s[2]) for s in sessions)
        logger.info(f"[sample {sample_idx}] {len(sessions)} sessions, {total_msgs} msgs/speaker × 2 speakers")

        for sid, ts, msgs_a, msgs_b, dia_ids in sessions:
            add_for_speaker(memory, user_id_a, msgs_a, dia_ids, ts, sid, sleep_sec=ADD_SLEEP_SEC)
            add_for_speaker(memory, user_id_b, msgs_b, dia_ids, ts, sid, sleep_sec=ADD_SLEEP_SEC)
    else:
        logger.info(f"[sample {sample_idx}] search_only: reusing persisted index")

    field_for_style = {"question": "question", "dialog": "dialog_query",
                       "implicit": "implicit_query", "counterfactual": "counterfactual_query"}
    for style in styles:
        if style == "composed":
            continue
        qfield = field_for_style.get(style)
        if qfield is None:
            continue
        ordered = select_qas_in_main_order(sample.get("qa", []), qfield)
        if not ordered:
            continue
        out_entries = []
        for q_idx, qa in ordered:
            qstr = qa.get(qfield) or ""
            uid = _subject_user_id(qa, sample_idx, speaker_a, speaker_b, qfield)
            try:
                res = memory.search(query=qstr, filters={"user_id": uid}, limit=top_k)
                items = res.get("results", []) if isinstance(res, dict) else res
            except Exception as e:
                logger.warning(f"[sample {sample_idx}] search failed q{q_idx} uid={uid}: {e}")
                items = []
            entry = {
                "q_idx": q_idx,
                "question": qa.get("question", ""),
                "query": qstr,
                "subject_user_id": uid,
                "category": int(qa.get("category", 0)),
                "evidence": qa.get("evidence", []),
                "answer": qa.get("answer"),
                "docs": [(it.get("memory") or it.get("text") or "") for it in items],
                "retrieved_metadata": [it.get("metadata", {}) for it in items],
                "scores": [it.get("score") for it in items],
            }
            out_entries.append(entry)
        out_path = os.path.join(sample_save_dir, f"queries_solutions_{qfield}.json")
        with open(out_path, "w") as f:
            json.dump(out_entries, f, ensure_ascii=False, indent=2)
        logger.info(f"[sample {sample_idx}] {style}: wrote {len(out_entries)} → {out_path}")

    if "composed" in styles and multimem_clusters:
        clusters = [c for c in multimem_clusters if c.get("sample_idx") == sample_idx and c.get("composed_query")]
        if clusters:
            out_entries = []
            for c in clusters:
                # composed clusters store subject_speaker too
                subj_key = c.get("subject_speaker") or "speaker_a"
                uid = user_id_a if subj_key == "speaker_a" else user_id_b
                try:
                    res = memory.search(query=c["composed_query"], filters={"user_id": uid}, limit=top_k)
                    items = res.get("results", []) if isinstance(res, dict) else res
                except Exception as e:
                    items = []
                entry = {
                    "cluster_id": c["cluster_id"],
                    "sample_idx": c["sample_idx"],
                    "subject_user_id": uid,
                    "member_q_idxs": c["member_q_idxs"],
                    "gold_dia_ids": c["gold_dia_ids"],
                    "composed_query": c["composed_query"],
                    "docs": [(it.get("memory") or it.get("text") or "") for it in items],
                    "retrieved_metadata": [it.get("metadata", {}) for it in items],
                    "scores": [it.get("score") for it in items],
                }
                out_entries.append(entry)
            out_path = os.path.join(sample_save_dir, "composed_solutions.json")
            with open(out_path, "w") as f:
                json.dump(out_entries, f, ensure_ascii=False, indent=2)
            logger.info(f"[sample {sample_idx}] composed: wrote {len(out_entries)} → {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_path", default="data/locomo10_dialog.json")
    p.add_argument("--multimem_path", default="data/locomo10_multimem.json")
    p.add_argument("--llm_base_url", default=os.environ.get(
        "LLM_BASE_URL",
        "http://localhost:8000/v1",
    ))
    p.add_argument("--llm_model", default=os.environ.get("LLM_MODEL", "./gemma-4-31B-it"))
    p.add_argument("--embed_model", default="all-MiniLM-L6-v2")
    p.add_argument("--save_dir", default="outputs_mem0")
    p.add_argument("--styles", default="question,dialog,implicit,counterfactual,composed")
    p.add_argument("--samples", default="all")
    p.add_argument("--top_k", type=int, default=10)
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--temperature", type=float, default=0.1, help="LLM temperature for mem0 fact extraction")
    p.add_argument("--save_dir_suffix", default="", help="Suffix appended to base_save_dir (e.g. _seed43_t07)")
    p.add_argument("--search_only", action="store_true",
                   help="Skip ingestion; run retrieval against the persisted per-sample chroma index.")
    p.add_argument("--chroma_override", default="",
                   help="Use this chroma dir instead of the per-sample default (for legacy shared-chroma runs).")
    args = p.parse_args()

    import random as _rnd
    _rnd.seed(args.seed)

    llm_name_for_path = args.llm_model.lstrip("./").replace("/", "_")
    base_save_dir = os.path.join(args.save_dir, f"locomo-{llm_name_for_path}{args.save_dir_suffix}")
    os.makedirs(base_save_dir, exist_ok=True)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger("mem0_runner")

    with open(args.dataset_path) as f:
        dialog_data = json.load(f)
    mm: List[Dict[str, Any]] = []
    if "composed" in args.styles and os.path.exists(args.multimem_path):
        with open(args.multimem_path) as f:
            mm = json.load(f)

    if args.samples == "all":
        sample_indices = list(range(len(dialog_data)))
    else:
        sample_indices = [int(x) for x in args.samples.split(",") if x.strip()]

    styles = [s.strip() for s in args.styles.split(",") if s.strip()]
    field_for_style = {"question": "question", "dialog": "dialog_query",
                       "implicit": "implicit_query", "counterfactual": "counterfactual_query"}

    for s_idx in sample_indices:
        sample_dir = os.path.join(base_save_dir, f"sample_{s_idx}")
        os.makedirs(sample_dir, exist_ok=True)
        expected = []
        for st in styles:
            if st == "composed":
                expected.append(os.path.join(sample_dir, "composed_solutions.json"))
            else:
                f_name = field_for_style.get(st)
                if f_name:
                    expected.append(os.path.join(sample_dir, f"queries_solutions_{f_name}.json"))
        if expected and all(os.path.exists(p) for p in expected):
            logger.info(f"[sample {s_idx}] all outputs present, skipping")
            continue

        # Per-sample chroma dir so parallel processes on different samples don't race.
        chroma_path = args.chroma_override or os.path.join(sample_dir, "_chroma_state")
        os.makedirs(chroma_path, exist_ok=True)
        logger.info(f"[sample {s_idx}] initializing mem0 (LLM={args.llm_model} @ {args.llm_base_url})")
        cfg = _build_mem0_config(args.llm_base_url, args.llm_model, chroma_path, args.embed_model, temperature=args.temperature)
        memory = Memory.from_config(cfg)

        sample = dialog_data[s_idx]
        run_sample(memory, sample, s_idx, styles, sample_dir, mm, top_k=args.top_k,
                   search_only=args.search_only)

    logger.info("All samples done.")


if __name__ == "__main__":
    main()
