"""Run A-mem retrieval on all 5 styles + composed clusters.

A-mem's ingestion calls the LLM twice per memory (content analyze + evolution
decision). We:
  - Monkey-patch A-mem's OpenAIController to point at our gemma vLLM endpoint
  - Use the SAME chunking logic as AnchorMem (3-turn windows with overlap=1)
  - Track chunk_id -> [dia_id] mapping externally for retrieval-side dia_id
    recovery (mapping from search result content back to dia_ids)
  - Save outputs in mirror of AnchorMem's per-sample directory layout under
    outputs_amem/locomo-<llm>/sample_<i>/queries_solutions_<style>.json
"""
from __future__ import annotations

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "A-mem"))
sys.path.append(".")

import multiprocessing as _mp
try:
    _mp.set_start_method("fork", force=True)
except RuntimeError:
    pass

import argparse
import json
import logging
import random
from typing import Any, Dict, List, Optional, Tuple

import httpx
from openai import OpenAI

# Late import (after sys.path append above)
from agentic_memory import llm_controller as _amem_llm
from agentic_memory.memory_system import AgenticMemorySystem

from src.datasets.locomo10_loader import make_docs_from_locomo10_conversations


SEED = 42
random.seed(SEED)
DEFAULT_TEMPERATURE = 0.7  # overridden by --temperature at CLI parse time


# ---------- Monkey-patch OpenAIController to point at vLLM ----------

class _PatchedOpenAIController(_amem_llm.BaseLLMController):
    """OpenAI-compatible controller that talks to our gemma vLLM via Cloudflare proxy."""

    def __init__(self, model: str = "./gemma-4-31B-it", api_key: Optional[str] = None,
                 base_url: Optional[str] = None):
        self.model = model
        self.base_url = base_url or os.environ.get(
            "LLM_BASE_URL",
            "http://localhost:8000/v1",
        )
        safe_ua = {"User-Agent": "curl/8.0"}
        http_client = httpx.Client(headers=safe_ua, timeout=30.0)
        self.client = OpenAI(
            base_url=self.base_url,
            api_key=api_key or os.environ.get("OPENAI_API_KEY", "EMPTY"),
            http_client=http_client,
            default_headers=safe_ua,
            max_retries=0,
        )

    def get_completion(self, prompt: str, response_format: dict = None, temperature: float = None) -> str:
        if temperature is None:
            temperature = DEFAULT_TEMPERATURE
        # vLLM is more reliable with json_object than the full json_schema spec.
        rf: Dict[str, Any] = {"type": "json_object"}
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You must respond with a JSON object."},
                    {"role": "user", "content": prompt},
                ],
                response_format=rf,
                temperature=temperature,
                max_tokens=512,
            )
            return response.choices[0].message.content or "{}"
        except Exception as e:
            logging.warning(f"vLLM completion failed: {e}; falling back to empty JSON")
            return "{}"


# Replace the real OpenAIController class
_amem_llm.OpenAIController = _PatchedOpenAIController


def _disable_evolution(memsys):
    """Patch the memory system instance so process_memory never calls the LLM
    for evolution decisions. Used for the 'no-evolution' A-mem variant.
    """
    def _no_op_process(note):
        return False, note
    memsys.process_memory = _no_op_process


# ---------- Adapter ----------

def turns_with_dia_ids(sample: Dict[str, Any]) -> List[Tuple[str, List[str]]]:
    """A-mem's native granularity is one note per utterance / turn.

    Each item: (turn_text, [dia_id]). Only includes turns from sessions that
    are referenced by at least one QA's evidence (matching AnchorMem's filter).
    """
    out: List[Tuple[str, List[str]]] = []
    referenced_sessions: set = set()
    for qa in sample.get("qa", []):
        for ev in qa.get("evidence") or []:
            if isinstance(ev, str) and ev.startswith("D") and ":" in ev:
                referenced_sessions.add(ev.split(":", 1)[0])

    conv = sample.get("conversation", {})
    for k, v in conv.items():
        if not (isinstance(k, str) and k.startswith("session_") and "date" not in k):
            continue
        if not isinstance(v, list) or not v:
            continue
        first_dia = str(v[0].get("dia_id", ""))
        session_id = first_dia.split(":", 1)[0] if ":" in first_dia else f"D{k.split('_',1)[1]}"
        if session_id not in referenced_sessions:
            continue
        date = conv.get(k + "_date_time", "Unknown Date")

        for turn in v:
            spk = str(turn.get("speaker", "")).strip()
            txt = str(turn.get("text", "")).strip()
            did = turn.get("dia_id")
            if not (spk and txt and did):
                continue
            line = f'DATE: {date}\n{spk} said, "{txt}"'
            if "blip_caption" in turn:
                line += " and shared %s." % turn["blip_caption"]
            out.append((line, [did]))
    return out


# Keep the old name as a thin wrapper for backward compat.
chunks_with_dia_ids = turns_with_dia_ids


def select_qas_in_main_order(sample_qa: List[Dict[str, Any]], query_field: str) -> List[Tuple[int, Dict[str, Any]]]:
    selected: List[Tuple[int, Dict[str, Any]]] = []
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


def run_sample(memsys: AgenticMemorySystem, sample: Dict[str, Any], sample_idx: int,
               styles: List[str], sample_save_dir: str,
               multimem_clusters: List[Dict[str, Any]],
               top_k: int = 10) -> None:
    """Ingest the sample once, then run all styles."""
    logger = logging.getLogger("amem_runner")

    # ---- Reset memsys for this sample (fresh memory store) ----
    memsys.memories = {}
    try:
        memsys.retriever.client.reset()
    except Exception:
        pass
    from agentic_memory.retrievers import ChromaRetriever
    memsys.retriever = ChromaRetriever(collection_name="memories", model_name=memsys.model_name)

    # ---- Ingest ----
    chunks = turns_with_dia_ids(sample)
    logger.info(f"[sample {sample_idx}] ingesting {len(chunks)} turns (A-mem native: 1 note per turn)")
    note_id_to_dia: Dict[str, List[str]] = {}
    for chunk_text, dia_ids in chunks:
        try:
            note_id = memsys.add_note(content=chunk_text)
            note_id_to_dia[note_id] = dia_ids
        except Exception as e:
            logger.warning(f"  add_note failed: {e}")

    # Save the dia mapping for later post-processing
    map_path = os.path.join(sample_save_dir, "amem_note_to_dia.json")
    with open(map_path, "w") as f:
        json.dump(note_id_to_dia, f, ensure_ascii=False, indent=2)
    logger.info(f"[sample {sample_idx}] wrote {len(note_id_to_dia)} note→dia mappings")

    # ---- Run styles ----
    field_for_style = {
        "question": "question",
        "dialog": "dialog_query",
        "implicit": "implicit_query",
        "counterfactual": "counterfactual_query",
    }
    for style in styles:
        if style == "composed":
            continue
        qfield = field_for_style.get(style)
        if qfield is None:
            continue
        ordered = select_qas_in_main_order(sample.get("qa", []), qfield)
        if not ordered:
            continue
        out_entries: List[Dict[str, Any]] = []
        for q_idx, qa in ordered:
            qstr = qa.get(qfield) or ""
            try:
                results = memsys.search_agentic(qstr, k=top_k)
            except Exception as e:
                logger.warning(f"  search failed: {e}")
                results = []
            retrieved_note_ids = [r.get("id") for r in results if r and r.get("id")]
            retrieved_docs = [r.get("content", "") for r in results if r]
            entry = {
                "q_idx": q_idx,
                "question": qa.get("question", ""),
                "query": qstr,
                "category": int(qa.get("category", 0)),
                "evidence": qa.get("evidence", []),
                "answer": qa.get("answer"),
                "retrieved_note_ids": retrieved_note_ids,
                "docs": retrieved_docs,
            }
            out_entries.append(entry)
        out_path = os.path.join(sample_save_dir, f"queries_solutions_{qfield}.json")
        with open(out_path, "w") as f:
            json.dump(out_entries, f, ensure_ascii=False, indent=2)
        logger.info(f"[sample {sample_idx}] {style}: wrote {len(out_entries)} → {out_path}")

    # ---- Composed (multi-memory clusters) ----
    if "composed" in styles and multimem_clusters:
        sample_clusters = [c for c in multimem_clusters if c.get("sample_idx") == sample_idx and c.get("composed_query")]
        if sample_clusters:
            out_entries = []
            for c in sample_clusters:
                try:
                    results = memsys.search_agentic(c["composed_query"], k=top_k)
                except Exception as e:
                    logger.warning(f"  composed search failed: {e}")
                    results = []
                retrieved_note_ids = [r.get("id") for r in results if r and r.get("id")]
                retrieved_docs = [r.get("content", "") for r in results if r]
                entry = {
                    "cluster_id": c["cluster_id"],
                    "sample_idx": c["sample_idx"],
                    "member_q_idxs": c["member_q_idxs"],
                    "gold_dia_ids": c["gold_dia_ids"],
                    "composed_query": c["composed_query"],
                    "retrieved_note_ids": retrieved_note_ids,
                    "docs": retrieved_docs,
                }
                out_entries.append(entry)
            out_path = os.path.join(sample_save_dir, "composed_solutions.json")
            with open(out_path, "w") as f:
                json.dump(out_entries, f, ensure_ascii=False, indent=2)
            logger.info(f"[sample {sample_idx}] composed: wrote {len(out_entries)} → {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_path", default="data/locomo10_dialog.json")
    p.add_argument("--locomo_path", default="data/locomo10.json")
    p.add_argument("--multimem_path", default="data/locomo10_multimem_full.json")
    p.add_argument("--llm_model", default="./gemma-4-31B-it")
    p.add_argument("--embedding_model", default="all-MiniLM-L6-v2")
    p.add_argument("--evo_threshold", type=int, default=10000,
                   help="High value effectively disables periodic evolution consolidation.")
    p.add_argument("--save_dir", default="outputs_amem")
    p.add_argument("--styles", default="question,dialog,implicit,counterfactual,composed")
    p.add_argument("--samples", default="all", help="comma-separated sample indices or 'all'")
    p.add_argument("--top_k", type=int, default=10)
    p.add_argument("--seed", type=int, default=42, help="Random seed for module RNG and LLM sampling")
    p.add_argument("--temperature", type=float, default=0.7, help="LLM temperature during memory construction / linking")
    p.add_argument("--save_dir_suffix", default="", help="Suffix appended to base_save_dir (e.g. _seed43_t07)")
    p.add_argument("--llm_base_url", default=None, help="Override LLM_BASE_URL env for OpenAI-compatible endpoint")
    p.add_argument("--disable_evolution", action="store_true",
                   help="Disable A-mem's memory evolution step (skip LLM call in process_memory). "
                        "Halves ingestion LLM cost at the price of dropping A-mem's signature feature.")
    args = p.parse_args()

    global SEED, DEFAULT_TEMPERATURE
    SEED = args.seed
    DEFAULT_TEMPERATURE = args.temperature
    random.seed(SEED)
    if args.llm_base_url:
        os.environ["LLM_BASE_URL"] = args.llm_base_url

    llm_name_for_path = args.llm_model.lstrip("./").replace("/", "_")
    base_save_dir = os.path.join(args.save_dir, f"locomo-{llm_name_for_path}{args.save_dir_suffix}")
    os.makedirs(base_save_dir, exist_ok=True)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger("amem_runner")

    with open(args.dataset_path) as f:
        dialog_data = json.load(f)
    with open(args.locomo_path) as f:
        locomo = json.load(f)
    mm: List[Dict[str, Any]] = []
    if "composed" in args.styles and os.path.exists(args.multimem_path):
        with open(args.multimem_path) as f:
            mm = json.load(f)

    if args.samples == "all":
        sample_indices = list(range(len(dialog_data)))
    else:
        sample_indices = [int(x) for x in args.samples.split(",") if x.strip()]

    # Initialize memory system ONCE (we reset retriever per sample)
    variant = "no-evo" if args.disable_evolution else "with-evo"
    logger.info(f"Initializing AgenticMemorySystem ({variant})...")
    memsys = AgenticMemorySystem(
        model_name=args.embedding_model,
        llm_backend="openai",
        llm_model=args.llm_model,
        evo_threshold=args.evo_threshold,
        api_key="EMPTY",
    )
    if args.disable_evolution:
        _disable_evolution(memsys)
        logger.info("Evolution disabled (process_memory short-circuited).")

    styles = [s.strip() for s in args.styles.split(",") if s.strip()]

    for s_idx in sample_indices:
        sample_dir = os.path.join(base_save_dir, f"sample_{s_idx}")
        os.makedirs(sample_dir, exist_ok=True)
        # Skip if this sample's outputs already exist for every requested style.
        # Each style writes queries_solutions_<field>.json; composed writes composed_solutions.json.
        field_for_style = {"question": "question", "dialog": "dialog_query",
                           "implicit": "implicit_query", "counterfactual": "counterfactual_query"}
        expected = []
        for st in styles:
            if st == "composed":
                expected.append(os.path.join(sample_dir, "composed_solutions.json"))
            else:
                f = field_for_style.get(st)
                if f:
                    expected.append(os.path.join(sample_dir, f"queries_solutions_{f}.json"))
        if expected and all(os.path.exists(p) for p in expected):
            logger.info(f"[sample {s_idx}] all style outputs already present, skipping ingestion+retrieval")
            continue
        sample = dialog_data[s_idx]
        run_sample(memsys, sample, s_idx, styles, sample_dir, mm, top_k=args.top_k)

    logger.info("All samples done.")


if __name__ == "__main__":
    main()
