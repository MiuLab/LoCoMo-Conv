"""BM25 lightweight retrieval baseline.

Same chunking as AnchorMem / A-mem (3-turn windows, overlap=1, only referenced
sessions). Per-sample BM25 index over chunks. Top-K retrieval per query.

No LLM, no embedding. Pure sparse keyword match.

Output mirrors A-mem layout:
  outputs_bm25/locomo-bm25/sample_<i>/queries_solutions_<style>.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Any, Dict, List, Tuple

from rank_bm25 import BM25Okapi


def build_units(sample: Dict[str, Any], granularity: str) -> List[Tuple[str, List[str]]]:
    """Yield retrieval units as (text, [dia_id]).

    granularity = "turn": one unit per turn (locomo-native).
    granularity = "chunk": 3-turn window with overlap=1 (AnchorMem default).
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

        if granularity == "turn":
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
        else:  # chunk
            chunk_size, overlap = 3, 1
            step = max(1, chunk_size - overlap)
            for i in range(0, len(v), step):
                window = v[i:i + chunk_size]
                if not window:
                    continue
                lines = [f"DATE: {date}"]
                dia_ids: List[str] = []
                for turn in window:
                    spk = str(turn.get("speaker", "")).strip()
                    txt = str(turn.get("text", "")).strip()
                    line = f'{spk} said, "{txt}"'
                    if "blip_caption" in turn:
                        line += " and shared %s." % turn["blip_caption"]
                    lines.append(line)
                    if turn.get("dia_id"):
                        dia_ids.append(turn["dia_id"])
                out.append(("\n".join(lines), dia_ids))
    return out


def chunks_with_dia_ids(sample: Dict[str, Any]) -> List[Tuple[str, List[str]]]:
    """Back-compat alias defaulting to chunk granularity."""
    return build_units(sample, "chunk")


_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall((text or "").lower())


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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_path", default="data/locomo10_dialog.json")
    p.add_argument("--multimem_path", default="data/locomo10_multimem_full.json")
    p.add_argument("--save_dir", default="outputs_bm25/locomo-bm25")
    p.add_argument("--styles", default="question,dialog,implicit,counterfactual,composed")
    p.add_argument("--top_k", type=int, default=10)
    p.add_argument("--granularity", choices=["turn", "chunk"], default="turn",
                   help="Retrieval unit: 'turn' (locomo-native, one dia_id per unit) or 'chunk' (3-turn windows).")
    args = p.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    with open(args.dataset_path) as f:
        data = json.load(f)
    mm: List[Dict[str, Any]] = []
    if "composed" in args.styles and os.path.exists(args.multimem_path):
        with open(args.multimem_path) as f:
            mm = json.load(f)

    styles = [s.strip() for s in args.styles.split(",") if s.strip()]
    field_for_style = {
        "question": "question",
        "dialog": "dialog_query",
        "implicit": "implicit_query",
        "counterfactual": "counterfactual_query",
    }

    for s_idx, sample in enumerate(data):
        sample_dir = os.path.join(args.save_dir, f"sample_{s_idx}")
        os.makedirs(sample_dir, exist_ok=True)
        chunks = build_units(sample, args.granularity)
        if not chunks:
            print(f"[sample {s_idx}] no units", file=sys.stderr)
            continue
        corpus_tokens = [tokenize(c[0]) for c in chunks]
        bm25 = BM25Okapi(corpus_tokens)
        chunk_texts = [c[0] for c in chunks]
        chunk_dias = [c[1] for c in chunks]
        print(f"[sample {s_idx}] indexed {len(chunks)} {args.granularity}s", file=sys.stderr)

        for style in styles:
            if style == "composed":
                continue
            qfield = field_for_style.get(style)
            if qfield is None:
                continue
            ordered = select_qas_in_main_order(sample.get("qa", []), qfield)
            out_entries: List[Dict[str, Any]] = []
            for q_idx, qa in ordered:
                qstr = qa.get(qfield) or ""
                q_tokens = tokenize(qstr)
                scores = bm25.get_scores(q_tokens)
                top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[: args.top_k]
                out_entries.append({
                    "q_idx": q_idx,
                    "question": qa.get("question", ""),
                    "query": qstr,
                    "category": int(qa.get("category", 0)),
                    "evidence": qa.get("evidence", []),
                    "answer": qa.get("answer"),
                    "docs": [chunk_texts[i] for i in top_idx],
                    "retrieved_chunk_dia_ids": [chunk_dias[i] for i in top_idx],
                    "scores": [float(scores[i]) for i in top_idx],
                })
            out_path = os.path.join(sample_dir, f"queries_solutions_{qfield}.json")
            with open(out_path, "w") as f:
                json.dump(out_entries, f, ensure_ascii=False, indent=2)
            print(f"[sample {s_idx}] {style}: wrote {len(out_entries)} → {out_path}", file=sys.stderr)

        if "composed" in styles and mm:
            sample_clusters = [c for c in mm if c.get("sample_idx") == s_idx and c.get("composed_query")]
            if sample_clusters:
                out_entries = []
                for c in sample_clusters:
                    qstr = c["composed_query"]
                    q_tokens = tokenize(qstr)
                    scores = bm25.get_scores(q_tokens)
                    top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[: args.top_k]
                    out_entries.append({
                        "cluster_id": c["cluster_id"],
                        "sample_idx": c["sample_idx"],
                        "member_q_idxs": c["member_q_idxs"],
                        "gold_dia_ids": c["gold_dia_ids"],
                        "composed_query": qstr,
                        "docs": [chunk_texts[i] for i in top_idx],
                        "retrieved_chunk_dia_ids": [chunk_dias[i] for i in top_idx],
                        "scores": [float(scores[i]) for i in top_idx],
                    })
                out_path = os.path.join(sample_dir, "composed_solutions.json")
                with open(out_path, "w") as f:
                    json.dump(out_entries, f, ensure_ascii=False, indent=2)
                print(f"[sample {s_idx}] composed: wrote {len(out_entries)} → {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
