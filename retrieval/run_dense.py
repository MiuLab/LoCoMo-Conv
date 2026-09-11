"""Pure dense retrieval baseline using sentence-transformers all-MiniLM-L6-v2.

Same ingestion granularity as BM25 (1 turn per document, only referenced
sessions), but uses dense embeddings + cosine similarity instead of sparse
lexical matching.

Output mirrors A-mem / BM25 layout:
  outputs_dense/locomo-dense/sample_<i>/queries_solutions_<style>.json
"""
from __future__ import annotations
import argparse, json, os, sys
from typing import Any, Dict, List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer


def build_units(sample: Dict[str, Any]) -> List[Tuple[str, List[str]]]:
    """1 turn per document. Only turns from sessions referenced by some QA evidence."""
    out: List[Tuple[str, List[str]]] = []
    referenced_sessions = set()
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
        for t in v:
            did = t.get("dia_id")
            spk = (t.get("speaker") or "").strip()
            txt = (t.get("text") or "").strip()
            if not (did and spk and txt):
                continue
            line = f'DATE: {date}\n{spk} said, "{txt}"'
            if "blip_caption" in t:
                line += f" and shared {t['blip_caption']}."
            out.append((line, [did]))
    return out


def select_qas_in_main_order(sample_qa: List[Dict[str, Any]], qfield: str):
    sel = []
    for q_idx, qa in enumerate(sample_qa):
        cat = int(qa.get("category", 0))
        if cat == 5 and qfield == "counterfactual_query":
            continue
        if cat == 5 and qfield == "question":
            continue
        qstr = qa.get(qfield) or ""
        if not qstr:
            continue
        sel.append((q_idx, qa))
    return sel


def cosine_topk(query_vec: np.ndarray, doc_vecs: np.ndarray, k: int) -> List[int]:
    # Vectors should be normalized; if not, do it
    sims = doc_vecs @ query_vec
    top_idx = np.argsort(-sims)[:k].tolist()
    return top_idx


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_path", default="data/locomo10_dialog.json")
    p.add_argument("--multimem_path", default="data/locomo10_multimem_full.json")
    p.add_argument("--save_dir", default="outputs_dense/locomo-dense")
    p.add_argument("--styles", default="dialog,implicit,counterfactual,composed")
    p.add_argument("--top_k", type=int, default=20)
    p.add_argument("--model_name", default="sentence-transformers/all-MiniLM-L6-v2")
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
        "dialog": "dialog_query",
        "implicit": "implicit_query",
        "counterfactual": "counterfactual_query",
    }

    print(f"Loading {args.model_name}...", file=sys.stderr)
    model = SentenceTransformer(args.model_name)

    for s_idx, sample in enumerate(data):
        sample_dir = os.path.join(args.save_dir, f"sample_{s_idx}")
        os.makedirs(sample_dir, exist_ok=True)
        chunks = build_units(sample)
        if not chunks:
            print(f"[sample {s_idx}] no units", file=sys.stderr)
            continue
        texts = [c[0] for c in chunks]
        dias = [c[1] for c in chunks]
        print(f"[sample {s_idx}] encoding {len(chunks)} turns", file=sys.stderr)
        doc_vecs = model.encode(texts, normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True)

        for style in styles:
            if style == "composed":
                continue
            qfield = field_for_style.get(style)
            if qfield is None:
                continue
            ordered = select_qas_in_main_order(sample.get("qa", []), qfield)
            if not ordered:
                continue
            qs = [qa.get(qfield) or "" for _, qa in ordered]
            q_vecs = model.encode(qs, normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True)
            out_entries = []
            for (q_idx, qa), q_vec in zip(ordered, q_vecs):
                top_idx = cosine_topk(q_vec, doc_vecs, args.top_k)
                out_entries.append({
                    "q_idx": q_idx,
                    "question": qa.get("question", ""),
                    "query": qa.get(qfield),
                    "category": int(qa.get("category", 0)),
                    "evidence": qa.get("evidence", []),
                    "answer": qa.get("answer"),
                    "docs": [texts[i] for i in top_idx],
                    "retrieved_chunk_dia_ids": [dias[i] for i in top_idx],
                    "scores": [float(doc_vecs[i] @ q_vec) for i in top_idx],
                })
            out_path = os.path.join(sample_dir, f"queries_solutions_{qfield}.json")
            with open(out_path, "w") as f:
                json.dump(out_entries, f, ensure_ascii=False, indent=2)
            print(f"[sample {s_idx}] {style}: wrote {len(out_entries)} → {out_path}", file=sys.stderr)

        if "composed" in styles and mm:
            sample_clusters = [c for c in mm if c.get("sample_idx") == s_idx and c.get("composed_query")]
            if sample_clusters:
                qs = [c["composed_query"] for c in sample_clusters]
                q_vecs = model.encode(qs, normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True)
                out_entries = []
                for c, q_vec in zip(sample_clusters, q_vecs):
                    top_idx = cosine_topk(q_vec, doc_vecs, args.top_k)
                    out_entries.append({
                        "cluster_id": c["cluster_id"],
                        "sample_idx": c["sample_idx"],
                        "member_q_idxs": c["member_q_idxs"],
                        "gold_dia_ids": c["gold_dia_ids"],
                        "composed_query": c["composed_query"],
                        "docs": [texts[i] for i in top_idx],
                        "retrieved_chunk_dia_ids": [dias[i] for i in top_idx],
                        "scores": [float(doc_vecs[i] @ q_vec) for i in top_idx],
                    })
                out_path = os.path.join(sample_dir, "composed_solutions.json")
                with open(out_path, "w") as f:
                    json.dump(out_entries, f, ensure_ascii=False, indent=2)
                print(f"[sample {s_idx}] composed: wrote {len(out_entries)} → {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
