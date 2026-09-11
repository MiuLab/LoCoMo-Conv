"""Nemori retrieval over LoCoMo-Conv queries, from the serialized stores snapshot.

Replicates nemori's vector search locally:
  - episodes embedded as f"{title} {content}"  (all-MiniLM-L6-v2)
  - semantic memories embedded as content
  - query embedded raw, cosine top-k per store

dia_id provenance:
  - episode  -> source_messages[*].metadata.dia_id
  - semantic -> source_episode_id -> episode -> dia_ids

Outputs per sample mirror other systems:
  outputs_nemori/locomo-gemma-4-31B-it/sample_{X}/queries_solutions_{field}.json
  (+ composed_solutions.json)

Two retrieval configs are recorded per query:
  - episodes top-10 only                      (turn-granular, comparable to other systems)
  - nemori default: episodes 10 + semantic 20 (its LoCoMo eval config)
"""
from __future__ import annotations
import argparse, json, os, time
import numpy as np


FIELD_FOR_STYLE = {
    "question": "question",
    "dialog": "dialog_query",
    "implicit": "implicit_query",
    "counterfactual": "counterfactual_query",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stores_path", default="baselines/AnchorMem/outputs_nemori_locomo10_v2_index_timing_stores.json")
    ap.add_argument("--dataset_path", default="baselines/AnchorMem/data/locomo10.json")
    ap.add_argument("--dialog_data_path", default="baselines/AnchorMem/data/locomo10_dialog.json")
    ap.add_argument("--multimem_path", default="data/locomo10_multimem_full.json")
    ap.add_argument("--output_dir", default="baselines/AnchorMem/outputs_nemori/locomo-gemma-4-31B-it")
    ap.add_argument("--embed_model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--top_k_episodes", type=int, default=10)
    ap.add_argument("--top_k_semantic", type=int, default=20)
    ap.add_argument("--styles", default="question,dialog,implicit,counterfactual,composed")
    args = ap.parse_args()

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(args.embed_model)

    stores = json.load(open(args.stores_path))
    episodes = stores["episodes"]
    semantics = stores["semantic_memories"]

    locomo = json.load(open(args.dataset_path))
    dialog_data = json.load(open(args.dialog_data_path))
    multimem = json.load(open(args.multimem_path)) if os.path.exists(args.multimem_path) else []

    # sample_idx <-> user_id
    idx_to_user = {}
    for i, sample in enumerate(locomo):
        sid = sample.get("sample_id") or f"sample_{i}"
        idx_to_user[i] = f"locomo10_{sid}"

    # episode -> dia_ids
    ep_dia = {}
    for ep in episodes:
        dias = []
        for m in ep.get("source_messages", []):
            did = (m.get("metadata") or {}).get("dia_id")
            if did:
                dias.append(str(did))
        ep_dia[ep["id"]] = dias

    # Pre-embed per user
    print("Embedding episodes and semantic memories...", flush=True)
    by_user_eps = {}
    by_user_sems = {}
    for uid in idx_to_user.values():
        u_eps = [ep for ep in episodes if ep["user_id"] == uid]
        u_sems = [s for s in semantics if s["user_id"] == uid]
        ep_texts = [f"{ep['title']} {ep['content']}" for ep in u_eps]
        sem_texts = [s["content"] for s in u_sems]
        ep_vecs = model.encode(ep_texts, normalize_embeddings=True) if ep_texts else np.zeros((0, 384))
        sem_vecs = model.encode(sem_texts, normalize_embeddings=True) if sem_texts else np.zeros((0, 384))
        by_user_eps[uid] = (u_eps, ep_vecs)
        by_user_sems[uid] = (u_sems, sem_vecs)
        print(f"  {uid}: {len(u_eps)} episodes, {len(u_sems)} semantic", flush=True)

    def search(uid, query):
        q = model.encode([query], normalize_embeddings=True)[0]
        u_eps, ep_vecs = by_user_eps[uid]
        u_sems, sem_vecs = by_user_sems[uid]
        ep_hits, sem_hits = [], []
        if len(u_eps):
            scores = ep_vecs @ q
            order = np.argsort(-scores)[: args.top_k_episodes]
            ep_hits = [(u_eps[i], float(scores[i])) for i in order]
        if len(u_sems):
            scores = sem_vecs @ q
            order = np.argsort(-scores)[: args.top_k_semantic]
            sem_hits = [(u_sems[i], float(scores[i])) for i in order]
        return ep_hits, sem_hits

    def dia_ids_for(ep_hits, sem_hits, include_semantic):
        out = set()
        for ep, _ in ep_hits:
            out.update(ep_dia.get(ep["id"], []))
        if include_semantic:
            for s, _ in sem_hits:
                src = s.get("source_episode_id")
                if src:
                    out.update(ep_dia.get(src, []))
        return sorted(out)

    styles = [s.strip() for s in args.styles.split(",") if s.strip()]

    for s_idx in range(len(locomo)):
        uid = idx_to_user[s_idx]
        sample_dir = os.path.join(args.output_dir, f"sample_{s_idx}")
        os.makedirs(sample_dir, exist_ok=True)
        qa_items = dialog_data[s_idx].get("qa", [])

        for style in styles:
            if style == "composed":
                out_path = os.path.join(sample_dir, "composed_solutions.json")
                if os.path.exists(out_path):
                    continue
                clusters = [c for c in multimem if c.get("sample_idx") == s_idx and c.get("composed_query")]
                entries = []
                t0 = time.time()
                for c in clusters:
                    ep_hits, sem_hits = search(uid, c["composed_query"])
                    entries.append({
                        "cluster_id": c["cluster_id"],
                        "sample_idx": s_idx,
                        "member_q_idxs": c.get("member_q_idxs", []),
                        "gold_dia_ids": c.get("gold_dia_ids", []),
                        "composed_query": c["composed_query"],
                        "docs": [f"{ep['title']}: {ep['content']}" for ep, _ in ep_hits],
                        "semantic_docs": [s["content"] for s, _ in sem_hits],
                        "retrieved_dia_ids_episodes": dia_ids_for(ep_hits, sem_hits, False),
                        "retrieved_dia_ids_full": dia_ids_for(ep_hits, sem_hits, True),
                    })
                json.dump(entries, open(out_path, "w"), ensure_ascii=False, indent=2)
                print(f"[sample {s_idx}] composed: wrote {len(entries)} in {time.time()-t0:.1f}s", flush=True)
                continue

            field = FIELD_FOR_STYLE[style]
            out_path = os.path.join(sample_dir, f"queries_solutions_{field}.json")
            if os.path.exists(out_path):
                continue
            entries = []
            t0 = time.time()
            for q_idx, qa in enumerate(qa_items):
                q_text = qa.get("question") if style == "question" else qa.get(field)
                if not q_text:
                    continue
                ep_hits, sem_hits = search(uid, q_text)
                entries.append({
                    "q_idx": q_idx,
                    "question": qa.get("question", ""),
                    "query": q_text,
                    "category": qa.get("category"),
                    "evidence": qa.get("evidence") or [],
                    "answer": qa.get("answer"),
                    "docs": [f"{ep['title']}: {ep['content']}" for ep, _ in ep_hits],
                    "semantic_docs": [s["content"] for s, _ in sem_hits],
                    "retrieved_dia_ids_episodes": dia_ids_for(ep_hits, sem_hits, False),
                    "retrieved_dia_ids_full": dia_ids_for(ep_hits, sem_hits, True),
                })
            json.dump(entries, open(out_path, "w"), ensure_ascii=False, indent=2)
            print(f"[sample {s_idx}] {style}: wrote {len(entries)} in {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
