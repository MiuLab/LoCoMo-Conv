"""Build the supportive_memory annotation layer (Appendix: Supportive Memory Construction).

For each implicit QA, a dialogue turn t enters supportive_memory iff, for any of the four
raw-turn systems (BM25, Dense, A-mem, AnchorMem):
  1. t is cited by that system's +cot response (cited_ids -> top-10 doc -> dia_id via
     verbatim turn-text matching), and
  2. that system's +cot response did not lose outright to Oracle in the pairwise
     comparison (winner is 'cot' or 'tie'), and
  3. t is not already in the QA's gold `evidence` (supportive_memory is disjoint from gold).

Inputs (produced by the evaluation pipeline in this repo):
  --cache SYSTEM=DIR   per-system retrieval cache dir containing
                       sample_*/queries_solutions_implicit_query.json (top-10 docs)
  --e3 SYSTEM=FILE     per-system pairwise file with entries
                       {sample_idx, q_idx, winner, cited_ids}
                       (score_e3_cot_vs_oracle.py output)

Outputs: data/supportive_memory.json and, with --inject, the supportive_memory field
inside data/locomo10_dialog.json.
"""
import argparse, glob, json
from collections import defaultdict

def id_text(sample):
    m = {}
    for k, v in sample.get("conversation", {}).items():
        if k.startswith("session_") and isinstance(v, list):
            for t in v:
                if t.get("dia_id"):
                    m[str(t["dia_id"])] = (t.get("text") or "").lower()
    return m

def load_cache(d):
    out = {}
    for f in glob.glob(f"{d}/sample_*/queries_solutions_implicit_query.json"):
        s = int(f.split("sample_")[1].split("/")[0])
        for i, e in enumerate(json.load(open(f))):
            out[(s, e.get("q_idx", i))] = e.get("docs", [])[:10]
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dialog_path", default="data/locomo10_dialog.json")
    ap.add_argument("--cache", action="append", required=True, metavar="SYSTEM=DIR")
    ap.add_argument("--e3", action="append", required=True, metavar="SYSTEM=FILE")
    ap.add_argument("--out", default="data/supportive_memory.json")
    ap.add_argument("--inject", action="store_true",
                    help="also write the supportive_memory field back into --dialog_path")
    args = ap.parse_args()

    dd = json.load(open(args.dialog_path))
    maps = {i: id_text(s) for i, s in enumerate(dd)}
    caches = dict(p.split("=", 1) for p in args.cache)
    e3s = dict(p.split("=", 1) for p in args.e3)

    support = defaultdict(dict)
    for sys_, cdir in caches.items():
        cache = load_cache(cdir)
        for x in json.load(open(e3s[sys_])):
            if x.get("winner") not in ("cot", "tie"):
                continue
            key = (x["sample_idx"], x["q_idx"])
            docs = cache.get(key)
            if not docs:
                continue
            gold = set(map(str, dd[key[0]]["qa"][key[1]].get("evidence") or []))
            m = maps[key[0]]
            for cid in (x.get("cited_ids") or []):
                if not isinstance(cid, int) or cid < 1 or cid > len(docs):
                    continue
                doc = docs[cid - 1].lower()
                for did, txt in m.items():
                    if did not in gold and txt and txt in doc:
                        support[key].setdefault(did, set()).add(sys_)

    standalone = {}
    for (s, q), d in sorted(support.items()):
        entries = [{"dia_id": did, "n_systems": len(v), "source_systems": sorted(v)}
                   for did, v in sorted(d.items())]
        standalone[f"{s}:{q}"] = entries
        if args.inject:
            dd[s]["qa"][q]["supportive_memory"] = entries
    json.dump(standalone, open(args.out, "w"), ensure_ascii=False, indent=1)
    if args.inject:
        json.dump(dd, open(args.dialog_path, "w"), ensure_ascii=False, indent=1)
    n = sum(len(v) for v in standalone.values())
    print(f"QAs: {len(standalone)}  turns: {n}  mean/QA: {n/max(len(standalone),1):.2f}")

if __name__ == "__main__":
    main()
