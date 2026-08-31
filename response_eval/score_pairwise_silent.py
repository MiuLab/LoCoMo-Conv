"""Pairwise silent-grounding judge.

For each (sample, q_idx) where the WITH-MEMORY response had
partial-credit fact_used = 0.0 (response did not state the gold),
compare WITH-MEMORY response vs NO-MEMORY response. Judge sees the
gold so they can assess whether the with-memory response still
reflects the user's actual context (silent grounding).

A/B order is randomized per judge call to avoid position bias.

Output per record:
  { style, sample_idx, q_idx, with_label, winner_variant, raw_label, error? }
  winner_variant in {'with', 'no_mem', 'tie'}
"""
from __future__ import annotations
import argparse, json, os, random, sys, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List

import httpx
from openai import OpenAI
from tqdm import tqdm

try:
    from anthropic import Anthropic
except ImportError:
    Anthropic = None


PAIRWISE_PROMPT = """You are comparing two AI assistant responses to the same user message. The assistant is in a long-running conversation with this user.

User (named {user_name}) says:
{query}

Relevant excerpts from the user's actual past conversations (use these to understand the user's situation):
{evidence_block}

---
Response A:
{response_a}
---
Response B:
{response_b}
---

Which response gives a more appropriate reply given the user's actual situation? Judge how well each response fits the context — its framing, relevance, tone, and engagement with what the user is actually dealing with.

Consider:
- Does the response engage with the user's situation in a way that fits the context, or does it feel generic / off-topic / disconnected?
- Does it respond to what the user is actually asking or feeling?
- A response that confidently asserts a fact that contradicts the excerpts is worse than one that hedges honestly.

If both are equally appropriate (or equally inappropriate), output TIE.

Output format (exactly):
<reason>One or two sentences explaining your judgment.</reason>
<verdict>A | B | TIE</verdict>"""


def _openai_client() -> OpenAI:
    kp = os.path.expanduser("~/.openai-key")
    key = open(kp).readline().strip() if os.path.exists(kp) else os.environ.get("OPENAI_API_KEY", "")
    if not key:
        raise RuntimeError("OpenAI API key missing")
    return OpenAI(api_key=key, http_client=httpx.Client(timeout=180.0))


def _anthropic_client():
    if Anthropic is None:
        raise RuntimeError("anthropic SDK not installed")
    kp = os.path.expanduser("~/.anthropic-key")
    key = open(kp).readline().strip() if os.path.exists(kp) else os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        raise RuntimeError("Anthropic API key missing")
    return Anthropic(api_key=key)


def _parse_label(raw: str):
    """Parse verdict from output. Prefers <verdict>...</verdict> if present; falls back to first A/B/TIE in text."""
    import re
    if not raw:
        return None, None
    s = raw.strip()
    # Try structured verdict + reason
    rm = re.search(r"<reason>(.*?)</reason>", s, re.DOTALL | re.IGNORECASE)
    vm = re.search(r"<verdict>\s*(A|B|TIE)\s*</verdict>", s, re.IGNORECASE)
    reason = rm.group(1).strip() if rm else None
    if vm:
        return vm.group(1).upper(), reason
    # Fallback: first occurrence
    su = s.upper()
    if "TIE" in su:
        return "TIE", reason
    for ch in su:
        if ch in ('A', 'B'):
            return ch, reason
    return None, reason


def judge(client, model: str, query: str, evidence_block: str, user_name: str,
          response_a: str, response_b: str):
    prompt = PAIRWISE_PROMPT.format(
        user_name=user_name or "the user",
        query=query, evidence_block=evidence_block,
        response_a=response_a, response_b=response_b,
    )
    if model.startswith("claude"):
        kw = {"model": model, "max_tokens": 400,
              "messages": [{"role": "user", "content": prompt}]}
        # claude-opus-4-7 deprecates `temperature`; only set it for older models.
        if not model.startswith("claude-opus-4-7"):
            kw["temperature"] = 0.0
        resp = client.messages.create(**kw)
        text = "".join(b.text for b in resp.content if hasattr(b, "text"))
        return _parse_label(text)
    kwargs: Dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
    }
    if model.startswith("gpt-5") or model.startswith("gpt-o") or model.startswith("o3"):
        kwargs["max_completion_tokens"] = 2048
    else:
        kwargs["max_tokens"] = 400
        kwargs["temperature"] = 0.0
    resp = client.chat.completions.create(**kwargs)
    return _parse_label(resp.choices[0].message.content or "")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--with_responses_path", required=True,
                   help="responses.json containing the WITH-memory responses (e.g. oracle)")
    p.add_argument("--with_label", required=True,
                   help="label for the with-memory variant (e.g. 'oracle', 'AnchorMem-top_k')")
    p.add_argument("--with_variant", required=True,
                   help="variant field value to filter with-memory responses (e.g. 'oracle', 'top_k')")
    p.add_argument("--pool_variant", default="",
                   help="variant value in fact_used file used to define the pool (defaults to --with_variant)")
    p.add_argument("--with_system", default="",
                   help="system field value to filter (empty = match None/empty)")
    p.add_argument("--no_memory_responses_path",
                   default="outputs_response_eval/no_memory/responses.json")
    p.add_argument("--dialog_data_path", default="data/locomo10_dialog.json")
    p.add_argument("--fact_used_path", required=True,
                   help="partial-credit fact_used.json — used to find score=0.0 pool")
    p.add_argument("--styles", default="dialog,implicit")
    p.add_argument("--output", required=True)
    p.add_argument("--judge_model", default="gpt-5.4-mini")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--concurrency", type=int, default=10)
    p.add_argument("--checkpoint_every", type=int, default=200)
    args = p.parse_args()

    styles = set(args.styles.split(","))

    # 1. Load fact_used judgments → find pool of (style, sample, q) where with_variant score == 0.0
    fu = json.load(open(args.fact_used_path))
    with_sys = args.with_system or None
    pool_variant = args.pool_variant or args.with_variant
    pool = set()
    for r in fu:
        if r.get("fact_score") != 0.0: continue
        if r.get("variant") != pool_variant: continue
        if r.get("system") != with_sys: continue
        if r.get("style") not in styles: continue
        pool.add((r["style"], r["sample_idx"], r["q_idx"]))
    print(f"Pool size (score=0.0): {len(pool)}", file=sys.stderr)

    # 2. Load responses
    with_resp = json.load(open(args.with_responses_path))
    nomem_resp = json.load(open(args.no_memory_responses_path))

    with_map = {}
    for r in with_resp:
        if r.get("variant") != args.with_variant: continue
        if r.get("system") != with_sys: continue
        if r.get("style") not in styles: continue
        with_map[(r["style"], r["sample_idx"], r["q_idx"])] = r

    nomem_map = {}
    for r in nomem_resp:
        if r.get("variant") != "no_memory": continue
        if r.get("style") not in styles: continue
        nomem_map[(r["style"], r["sample_idx"], r["q_idx"])] = r

    # 3. Build job list (intersect pool with both response maps)
    # Load dialog data once and resolve evidence turns per job.
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from run_response_eval import all_speaker_turns, format_turns_block

    dialog_data = json.load(open(args.dialog_data_path))

    def evidence_block_for(sample_idx, evidence_ids):
        if not evidence_ids:
            return "(no excerpts available)"
        sample_obj = dialog_data[sample_idx]
        all_t = all_speaker_turns(sample_obj)
        keep = [t for t in all_t if t[0] in set(evidence_ids)]
        if not keep:
            return "(no excerpts found)"
        return format_turns_block(keep)

    jobs = []
    for k in pool:
        wr = with_map.get(k); nm = nomem_map.get(k)
        if not wr or not nm: continue
        if not wr.get("response") or not nm.get("response"): continue
        if wr.get("error") or nm.get("error"): continue
        evidence_ids = wr.get("gold_evidence") or []
        evidence_block = evidence_block_for(k[1], evidence_ids)
        jobs.append({
            "style": k[0], "sample_idx": k[1], "q_idx": k[2],
            "query": wr["query"],
            "evidence_block": evidence_block,
            "user_name": wr.get("subject_speaker_name") or "the user",
            "with_response": wr["response"],
            "nomem_response": nm["response"],
        })
    print(f"Jobs: {len(jobs)}", file=sys.stderr)

    # Resume
    out: List[Dict[str, Any]] = []
    done = set()
    if os.path.exists(args.output):
        out = json.load(open(args.output))
        for r in out:
            done.add((r["style"], r["sample_idx"], r["q_idx"]))
        print(f"Resuming, {len(done)} already done", file=sys.stderr)

    pending = [j for j in jobs if (j["style"], j["sample_idx"], j["q_idx"]) not in done]
    print(f"Pending: {len(pending)}", file=sys.stderr)

    rng = random.Random(args.seed)
    # Pre-decide A/B mapping per job (reproducible)
    for j in pending:
        j["_with_is_A"] = bool(rng.randint(0, 1))

    if args.judge_model.startswith("claude"):
        client = _anthropic_client()
    else:
        client = _openai_client()
    lock = threading.Lock()

    def worker(j):
        try:
            if j["_with_is_A"]:
                response_a, response_b = j["with_response"], j["nomem_response"]
            else:
                response_a, response_b = j["nomem_response"], j["with_response"]
            lab, reason = judge(client, args.judge_model, j["query"], j["evidence_block"], j["user_name"],
                        response_a, response_b)
            if lab is None: raise ValueError("label parse failed")
            if lab == "TIE":
                winner = "tie"
            elif lab == "A":
                winner = "with" if j["_with_is_A"] else "no_mem"
            elif lab == "B":
                winner = "no_mem" if j["_with_is_A"] else "with"
            else:
                raise ValueError(f"unknown label: {lab}")
            entry = {
                "with_label": args.with_label,
                "style": j["style"], "sample_idx": j["sample_idx"], "q_idx": j["q_idx"],
                "with_is_A": j["_with_is_A"],
                "raw_label": lab,
                "winner": winner,
                "reason": reason,
            }
        except Exception as ex:
            entry = {
                "with_label": args.with_label,
                "style": j["style"], "sample_idx": j["sample_idx"], "q_idx": j["q_idx"],
                "error": f"{type(ex).__name__}: {str(ex)[:150]}",
            }
        with lock:
            out.append(entry)
            if len(out) % args.checkpoint_every == 0:
                with open(args.output + ".tmp", "w") as f:
                    json.dump(out, f, ensure_ascii=False, indent=2)
                os.replace(args.output + ".tmp", args.output)

    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = [ex.submit(worker, j) for j in pending]
        for _ in tqdm(as_completed(futs), total=len(futs), desc="pairwise"):
            pass

    with open(args.output, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"Wrote {args.output}", file=sys.stderr)

    # Aggregate by style
    from collections import Counter
    by_style = {}
    for r in out:
        if "winner" not in r: continue
        by_style.setdefault(r["style"], Counter())[r["winner"]] += 1
    print(f"\n{'style':<14} {'n':<5} {'with-win':<10} {'no-mem-win':<12} {'tie':<5}")
    for st, c in by_style.items():
        n = c["with"] + c["no_mem"] + c["tie"]
        if n == 0: continue
        wr = c["with"] / n
        tr = c["tie"] / n
        # silent-grounding score: win-rate adjusted for ties (count tie as 0.5)
        sg = (c["with"] + 0.5 * c["tie"]) / n
        print(f"{st:<14} {n:<5} {c['with']} ({wr:.2%})  {c['no_mem']}  {c['tie']} ({tr:.2%})  SG-rate={sg:.3f}")


if __name__ == "__main__":
    main()
