"""Score response-eval responses with token-F1 and LLM-as-judge.

Input:  outputs_response_eval/<run>/responses.json
Output: outputs_response_eval/<run>/scored.json
        outputs_response_eval/<run>/aggregate.json

Metrics:
  - token_f1     : LoCoMo-style token-level F1 (lowercase, strip punct, split on ws)
  - llm_judge    : binary correct/incorrect by qwen (1 / 0)
  - cat5_halluc  : 1 if response mentions adversarial_answer (only for cat 5)
"""
from __future__ import annotations

import argparse, json, os, re, string, sys, threading
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

import httpx
from openai import OpenAI
from tqdm import tqdm


_SAFE_UA = {"User-Agent": "curl/8.0"}


def _vllm_client(base_url: str) -> OpenAI:
    return OpenAI(
        base_url=base_url,
        api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"),
        http_client=httpx.Client(headers=_SAFE_UA, timeout=180.0),
        default_headers=_SAFE_UA,
    )


def _openai_client() -> OpenAI:
    kp = os.path.expanduser("~/.openai-key")
    if os.path.exists(kp):
        api_key = open(kp).readline().strip()
    else:
        api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError("OpenAI API key missing (set OPENAI_API_KEY or ~/.openai-key)")
    return OpenAI(api_key=api_key, http_client=httpx.Client(timeout=180.0))


def _is_openai_model(model: str) -> bool:
    m = model.lower()
    return m.startswith("gpt-") or m.startswith("o1") or m.startswith("o3") or m.startswith("o4")


def _is_gpt5_style(model: str) -> bool:
    m = model.lower()
    return m.startswith("gpt-5") or m.startswith("o3") or m.startswith("o4") or m.startswith("gpt-o")


def _normalize(s: str) -> List[str]:
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(c if c not in string.punctuation else " " for c in s)
    return s.split()


def token_f1(pred: str, gold: str) -> float:
    """LoCoMo-style token F1."""
    if not gold or not pred:
        return 0.0
    p = _normalize(pred)
    g = _normalize(gold)
    if not p or not g:
        return 0.0
    common = Counter(p) & Counter(g)
    n_common = sum(common.values())
    if n_common == 0:
        return 0.0
    precision = n_common / len(p)
    recall = n_common / len(g)
    return 2 * precision * recall / (precision + recall)


JUDGE_PROMPT_DIALOG = """You are evaluating whether an AI assistant correctly answered a user's question.

The user (speaking in first person to the assistant) asked:
{query}

The factually correct answer (from the conversation history) is:
{gold}

Assistant's response:
{response}

Does the response correctly convey the gold fact? Be lenient about wording (paraphrasing fine) but strict about facts: the core fact must be present and correct. Extra unrelated detail is OK. Hedging like "I don't remember" when the gold fact IS available counts as NO.

Output ONLY one word: YES or NO."""


JUDGE_PROMPT_IMPLICIT = """You are evaluating an AI assistant's response in a conversational setting. The user did NOT directly ask a question; instead they shared a situation/concern where a SPECIFIC past fact from prior conversations would be highly relevant for the assistant to surface proactively.

User's message:
{query}

The relevant fact the assistant SHOULD bring up to be helpful:
{gold}

Assistant's response:
{response}

Did the assistant proactively use the relevant fact in a way that helps the user? Acceptable: the response surfaces the fact (paraphrased OK) and ties it to the user's situation. Not acceptable: the response is generic, ignores the fact, or hedges ("I don't have info on that") despite the fact being available. The response may include other helpful detail beyond the fact — that's fine.

Output ONLY one word: YES (proactively used the fact) or NO (did not)."""


JUDGE_PROMPT_COUNTERFACTUAL = """You are evaluating an AI assistant's response. The user has made a FALSE assertion about the past. The assistant should correct the user using the actual fact from prior conversation.

User's (incorrect) message:
{query}

The actual correct fact:
{gold}

Assistant's response:
{response}

Did the assistant CORRECT the user using the gold fact? Acceptable: the response identifies the user's error AND states the correct fact (paraphrasing OK). Not acceptable: the response agrees with the user's wrong claim, hedges without correcting, or restates a different fact. Polite tone is fine; the substance must be a correction grounded in the gold fact.

Output ONLY one word: YES (corrected with the gold fact) or NO (did not)."""


JUDGE_PROMPT_COMPOSED = """You are evaluating an AI assistant's response to a multi-part question. The assistant should draw on MULTIPLE facts from prior conversations.

User's message:
{query}

The set of relevant gold facts (the assistant should use all of them):
{gold}

Assistant's response:
{response}

Did the assistant correctly use ALL the gold facts? Acceptable: the response touches on each fact (paraphrasing fine). Not acceptable: missing any of the facts, or asserting incorrect alternatives.

Output ONLY one word: YES (all facts present and correct) or NO (any missing or wrong)."""


CAT5_PROMPT = """You are checking whether an AI assistant's response mentions a specific (false) claim. The user asked something that has NO answer in the conversation, and the response should not invent an answer.

User question:
{query}

A false/adversarial claim (this is what we DO NOT want to see in the response):
"{adversarial}"

Assistant's response:
{response}

Does the assistant's response assert or imply the false claim above (or anything semantically equivalent)? Output ONLY one word: YES (it hallucinates) or NO (it does not assert it)."""


JUDGE_PROMPTS = {
    "dialog": JUDGE_PROMPT_DIALOG,
    "implicit": JUDGE_PROMPT_IMPLICIT,
    "counterfactual": JUDGE_PROMPT_COUNTERFACTUAL,
    "composed": JUDGE_PROMPT_COMPOSED,
}


def _judge(client: OpenAI, model: str, prompt: str, max_tokens: int = 8) -> str:
    kwargs: Dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
    }
    if _is_gpt5_style(model):
        # gpt-5 / o-series reasoning models: temperature must default (=1), tokens param renamed,
        # and they emit reasoning tokens — need a big budget to leave room for the final word.
        kwargs["max_completion_tokens"] = 2048
    else:
        # Allow qwen to use its default thinking-on mode → leave kwargs alone.
        # max_tokens must be large enough to fit reasoning + final yes/no.
        if "qwen" in model.lower():
            kwargs["max_tokens"] = max(max_tokens, 2048)
            kwargs["temperature"] = 0.0
        else:
            kwargs["max_tokens"] = max_tokens
            kwargs["temperature"] = 0.0
    resp = client.chat.completions.create(**kwargs)
    raw = (resp.choices[0].message.content or "").strip().upper()
    # gpt-5 sometimes wraps answer in markdown / sentences; search for YES/NO inside.
    if "YES" in raw and "NO" not in raw.split("YES", 1)[0]:
        # Either starts with YES, or YES appears before any NO
        return "yes"
    if "NO" in raw and "YES" not in raw.split("NO", 1)[0]:
        return "no"
    if raw.startswith("YES"):
        return "yes"
    if raw.startswith("NO"):
        return "no"
    return ""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--judge_model", default=os.environ.get("JUDGE_MODEL", "gpt-5.4-mini"))
    p.add_argument("--judge_base_url", default=os.environ.get(
        "JUDGE_BASE_URL",
        "http://localhost:8000/v1",
    ), help="Only used when judge_model is a local vLLM model.")
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--checkpoint_every", type=int, default=100)
    p.add_argument("--skip_judge", action="store_true", help="Compute only token-F1")
    args = p.parse_args()

    out_dir = os.path.dirname(args.input)
    scored_path = os.path.join(out_dir, "scored.json")
    agg_path = os.path.join(out_dir, "aggregate.json")

    with open(args.input) as f:
        responses = json.load(f)
    print(f"Loaded {len(responses)} responses", file=sys.stderr)

    # Resume
    scored: List[Dict[str, Any]] = []
    done_keys = set()
    if os.path.exists(scored_path):
        scored = json.load(open(scored_path))
        for r in scored:
            done_keys.add(_key(r))
        print(f"Resuming: {len(done_keys)} already scored", file=sys.stderr)

    todo = [r for r in responses if _key(r) not in done_keys]
    print(f"Scoring {len(todo)} new items", file=sys.stderr)

    judge_client = None
    if not args.skip_judge:
        if _is_openai_model(args.judge_model):
            judge_client = _openai_client()
            print(f"Judge: OpenAI model `{args.judge_model}`", file=sys.stderr)
        else:
            judge_client = _vllm_client(args.judge_base_url)
            print(f"Judge: vLLM model `{args.judge_model}` @ {args.judge_base_url}", file=sys.stderr)

    lock = threading.Lock()

    def score_one(rec: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        resp = rec.get("response") or ""
        gold = rec.get("gold_answer") or ""
        cat = rec.get("category")
        is_cat5 = (cat == 5) or (cat is None and not gold and rec.get("adversarial_answer"))
        adv = rec.get("adversarial_answer") or ""

        f1 = token_f1(resp, gold) if gold else 0.0
        judge = ""
        halluc = -1  # not applicable

        if judge_client is not None:
            try:
                if is_cat5 and adv:
                    j = _judge(judge_client, args.judge_model,
                               CAT5_PROMPT.format(query=rec.get("query", ""),
                                                   adversarial=adv,
                                                   response=resp))
                    halluc = 1 if j == "yes" else (0 if j == "no" else -1)
                if gold and not is_cat5:
                    style = rec.get("style", "dialog")
                    prompt_tmpl = JUDGE_PROMPTS.get(style, JUDGE_PROMPT_DIALOG)
                    judge = _judge(judge_client, args.judge_model,
                                   prompt_tmpl.format(query=rec.get("query", ""),
                                                      gold=gold,
                                                      response=resp))
            except Exception:
                judge = ""
                halluc = -1

        out = {**rec, "token_f1": f1, "llm_judge": judge, "cat5_halluc": halluc}
        with lock:
            scored.append(out)
            if len(scored) % args.checkpoint_every == 0:
                _flush(scored_path, scored)
        return out

    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = [ex.submit(score_one, r) for r in todo]
        for _ in tqdm(as_completed(futs), total=len(futs), desc="scoring"):
            pass

    _flush(scored_path, scored)

    agg = aggregate(scored)
    with open(agg_path, "w") as f:
        json.dump(agg, f, ensure_ascii=False, indent=2)
    print(json.dumps(agg, indent=2))
    print(f"\nWrote {scored_path} and {agg_path}", file=sys.stderr)


def _key(r: Dict[str, Any]) -> str:
    return f"{r['variant']}|{r.get('system')}|{r.get('k')}|{r['style']}|{r.get('sample_idx')}|{r.get('q_idx')}|{r.get('cluster_id')}"


def _flush(path: str, data: List[Dict[str, Any]]) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def aggregate(scored: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Group by (variant, system, k, style) and compute mean F1 / judge-accuracy / cat5-halluc-rate."""
    groups: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)
    for r in scored:
        key = (r["variant"], r.get("system"), r.get("k"), r["style"])
        groups[key].append(r)

    out = []
    for key, items in sorted(groups.items(), key=lambda kv: (kv[0][0], str(kv[0][1]), str(kv[0][2]), kv[0][3])):
        variant, system, k, style = key
        # F1 over items with gold
        f1_items = [it["token_f1"] for it in items if it.get("gold_answer")]
        judge_items = [it["llm_judge"] for it in items if it.get("llm_judge") in ("yes", "no")]
        halluc_items = [it["cat5_halluc"] for it in items if it.get("cat5_halluc") in (0, 1)]
        row = {
            "variant": variant, "system": system, "k": k, "style": style,
            "n": len(items),
            "mean_f1": sum(f1_items) / len(f1_items) if f1_items else 0.0,
            "n_f1": len(f1_items),
            "judge_acc": sum(1 for j in judge_items if j == "yes") / len(judge_items) if judge_items else 0.0,
            "n_judge": len(judge_items),
            "halluc_rate": sum(halluc_items) / len(halluc_items) if halluc_items else 0.0,
            "n_halluc": len(halluc_items),
        }
        out.append(row)
    return {"rows": out, "n_total": len(scored)}


if __name__ == "__main__":
    main()
