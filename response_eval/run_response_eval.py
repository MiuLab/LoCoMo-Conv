"""Unified response-quality evaluation driver.

Supports 5 prompt-construction variants:
  1. all_memory          : every turn of the conversation in the prompt
  2. top_k               : top-K from a memory system (K configurable)
  3. compressed          : top-K compressed by an LLM conditioned on the query
  4. oracle              : only the gold-evidence turns
  5. reasoning_rewrite   : reasoning model rewrites query → re-retrieve → top-K

For each (variant, system?, K?, query), produces an answer string using the
configured answer LLM (default: gemma-4-31B-it).

Output: outputs_response_eval/<run_name>/responses.json
        [{variant, system, k, sample_idx, q_idx, style, query, gold_answer,
          response, prompt_len}]
"""
from __future__ import annotations

import argparse, json, os, sys, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Tuple

import httpx
from openai import OpenAI
from tqdm import tqdm


# =================== LLM clients ===================

_SAFE_UA = {"User-Agent": "curl/8.0"}


def _vllm_client(base_url: str) -> OpenAI:
    http = httpx.Client(headers=_SAFE_UA, timeout=180.0)
    return OpenAI(
        base_url=base_url,
        api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"),
        http_client=http,
        default_headers=_SAFE_UA,
    )


def _openai_client() -> OpenAI:
    kp = os.path.expanduser("~/.openai-key")
    if os.path.exists(kp):
        api_key = open(kp).readline().strip()
    else:
        api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError("OpenAI API key missing")
    return OpenAI(api_key=api_key, http_client=httpx.Client(timeout=180.0))


def _chat(client: OpenAI, model: str, prompt: str, max_tokens: int = 300,
          temperature: float = 0.0, stream: bool = True,
          enable_thinking: str = "default", seed: int = None) -> str:
    """Stream by default so each chunk resets Cloudflare's 120s origin-response timer.
    Large all_memory prompts otherwise hit Cloudflare 524 even when gemma is healthy.

    enable_thinking: 'default' (don't pass), 'on' / 'off' (pass via chat_template_kwargs).
    """
    is_gpt5 = model.startswith("gpt-5") or model.startswith("gpt-o")
    kwargs: Dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
    }
    if is_gpt5:
        kwargs["max_completion_tokens"] = max_tokens + 200  # buffer for reasoning
        stream = False  # OpenAI reasoning models don't stream usefully here
    else:
        kwargs["temperature"] = temperature
        if seed is not None:
            kwargs["seed"] = seed
        wants_thinking = enable_thinking == "on" or (
            enable_thinking == "default" and "qwen" in model.lower()
        )
        # need extra budget when thinking blocks are emitted
        kwargs["max_tokens"] = max(max_tokens, 3000) if wants_thinking else max_tokens
        if enable_thinking in ("on", "off"):
            kwargs["extra_body"] = {
                "chat_template_kwargs": {"enable_thinking": enable_thinking == "on"}
            }

    if not stream:
        resp = client.chat.completions.create(**kwargs)
        return (resp.choices[0].message.content or "").strip()

    # Streaming: accumulate deltas
    kwargs["stream"] = True
    parts: List[str] = []
    for chunk in client.chat.completions.create(**kwargs):
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta.content
        if delta:
            parts.append(delta)
    return "".join(parts).strip()


# =================== Memory loaders ===================

def all_speaker_turns(sample: Dict[str, Any]) -> List[Tuple[str, str, str, str]]:
    """Yield (dia_id, date, speaker, text) for every turn in conversation order."""
    out = []
    for k, v in sample.get("conversation", {}).items():
        if not (isinstance(k, str) and k.startswith("session_") and "date" not in k):
            continue
        if not isinstance(v, list):
            continue
        date = sample["conversation"].get(k + "_date_time", "")
        for t in v:
            did = t.get("dia_id")
            spk = (t.get("speaker") or "").strip()
            txt = (t.get("text") or "").strip()
            if did and txt:
                out.append((did, date, spk, txt))
    return out


def format_turns_block(turns: List[Tuple[str, str, str, str]]) -> str:
    lines = []
    for did, date, spk, txt in turns:
        lines.append(f"[{did} | {date}] {spk}: {txt}")
    return "\n".join(lines)


def gold_evidence_turns(sample: Dict[str, Any], dia_ids: List[str]) -> List[Tuple[str, str, str, str]]:
    """Get verbatim evidence turns for an item."""
    all_t = all_speaker_turns(sample)
    target = set(dia_ids)
    return [t for t in all_t if t[0] in target]


# Cache file naming per system
SYSTEM_CACHE_DIR = {
    "AnchorMem": "outputs/locomo-gemma-4-31B-it",
    "A-mem":     "outputs_amem/no_evo/locomo-gemma-4-31B-it",
    "A-mem-evo": "outputs_amem/with_evo/locomo-gemma-4-31B-it",
    "mem0":      "outputs_mem0/locomo-gemma-4-31B-it",
    "BM25":      "outputs_bm25/locomo-bm25",
    "Dense":     "outputs_dense/locomo-dense",
    "memora":    "outputs_memora_v3_nothresh/locomo-gemma-4-31B-it",
}


def load_system_topk(system: str, sample_idx: int, style: str,
                     q_idx: int, k: int) -> List[str]:
    """Return top-K retrieved docs (text) for (system, sample, query, style)."""
    style_to_field = {
        "dialog": "dialog_query",
        "implicit": "implicit_query",
        "counterfactual": "counterfactual_query",
    }
    field = style_to_field.get(style)
    if not field:
        return []
    base = SYSTEM_CACHE_DIR[system]
    cache = os.path.join(base, f"sample_{sample_idx}", f"queries_solutions_{field}.json")
    if not os.path.exists(cache):
        return []
    with open(cache) as f:
        sols = json.load(f)
    # Need to find the entry corresponding to this q_idx via select order
    # For simplicity, we match by stored q_idx field (A-mem/mem0/BM25 saved it explicitly).
    # For AnchorMem (legacy QuerySolution), match by question string.
    for entry in sols:
        if entry.get("q_idx") == q_idx:
            return (entry.get("docs") or [])[:k]
    # AnchorMem path: store doesn't have q_idx; build map by question text
    # (Original_q must match.)
    return []  # caller handles


def load_anchormem_topk_by_question(sample_idx: int, style: str, original_q: str, k: int) -> List[str]:
    field = {"dialog": "dialog_query", "implicit": "implicit_query",
             "counterfactual": "counterfactual_query"}.get(style)
    if not field:
        return []
    base = SYSTEM_CACHE_DIR["AnchorMem"]
    cache = os.path.join(base, f"sample_{sample_idx}", f"queries_solutions_{field}.json")
    if not os.path.exists(cache):
        return []
    with open(cache) as f:
        sols = json.load(f)
    # AnchorMem's QuerySolution has .question = the rewritten query, NOT original.
    # Need to map: locate the entry with the same query as a rewriting.
    # We approximate by question-text alignment from data file.
    # Caller must pass the actual query text (not original Q) — pass implicit_query etc.
    # Here we just look up by exact query string.
    for entry in sols:
        if entry.get("question") == original_q or entry.get("query") == original_q:
            return (entry.get("docs") or [])[:k]
    return []


# =================== Prompt templates ===================

ANSWER_PROMPT = """You are an AI assistant with long-term memory of past conversations with the user.

Below are relevant memory items the assistant has access to:
<memory>
{memory_block}
</memory>

The user (speaker: {speaker}) now says:
"{query}"

Provide a concise answer that directly addresses the user. If the memory clearly contains the relevant information, use it. If the memory does not contain the needed information, say so plainly.

Answer:"""


NO_MEMORY_PROMPT = """You are an AI assistant. You do not have access to any prior conversation history with this user.

The user (speaker: {speaker}) says:
"{query}"

Provide a concise answer that directly addresses the user. If you do not have the information needed to answer specifically, say so plainly.

Answer:"""


COMPRESS_PROMPT = """You are summarizing relevant past memories for an AI assistant. The assistant is in a long-running conversation with {speaker}, and {speaker} has just sent a new message. You will see retrieved memory items from {speaker}'s prior conversations with the assistant. Your job is to distill the parts of those memories that will help the assistant respond well.

{speaker}'s current message:
"{query}"

Retrieved memory items (from prior conversations with {speaker}):
{memory_block}

Write a coherent summary of the relevant facts from these memories. Keep specific dates, names, numbers, places, and list items VERBATIM. If multiple memories relate to the message, include all of them. Do not invent facts not present in the memories. Do not include disclaimers or meta-comments — just the relevant content. Length: as long as needed to preserve all relevant details."""


COT_SELECT_PROMPT = """You are an AI assistant with long-term memory of past conversations with {speaker}.

Below are numbered memory items the assistant has retrieved. Some may be relevant to the user's message, others may not.

{memory_block}

The user (speaker: {speaker}) now says:
"{query}"

Work through this in three steps:
1. Identify what the user is conveying (a question, a situation, a plan, an emotion). Write this as a single sentence.
2. List which numbered memories actually carry information the assistant can draw on when responding — these may be facts that answer a question, or context that grounds an empathetic / contextual reply. For EACH memory you select, output one bullet on its own line in the exact format `- [N]: <one-line note on what it adds>`. Do NOT group multiple memories into one bullet. Do NOT add a bullet for memories you are not selecting. If none are useful, write a single line `- none`.
3. Based ONLY on the memories you cited in step 2, write a concise, natural response to the user. If the cited memories do not provide enough relevant content to respond meaningfully, say so plainly.

Respond using this EXACT format (do not add any other sections, headings, or markdown):

<thinking>
Step 1: <one sentence>
Step 2:
- [N]: <one-line note>
- [M]: <one-line note>
</thinking>
<cited>1, 5, 7</cited>
<response>
... your final response to the user ...
</response>"""


COT_NOSELECT_PROMPT = """You are an AI assistant with long-term memory of past conversations with {speaker}.

Below are numbered memory items the assistant has retrieved. Some may be relevant to the user's message, others may not.

{memory_block}

The user (speaker: {speaker}) now says:
"{query}"

Work through this in two steps:
1. Identify what the user is conveying (a question, a situation, a plan, an emotion). Write this as a single sentence.
2. Considering all the memories above, write a concise, natural response to the user. If the memories do not provide enough relevant content to respond meaningfully, say so plainly.

Respond using this EXACT format (do not add any other sections, headings, or markdown):

<thinking>
Step 1: <one sentence>
</thinking>
<response>
... your final response to the user ...
</response>"""


REWRITE_PROMPT = """A user has said the following in conversation with an AI assistant. The AI needs to retrieve relevant memories from past conversations to respond well. Rewrite the user's message into a CONCISE search query that captures the topic, entity, and intent in a form a retrieval system can match (use third-person if helpful).

User message:
"{query}"

Output ONLY the search query string, nothing else. No quotes, no preamble."""


# =================== Prompt construction per variant ===================

# Model context limit: 32768 tokens. Observed ~3.10 chars/token for gemma on LoCoMo dialogue.
# Budget: 32768 - 300 (answer) - 200 (safety) = 32268 input tokens → 32268 × 3.10 ≈ 100k chars TOTAL.
# Minus ~800 chars scaffolding → 99k chars memory budget. Use 95k to leave slack.
MAX_MEMORY_CHARS = 95_000


def _clip_memory(mem: str) -> str:
    if len(mem) <= MAX_MEMORY_CHARS:
        return mem
    # Keep the most recent turns: cut from the front (older), keep tail.
    cut = mem[-MAX_MEMORY_CHARS:]
    nl = cut.find("\n")
    if nl > 0:
        cut = cut[nl + 1:]
    return "[... earlier conversation truncated to fit context window ...]\n" + cut


def build_prompt_all(sample: Dict[str, Any], query: str, subject_name: str) -> str:
    turns = all_speaker_turns(sample)
    mem = _clip_memory(format_turns_block(turns))
    return ANSWER_PROMPT.format(memory_block=mem, query=query, speaker=subject_name)


def build_prompt_oracle(sample: Dict[str, Any], evidence: List[str],
                        query: str, subject_name: str) -> str:
    turns = gold_evidence_turns(sample, evidence)
    mem = format_turns_block(turns) if turns else "(no evidence available)"
    return ANSWER_PROMPT.format(memory_block=mem, query=query, speaker=subject_name)


def build_prompt_topk(docs: List[str], query: str, subject_name: str) -> str:
    if not docs:
        mem = "(no memories retrieved)"
    else:
        mem = "\n\n".join(f"- {d}" for d in docs)
    return ANSWER_PROMPT.format(memory_block=mem, query=query, speaker=subject_name)


def build_prompt_no_memory(query: str, subject_name: str) -> str:
    return NO_MEMORY_PROMPT.format(query=query, speaker=subject_name)


def build_prompt_cot_select(docs: List[str], query: str, subject_name: str) -> str:
    if not docs:
        mem = "(no memories retrieved)"
    else:
        mem = "\n\n".join(f"[{i+1}] {d}" for i, d in enumerate(docs))
    return COT_SELECT_PROMPT.format(memory_block=mem, query=query, speaker=subject_name)


def build_prompt_cot_noselect(docs: List[str], query: str, subject_name: str) -> str:
    if not docs:
        mem = "(no memories retrieved)"
    else:
        mem = "\n\n".join(f"[{i+1}] {d}" for i, d in enumerate(docs))
    return COT_NOSELECT_PROMPT.format(memory_block=mem, query=query, speaker=subject_name)


def parse_cot_response(raw: str) -> Tuple[List[int], str, str]:
    """Parse <thinking>...</thinking><cited>...</cited><response>...</response> output.

    Returns (cited_ids, response, full_text). Falls back gracefully if tags missing.
    """
    import re
    cited_ids: List[int] = []
    response = raw.strip()
    m_cited = re.search(r"<cited>(.*?)</cited>", raw, re.DOTALL)
    if m_cited:
        for tok in re.split(r"[,\s]+", m_cited.group(1)):
            tok = tok.strip()
            if tok.isdigit():
                cited_ids.append(int(tok))
    m_resp = re.search(r"<response>(.*?)</response>", raw, re.DOTALL)
    if m_resp:
        response = m_resp.group(1).strip()
    return cited_ids, response, raw


# =================== Main driver ===================

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sample_path", default="data/response_eval_sample.json")
    p.add_argument("--dataset_path", default="data/locomo10_dialog.json")
    p.add_argument("--multimem_path", default="data/locomo10_multimem.json")
    p.add_argument("--variants", default="all_memory,top_k,compressed,oracle,reasoning_rewrite",
                   help="comma-separated subset of variants to run")
    p.add_argument("--styles", default="dialog,implicit,counterfactual,composed")
    p.add_argument("--ks", default="5,10,20", help="K values for top_k variant")
    p.add_argument("--systems", default="AnchorMem,A-mem,mem0,BM25",
                   help="systems for top_k / compressed / rewrite variants")
    p.add_argument("--rewrite_systems", default="AnchorMem,A-mem,BM25",
                   help="systems for reasoning_rewrite variant (subset of --systems)")

    p.add_argument("--answer_model", default="./gemma-4-31B-it")
    p.add_argument("--answer_base_url", default=os.environ.get(
        "ANSWER_BASE_URL",
        "http://localhost:8000/v1",
    ))
    p.add_argument("--rewrite_model", default="gpt-5.4-mini")
    p.add_argument("--compress_model", default="./gemma-4-31B-it")

    p.add_argument("--cache_override", default="",
                   help="Redirect a system's top-k cache dir: 'memora=outputs_memora_v3_rewrite/locomo-gemma-4-31B-it'")
    p.add_argument("--output_dir", default="outputs_response_eval/run1")
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--max_tokens", type=int, default=300)
    p.add_argument("--checkpoint_every", type=int, default=100)
    p.add_argument("--enable_thinking", choices=["default", "on", "off"], default="default",
                   help="Pass chat_template_kwargs.enable_thinking to vLLM-hosted model (qwen/gemma).")
    p.add_argument("--seed", type=int, default=None, help="Seed for answer-model sampling (for variance runs)")
    p.add_argument("--temperature", type=float, default=0.0, help="Temperature for answer-model sampling")
    args = p.parse_args()

    # Redirect a system's top-k retrieval cache, e.g. to evaluate a query-rewriting
    # variant that lives in a separate retrieval dir. Format: "system=dir[,system=dir]".
    if getattr(args, "cache_override", ""):
        for pair in args.cache_override.split(","):
            if "=" in pair:
                syskey, d = pair.split("=", 1)
                SYSTEM_CACHE_DIR[syskey.strip()] = d.strip()
                print(f"[cache_override] {syskey.strip()} -> {d.strip()}", file=sys.stderr)

    os.makedirs(args.output_dir, exist_ok=True)

    # Load resources
    with open(args.sample_path) as f:
        sampled = json.load(f)
    with open(args.dataset_path) as f:
        dialog_data = json.load(f)

    # Build clients
    answer_client = _vllm_client(args.answer_base_url)
    compress_client = answer_client  # same gemma endpoint
    rewrite_client = None
    if "reasoning_rewrite" in args.variants:
        rewrite_client = _openai_client()

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    styles = [s.strip() for s in args.styles.split(",") if s.strip()]
    ks = [int(k) for k in args.ks.split(",") if k.strip()]
    systems = [s.strip() for s in args.systems.split(",") if s.strip()]
    rewrite_systems = [s.strip() for s in args.rewrite_systems.split(",") if s.strip()]

    # Build job list. Each job = one prompt to send to gemma.
    jobs: List[Dict[str, Any]] = []

    def speaker_name(sample, subj_key):
        conv = sample.get("conversation", {})
        return conv.get("speaker_a") if subj_key == "speaker_a" else conv.get("speaker_b")

    for style in styles:
        items = sampled["samples"].get(style, [])
        for item in items:
            if style == "composed":
                # Composed clusters: query = composed_query, gold = gold_dia_ids, no subject
                sample_idx = item["sample_idx"]
                cluster_id = item["cluster_id"]
                gold = item["gold_dia_ids"]
                query = item["composed_query"]
                subj_key = item.get("subject_speaker", "speaker_a")
                subj_name = speaker_name(dialog_data[sample_idx], subj_key)
                gold_answers_text = "; ".join(
                    (", ".join(str(a) for a in mf["answer"])
                     if isinstance(mf.get("answer"), list)
                     else str(mf.get("answer") or ""))
                    for mf in item.get("member_facts", [])
                )
                key_base = {
                    "style": style, "sample_idx": sample_idx, "q_idx": None,
                    "cluster_id": cluster_id, "query": query,
                    "gold_answer": gold_answers_text,
                    "gold_evidence": gold,
                    "subject_speaker": subj_key,
                    "subject_speaker_name": subj_name,
                }
            else:
                sample_idx = item["sample_idx"]
                q_idx = item["q_idx"]
                query = item["query"]
                if not query:
                    continue
                gold = item.get("evidence") or []
                gold_ans = item.get("answer")
                if isinstance(gold_ans, list):
                    gold_ans_str = ", ".join(str(a) for a in gold_ans)
                else:
                    gold_ans_str = str(gold_ans or "")
                # cat 5 has no gold answer; track adversarial for hallucination eval
                if not gold and item.get("adversarial_answer"):
                    pass
                subj_name = speaker_name(dialog_data[sample_idx], item.get("subject_speaker", "speaker_a"))
                key_base = {
                    "style": style, "sample_idx": sample_idx, "q_idx": q_idx,
                    "query": query,
                    "gold_answer": gold_ans_str,
                    "gold_evidence": gold,
                    "adversarial_answer": item.get("adversarial_answer"),
                    "subject_speaker": item.get("subject_speaker", "speaker_a"),
                    "subject_speaker_name": subj_name,
                    "category": item.get("category"),
                }

            # Now enumerate variant-specific jobs
            if "all_memory" in variants:
                jobs.append({**key_base, "variant": "all_memory", "system": None, "k": None})
            if "oracle" in variants and key_base.get("gold_evidence"):
                jobs.append({**key_base, "variant": "oracle", "system": None, "k": None})
            if "top_k" in variants:
                for system in systems:
                    for k in ks:
                        jobs.append({**key_base, "variant": "top_k", "system": system, "k": k})
            if "compressed" in variants:
                for system in systems:
                    jobs.append({**key_base, "variant": "compressed", "system": system, "k": 10})
            if "reasoning_rewrite" in variants:
                for system in rewrite_systems:
                    jobs.append({**key_base, "variant": "reasoning_rewrite", "system": system, "k": 10})
            if "cot_select" in variants:
                for system in systems:
                    jobs.append({**key_base, "variant": "cot_select", "system": system, "k": 10})
            if "cot_noselect" in variants:
                for system in systems:
                    jobs.append({**key_base, "variant": "cot_noselect", "system": system, "k": 10})
            if "no_memory" in variants:
                jobs.append({**key_base, "variant": "no_memory", "system": None, "k": None})

    print(f"Total jobs queued: {len(jobs)}", file=sys.stderr)

    # Output file (resume-friendly)
    out_path = os.path.join(args.output_dir, "responses.json")
    done_keys = set()
    results: List[Dict[str, Any]] = []
    if os.path.exists(out_path):
        with open(out_path) as f:
            results = json.load(f)
        for r in results:
            done_keys.add(_job_key(r))
        print(f"Resuming: {len(done_keys)} responses already saved", file=sys.stderr)

    # Pre-compute rewritten queries once per (sample,q_idx,query) — they don't depend on system
    rewrite_cache: Dict[str, str] = {}
    if "reasoning_rewrite" in variants:
        unique_queries = set()
        for j in jobs:
            if j["variant"] == "reasoning_rewrite":
                unique_queries.add(j["query"])
        print(f"Pre-computing {len(unique_queries)} query rewrites via {args.rewrite_model}...", file=sys.stderr)
        rcache_path = os.path.join(args.output_dir, "rewrites.json")
        if os.path.exists(rcache_path):
            rewrite_cache = json.load(open(rcache_path))
            print(f"  loaded {len(rewrite_cache)} cached rewrites", file=sys.stderr)
        todo = [q for q in unique_queries if q not in rewrite_cache]
        assert rewrite_client is not None
        for q in tqdm(todo, desc="rewrites"):
            try:
                rw = _chat(rewrite_client, args.rewrite_model,
                           REWRITE_PROMPT.format(query=q), max_tokens=120)
                rewrite_cache[q] = rw.strip().strip('"')
            except Exception:
                rewrite_cache[q] = q  # fallback: use original
        with open(rcache_path, "w") as f:
            json.dump(rewrite_cache, f, ensure_ascii=False, indent=2)

    # Optional cache: rewrite_retrievals[<system>|<sample>|<q_idx|cluster_id>] = [doc, ...]
    rewrite_retrievals: Dict[str, List[str]] = {}
    rr_path = os.path.join(args.output_dir, "rewrite_retrievals.json")
    if os.path.exists(rr_path):
        with open(rr_path) as f:
            rewrite_retrievals = json.load(f)
        print(f"Loaded {len(rewrite_retrievals)} pre-computed rewrite retrievals", file=sys.stderr)

    # ---- Build prompt + run answer per job ----
    lock = threading.Lock()

    def get_topk_for_job(j) -> List[str]:
        """Resolve top-K docs for variant top_k/compressed (uses cached retrievals)."""
        system = j["system"]
        sample_idx = j["sample_idx"]
        k = j["k"]
        style = j["style"]
        if style == "composed":
            # Composed clusters use their own cache file (composed_solutions.json)
            base = SYSTEM_CACHE_DIR[system]
            cache = os.path.join(base, f"sample_{sample_idx}", "composed_solutions.json")
            if not os.path.exists(cache):
                return []
            with open(cache) as f:
                sols = json.load(f)
            cid = j["cluster_id"]
            for entry in sols:
                if entry.get("cluster_id") == cid:
                    return (entry.get("docs") or [])[:k]
            return []
        else:
            field = {"dialog": "dialog_query", "implicit": "implicit_query",
                     "counterfactual": "counterfactual_query"}.get(style)
            if not field:
                return []
            base = SYSTEM_CACHE_DIR[system]
            cache = os.path.join(base, f"sample_{sample_idx}", f"queries_solutions_{field}.json")
            if not os.path.exists(cache):
                return []
            with open(cache) as f:
                sols = json.load(f)
            q_idx = j["q_idx"]
            # A-mem/mem0/BM25 have q_idx field
            for entry in sols:
                if entry.get("q_idx") == q_idx:
                    return (entry.get("docs") or [])[:k]
            # AnchorMem (no q_idx) — match by query text
            qstr = j["query"]
            for entry in sols:
                if entry.get("question") == qstr or entry.get("query") == qstr:
                    return (entry.get("docs") or [])[:k]
            return []

    def worker(job_idx: int):
        j = jobs[job_idx]
        key = _job_key(j)
        if key in done_keys:
            return None
        try:
            sample_obj = dialog_data[j["sample_idx"]]
            subj_name = j["subject_speaker_name"]
            v = j["variant"]

            if v == "all_memory":
                prompt = build_prompt_all(sample_obj, j["query"], subj_name)
            elif v == "oracle":
                prompt = build_prompt_oracle(sample_obj, j.get("gold_evidence", []), j["query"], subj_name)
            elif v == "top_k":
                docs = get_topk_for_job(j)
                prompt = build_prompt_topk(docs, j["query"], subj_name)
            elif v == "compressed":
                docs = get_topk_for_job(j)
                if docs:
                    compress_input = "\n".join(f"- {d}" for d in docs)
                    summary = _chat(compress_client, args.compress_model,
                                    COMPRESS_PROMPT.format(query=j["query"], memory_block=compress_input, speaker=subj_name),
                                    max_tokens=200)
                else:
                    summary = "No relevant memory."
                prompt = build_prompt_topk([summary], j["query"], subj_name)
            elif v == "reasoning_rewrite":
                rewritten = rewrite_cache.get(j["query"], j["query"])
                # Prefer pre-computed re-retrieval cache (filled by rerun_retrieval_with_rewrites.py).
                # Falls back to original top-K + rewrite hint if cache miss.
                docs = rewrite_retrievals.get(_rrkey(j), None)
                if docs is None:
                    docs = get_topk_for_job(j)
                    prompt = (build_prompt_topk(docs, j["query"], subj_name)
                              + f"\n\n[Hint: this query relates to: {rewritten}]")
                else:
                    prompt = build_prompt_topk(docs[: j["k"]], j["query"], subj_name)
            elif v == "cot_select":
                docs = get_topk_for_job(j)
                prompt = build_prompt_cot_select(docs, j["query"], subj_name)
            elif v == "cot_noselect":
                docs = get_topk_for_job(j)
                prompt = build_prompt_cot_noselect(docs, j["query"], subj_name)
            elif v == "no_memory":
                prompt = build_prompt_no_memory(j["query"], subj_name)
            else:
                return None

            prompt_len = len(prompt)
            ans = _chat(answer_client, args.answer_model, prompt, max_tokens=args.max_tokens,
                        temperature=args.temperature, seed=args.seed,
                        enable_thinking=args.enable_thinking)
        except Exception as e:
            ans = ""
            prompt_len = -1
            err = f"{type(e).__name__}: {e}"
            with lock:
                results.append({**j, "response": "", "prompt_len": prompt_len, "error": err})
            return None

        entry = {**j, "response": ans, "prompt_len": prompt_len, "error": None}
        if v == "cot_select":
            cited, response_only, _ = parse_cot_response(ans)
            entry["cited_ids"] = cited
            entry["response_only"] = response_only
            entry["cot_parse_ok"] = bool(cited) and ("<response>" in ans)
        elif v == "cot_noselect":
            _, response_only, _ = parse_cot_response(ans)
            entry["response_only"] = response_only
            entry["cot_parse_ok"] = "<response>" in ans
        with lock:
            results.append(entry)
            # Periodic checkpoint
            if len(results) % args.checkpoint_every == 0:
                _flush(out_path, results)
        return None

    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = [ex.submit(worker, i) for i in range(len(jobs))]
        for _ in tqdm(as_completed(futs), total=len(futs)):
            pass

    _flush(out_path, results)
    print(f"\nWrote {out_path}  (n={len(results)})", file=sys.stderr)


def _job_key(j: Dict[str, Any]) -> str:
    return f"{j['variant']}|{j.get('system')}|{j.get('k')}|{j['style']}|{j.get('sample_idx')}|{j.get('q_idx')}|{j.get('cluster_id')}"


def _rrkey(j: Dict[str, Any]) -> str:
    """Cache key for rewrite_retrievals: per (system, sample, qa-or-cluster)."""
    qid = j.get("q_idx") if j.get("q_idx") is not None else j.get("cluster_id")
    return f"{j['system']}|{j['style']}|{j['sample_idx']}|{qid}"


def _flush(path, results):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


if __name__ == "__main__":
    main()
