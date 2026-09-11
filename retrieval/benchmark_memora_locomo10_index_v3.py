#!/usr/bin/env python3
"""Memora indexing v3 — turn-level source_dia_ids.

Every turn in the session doc is prefixed with a `[dia_id]` marker. The
extraction prompt is patched to ask the LLM to list all supporting `[dia_id]`
markers per memory in a `SOURCE:` suffix, which we then parse into the
MemoryEntry.original_text field so retrieval can recover per-memory turn
provenance (comparable to mem0's `dia_ids` metadata).

Falls back gracefully: if the LLM omits SOURCE for a memory, original_text
stores the session's dia_id list (session-level fallback).
"""

from __future__ import annotations
import argparse, importlib.util, json, re, sys, time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, List

import requests

DEFAULT_MEMORA_SRC = "third_party/Memora/src"
DEFAULT_DATA = "./data/locomo10.json"
DEFAULT_ENDPOINT = (
    "http://localhost:8000/v1"
    "job-4780661c5641-20260703031902/proxy/5000/v1/chat/completions"
)
DEFAULT_CHAT_MODEL = "../gemma-4-31B-it"
DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass
class AddRecord:
    user_id: str
    sample_id: str
    session: str
    turns: int
    words: int
    extracted_entries: int
    memories_with_source: int
    memory_count_before: int
    memory_count_after: int
    wall_time: float


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--memora-src", default=DEFAULT_MEMORA_SRC)
    ap.add_argument("--data-path", default=DEFAULT_DATA)
    ap.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    ap.add_argument("--chat-model", default=DEFAULT_CHAT_MODEL)
    ap.add_argument("--embedding-model", default=DEFAULT_EMBED_MODEL)
    ap.add_argument("--persist-path", default="./indices/memora_locomo10_v3_index")
    ap.add_argument("--output", default="./outputs_memora_locomo10_v3_index_timing.json")
    ap.add_argument("--limit-samples", type=int, default=0)
    ap.add_argument("--skip-samples", default="",
                    help="Comma-separated sample_ids to skip entirely (resume after a crash). "
                         "Skip whole conversations only — a partially-indexed conversation must be "
                         "redone, because LocalMemoraClient.clear() wipes the store on first touch.")
    ap.add_argument("--limit-sessions", type=int, default=0)
    ap.add_argument("--request-timeout", type=float, default=300.0)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--no-cue-index", action="store_true")
    ap.add_argument("--enable-episodic", action="store_true")
    return ap.parse_args()


def _import_v1():
    v1_path = Path(__file__).parent / "benchmark_memora_locomo10_index.py"
    spec = importlib.util.spec_from_file_location("bench_memora_v1", v1_path)
    m = importlib.util.module_from_spec(spec)
    sys.modules["bench_memora_v1"] = m
    spec.loader.exec_module(m)
    return m


SOURCE_MARKER_RE = re.compile(r"D\d+:\d+")
SOURCE_SUFFIX_RE = re.compile(r"\s*SOURCE\s*:\s*\[([^\]]*)\]\s*$", re.IGNORECASE)


PATCHED_PROMPT = """You are an expert factual memory extraction assistant. Your goal is to extract factual memories from a conversation segment.

# TASK:
Read the input conversation carefully, extract ALL factual memories that could be useful for future reference.

Every turn in the input is prefixed with a `[dia_id]` marker (like `[D1:5]`). For each memory
you extract, you MUST list ALL turn markers that support that memory in a `SOURCE:` suffix
appended to the end of the value.

Produce each memory as a key-value pair in the following format:

MemIndex: memory index for retrieval
MemValue: memory value with all the details supported directly from the given text. SOURCE:[D1:5,D1:7]

# GUIDELINES:
1. Content and Scope:
- Use only information explicitly mentioned in the context to create the factual memories.
- Capture ALL factual information that could be useful for future retrieval.
- Do not include greetings, small talk, or filler in the memories.
- Split distinct facts into separate entries.
- Capture all details about people's identities, experiences, past or upcoming events, intentions,
  hobbies, preferences, states, beliefs, goals, or future plans.
- Include time of events, location, and other contextual details when mentioned.

2. Format and Style:
- The MemIndex must be a short, human-readable phrase that is self-contained and unambiguous.
- Write MemValue as one or two full factual sentences, capturing all relevant details.
- Replace pronouns with specific names.
- Convert relative times using the conversation timestamp as reference.

3. SOURCE citation (REQUIRED):
- At the end of every MemValue, add `SOURCE:[dia_id_list]` listing every input turn marker
  (like [D1:5]) whose text supports this memory.
- If a memory is supported by turns [D1:5] and [D1:7], write `SOURCE:[D1:5,D1:7]`.
- If unsure which turn a fact came from, cite ALL plausible source turns.
- NEVER omit the SOURCE suffix. It is required for every extracted memory.

Timestamp of conversation: {timestamp}

Input Conversation:
{content}

Output:
"""


def _parse_source_suffix(value: str):
    """Return (cleaned_value, list_of_dia_ids) from a MemValue that may end in SOURCE:[...]"""
    m = SOURCE_SUFFIX_RE.search(value)
    if not m:
        # Fallback: scan the entire value for any dia_id markers
        found = SOURCE_MARKER_RE.findall(value)
        return value.strip(), sorted(set(found))
    inside = m.group(1)
    cleaned = value[: m.start()].strip()
    dia_ids = SOURCE_MARKER_RE.findall(inside)
    if not dia_ids:
        # Also scan cleaned in case
        dia_ids = SOURCE_MARKER_RE.findall(cleaned)
    return cleaned, sorted(set(dia_ids))


def session_documents_with_markers(samples, limit_samples, limit_sessions):
    emitted = 0
    sample_iter = samples[:limit_samples] if limit_samples else samples
    for sample_idx, sample in enumerate(sample_iter):
        sample_id = str(sample.get("sample_id") or f"sample_{sample_idx}")
        conversation = sample.get("conversation", {})
        user_id = f"locomo10_{sample_id}"
        for key in sorted(conversation.keys()):
            if not key.startswith("session_") or key.endswith("_date_time"):
                continue
            turns = conversation.get(key)
            if not isinstance(turns, list) or not turns:
                continue
            timestamp = str(conversation.get(f"{key}_date_time", ""))
            lines = [f"DATE: {timestamp}"]
            dia_ids = []
            for turn in turns:
                speaker = str(turn.get("speaker", "")).strip()
                text = str(turn.get("text", "")).strip()
                did = str(turn.get("dia_id", "")).strip()
                if did:
                    dia_ids.append(did)
                    line = f"[{did}] {speaker}: {text}"
                else:
                    line = f"{speaker}: {text}"
                if turn.get("blip_caption"):
                    line += f" [Image Description: {turn['blip_caption']}]"
                lines.append(line)
            doc = "\n".join(lines)
            yield user_id, sample_id, key, doc, timestamp, len(turns), dia_ids
            emitted += 1
            if limit_sessions and emitted >= limit_sessions:
                return


def _patch_memory_builder():
    """Monkey-patch memora's builder to (a) use PATCHED_PROMPT, (b) post-process
    each memory's MemValue to extract SOURCE:[dia_ids] into original_text and
    clean the value."""
    from memora.builder import memory_builder as mb
    from memora.utils import memory as mu

    mb.PROMPT_BUILD_MEMORY = PATCHED_PROMPT

    # Wrap convert_memory_output so that each MemoryEntry gets original_text set
    _orig_convert = mu.convert_memory_output

    def convert_with_source(memories, metadata, enable_cue_index):
        entries = _orig_convert(memories, metadata, enable_cue_index)
        session_dia_ids = metadata.get("_session_dia_ids", []) if metadata else []
        for entry in entries:
            cleaned, dia_ids = _parse_source_suffix(entry.value)
            if dia_ids:
                entry.value = cleaned
                entry.original_text = ";".join(dia_ids)
            else:
                # LLM omitted SOURCE — fallback to session-level
                entry.original_text = ";".join(session_dia_ids) if session_dia_ids else ""
        return entries

    mu.convert_memory_output = convert_with_source
    # Also patch the reference the ChatMemoryBuilder holds via `from ... import convert_memory_output`
    from memora.builder import chat_memory_builder as cmb
    cmb.convert_memory_output = convert_with_source


def _add_http_retry(args, attempts: int = 5, base_delay: float = 5.0):
    """Retry transient endpoint failures (502/503/504/524, timeouts, resets).

    The vLLM proxy intermittently drops connections; without this a single 524
    aborts a multi-hour index build and the partial store cannot be resumed
    (LocalMemoraClient.clear() wipes a user's store on first touch).
    """
    import requests as _rq
    import memora.utils.llm as llm_mod

    _inner = llm_mod.ChatCompletionModel.invoke

    def _retrying(self, *a, **kw):
        last = None
        for attempt in range(attempts):
            try:
                return _inner(self, *a, **kw)
            except (_rq.exceptions.RequestException, ConnectionError) as e:
                last = e
                delay = base_delay * (2 ** attempt)
                print(json.dumps({"event": "http_retry", "attempt": attempt + 1,
                                  "of": attempts, "sleep": delay,
                                  "error": str(e)[:200]}), flush=True)
                time.sleep(delay)
        raise last if last else RuntimeError("invoke failed with no exception recorded")

    llm_mod.ChatCompletionModel.invoke = _retrying


def main():
    args = parse_args()
    data_path = Path(args.data_path)
    samples = json.loads(data_path.read_text(encoding="utf-8"))

    v1 = _import_v1()
    stats = v1.LLMStats()
    v1.patch_memora(args, stats)
    _add_http_retry(args)

    _patch_memory_builder()

    from memora.core.local_client import LocalMemoraClient as MemoraClient

    cfg = v1.make_config(args)
    docs = list(session_documents_with_markers(samples, args.limit_samples, args.limit_sessions))
    skip = {x.strip() for x in args.skip_samples.split(",") if x.strip()}
    if skip:
        before_n = len(docs)
        docs = [d for d in docs if d[1] not in skip]
        print(json.dumps({"event": "resume", "skipped_samples": sorted(skip),
                          "sessions_skipped": before_n - len(docs),
                          "sessions_to_run": len(docs)}), flush=True)
    if not docs:
        raise RuntimeError("No sessions.")

    clients: dict[str, Any] = {}
    records: List[AddRecord] = []
    total_started = time.perf_counter()

    print(json.dumps({
        "event": "start", "system": "memora_v3",
        "sessions": len(docs), "samples": len({d[1] for d in docs}),
        "persist_path": args.persist_path,
    }), flush=True)

    for i, (user_id, sample_id, session, doc, timestamp, turn_count, dia_ids) in enumerate(docs, 1):
        if user_id not in clients:
            clients[user_id] = MemoraClient(cfg=cfg, user_id=user_id)
            clients[user_id].clear()
        client = clients[user_id]

        before = client.count()
        started = time.perf_counter()
        entries = client.add(
            doc,
            type="chat",
            metadata={
                "timestamp": timestamp,
                "sample_id": sample_id,
                "session": session,
                "_session_dia_ids": dia_ids,   # <-- fallback for LLM omissions
            },
        )
        elapsed = time.perf_counter() - started
        after = client.count()
        n_with_source = sum(1 for e in entries if e.original_text and any(
            not d.startswith(dia_ids[0].split(":")[0]) or True for d in e.original_text.split(";")
        )) if dia_ids else 0

        record = AddRecord(
            user_id=user_id, sample_id=sample_id, session=session,
            turns=turn_count, words=len(doc.split()),
            extracted_entries=len(entries),
            memories_with_source=sum(1 for e in entries if e.original_text),
            memory_count_before=before, memory_count_after=after,
            wall_time=elapsed,
        )
        records.append(record)
        print(json.dumps({
            "event": "indexed", "n": i, "of": len(docs),
            "sample_id": sample_id, "session": session,
            "turns": turn_count, "dia_ids": len(dia_ids),
            "extracted": len(entries), "with_source": record.memories_with_source,
            "wall_time": round(elapsed, 3),
            "llm_calls": stats.calls,
        }), flush=True)

    total_elapsed = time.perf_counter() - total_started
    summary = {
        "data_path": str(data_path),
        "sessions_indexed": len(records),
        "samples_indexed": len({r.sample_id for r in records}),
        "total_turns": sum(r.turns for r in records),
        "total_extracted": sum(r.extracted_entries for r in records),
        "total_with_source": sum(r.memories_with_source for r in records),
        "total_wall_time_sec": total_elapsed,
        "avg_wall_time_per_session_sec": total_elapsed / len(records),
        "llm": asdict(stats),
        "config": {
            "chat_endpoint": args.endpoint,
            "chat_model": args.chat_model,
            "embedding_model": args.embedding_model,
            "cue_index": not args.no_cue_index,
            "persist_path": args.persist_path,
        },
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"summary": summary, "records": [asdict(r) for r in records]}, indent=2))
    print(json.dumps({"event": "done", "output": str(out_path), "summary": summary}), flush=True)


if __name__ == "__main__":
    main()
