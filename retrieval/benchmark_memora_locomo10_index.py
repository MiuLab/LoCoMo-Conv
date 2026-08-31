#!/usr/bin/env python3
"""Benchmark Memora indexing on LoCoMo10 with a local OpenAI-compatible LLM.

The provided Zettabyte endpoint accepts unauthenticated HTTP requests, while the
OpenAI Python SDK always sends an Authorization header. This script patches
Memora's model wrapper to call the endpoint with requests directly, and patches
the embedding wrapper to use a local SentenceTransformer model.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import requests
from omegaconf import OmegaConf
from pydantic import ValidationError


DEFAULT_MEMORA_SRC = "third_party/Memora/src"
DEFAULT_DATA = "baselines/AnchorMem/data/locomo10.json"
DEFAULT_ENDPOINT = (
    "http://localhost:8000/v1"
    "job-4780661c5641-20260703031902/proxy/5000/v1/chat/completions"
)
DEFAULT_CHAT_MODEL = "../gemma-4-31B-it"
DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass
class LLMStats:
    calls: int = 0
    wall_time: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    parse_retries: int = 0


@dataclass
class AddRecord:
    user_id: str
    sample_id: str
    session: str
    turns: int
    words: int
    extracted_entries: int
    memory_count_before: int
    memory_count_after: int
    wall_time: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--memora-src", default=DEFAULT_MEMORA_SRC)
    parser.add_argument("--data-path", default=DEFAULT_DATA)
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--chat-model", default=DEFAULT_CHAT_MODEL)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBED_MODEL)
    parser.add_argument("--persist-path", default="baselines/AnchorMem/indices/memora_locomo10_local_index")
    parser.add_argument("--output", default="baselines/AnchorMem/outputs_memora_locomo10_index_timing.json")
    parser.add_argument("--limit-samples", type=int, default=0)
    parser.add_argument("--limit-sessions", type=int, default=0)
    parser.add_argument("--request-timeout", type=float, default=300.0)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--no-cue-index", action="store_true")
    parser.add_argument("--enable-episodic", action="store_true")
    return parser.parse_args()


def strip_json(text: str) -> Any:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    decoder = json.JSONDecoder()
    for idx, char in enumerate(stripped):
        if char == "{":
            return decoder.raw_decode(stripped[idx:])[0]
    raise ValueError(f"No JSON object found in response: {text[:300]}")


def schema_instruction(response_format: Any) -> str:
    schema = response_format.model_json_schema()
    return (
        "\n\nReturn ONLY a valid JSON object. Do not use markdown. "
        "The JSON must conform to this schema:\n"
        f"{json.dumps(schema, ensure_ascii=False)}\n"
    )


def make_patched_invoke(endpoint: str, model: str, timeout: float, default_max_tokens: int, stats: LLMStats):
    # Optional bearer auth for hosted OpenAI-compatible endpoints (e.g. NVIDIA Build).
    # Self-hosted vLLM ignores the header, so this is safe to send unconditionally when set.
    _api_key = os.environ.get("MEMORA_ENDPOINT_API_KEY") or os.environ.get("NVIDIA_API_KEY") or ""
    _headers = {"Content-Type": "application/json"}
    if _api_key:
        _headers["Authorization"] = f"Bearer {_api_key}"

    def invoke(self, input, prompt_args=None, response_format=None, source="Unknown", **kwargs):
        if isinstance(input, str):
            formatted = input.format(**prompt_args) if prompt_args else input
            messages = [{"role": "user", "content": formatted}]
        elif isinstance(input, list):
            messages = input
        else:
            raise ValueError("Input must be a string or list of chat messages.")

        max_tokens = int(kwargs.pop("max_tokens", kwargs.pop("max_new_tokens", default_max_tokens)))
        temperature = float(kwargs.pop("temperature", 0.0))

        if response_format is not None:
            messages = [dict(m) for m in messages]
            last = messages[-1]
            if isinstance(last.get("content"), str):
                last["content"] = last["content"] + schema_instruction(response_format)
            else:
                raise ValueError("Structured parsing for multimodal messages is not supported in this benchmark.")

        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if response_format is not None:
            payload["response_format"] = {"type": "json_object"}

        attempts = 3 if response_format is not None else 1
        last_error: Exception | None = None
        for attempt in range(attempts):
            started = time.perf_counter()
            response = requests.post(
                endpoint,
                headers=_headers,
                data=json.dumps(payload),
                timeout=timeout,
            )
            elapsed = time.perf_counter() - started
            stats.calls += 1
            stats.wall_time += elapsed
            response.raise_for_status()
            body = response.json()
            usage = body.get("usage") or {}
            stats.prompt_tokens += int(usage.get("prompt_tokens") or 0)
            stats.completion_tokens += int(usage.get("completion_tokens") or 0)
            stats.total_tokens += int(usage.get("total_tokens") or 0)
            content = body["choices"][0]["message"]["content"]

            if response_format is None:
                return content

            try:
                parsed = strip_json(content)
                return response_format.model_validate(parsed)
            except (ValueError, ValidationError) as exc:
                last_error = exc
                stats.parse_retries += 1
                payload["messages"] = messages + [
                    {
                        "role": "assistant",
                        "content": content,
                    },
                    {
                        "role": "user",
                        "content": (
                            "The previous response did not validate. Return ONLY valid JSON "
                            "that conforms exactly to the schema."
                        ),
                    },
                ]

        raise RuntimeError(f"Failed to parse structured LLM response after retries: {last_error}")

    return invoke


def patch_memora(args: argparse.Namespace, stats: LLMStats):
    sys.path.insert(0, str(Path(args.memora_src).resolve()))

    import memora.utils.llm as llm_mod
    import memora.utils.embedding as emb_mod
    from sentence_transformers import SentenceTransformer

    def patched_llm_init(self, cfg, token_usage_callback=None):
        self.cfg = cfg
        self.token_usage_callback = token_usage_callback
        self.model_type = "http_openai_compatible"
        self.client = None
        self.hf_model = None
        self.hf_tokenizer = None

    llm_mod.ChatCompletionModel.__init__ = patched_llm_init
    llm_mod.ChatCompletionModel.invoke = make_patched_invoke(
        args.endpoint,
        args.chat_model,
        args.request_timeout,
        args.max_tokens,
        stats,
    )

    st_model = SentenceTransformer(args.embedding_model, local_files_only=True)

    def patched_embedding_init(self, cfg, client=None):
        self.cfg = cfg
        self.client = None

    def patched_generate_embeddings(self, input):
        vectors = st_model.encode(
            list(input),
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vectors.astype(float).tolist()

    emb_mod.BaseEmbeddingModel.__init__ = patched_embedding_init
    emb_mod.BaseEmbeddingModel.generate_embeddings = patched_generate_embeddings


def make_config(args: argparse.Namespace):
    return OmegaConf.create(
        {
            "llm": {
                "model": args.chat_model,
                "seed": 42,
            },
            "openai": {
                "api_type": "openai",
                "api_key": "",
                "embedding_model": args.embedding_model,
                "model": args.chat_model,
                "llm_api_base": "",
                "llm_api_version": "",
                "embedding_api_base": "",
                "embedding_api_version": "",
                "embedding_deployment_name": args.embedding_model,
                "managed_identity": None,
            },
            "memory": {
                "memory_store": "locomo10_local_index",
                "persist_path": args.persist_path,
                "collection_name": "agent_memory",
                "distance": "cosine",
                "query_score_threshold": 0.4,
                "update_score_threshold": 2.0,
                "force_rebuild": False,
                "enhance_query": False,
                "return_history": False,
                "multimodal_support": False,
                "top_k": 10,
                "cue_top_k": 10,
                "enable_hybrid_search": False,
                "enable_segmentation": False,
                "enable_episodic_memory": bool(args.enable_episodic),
                "use_segments_as_episodic": True,
                "enable_cue_index": not bool(args.no_cue_index),
            },
            "retrieval": {"strategy": "semantic"},
            "eval": {"max_workers": 1},
        }
    )


def session_documents(samples: list[dict[str, Any]], limit_samples: int, limit_sessions: int):
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
            for turn in turns:
                speaker = str(turn.get("speaker", "")).strip()
                text = str(turn.get("text", "")).strip()
                line = f"{speaker}: {text}"
                if turn.get("blip_caption"):
                    line += f" [Image Description: {turn['blip_caption']}]"
                lines.append(line)
            doc = "\n".join(lines)
            yield user_id, sample_id, key, doc, timestamp, len(turns)
            emitted += 1
            if limit_sessions and emitted >= limit_sessions:
                return


def main() -> None:
    args = parse_args()
    data_path = Path(args.data_path)
    samples = json.loads(data_path.read_text(encoding="utf-8"))

    stats = LLMStats()
    patch_memora(args, stats)

    # Avoid importing the public facade here: it imports optional GRPO/PEFT
    # retrievers that are irrelevant for indexing and may not be installed.
    from memora.core.local_client import LocalMemoraClient as MemoraClient

    cfg = make_config(args)
    docs = list(session_documents(samples, args.limit_samples, args.limit_sessions))
    if not docs:
        raise RuntimeError("No LoCoMo10 sessions found.")

    clients: dict[str, Any] = {}
    records: list[AddRecord] = []
    total_started = time.perf_counter()

    print(
        json.dumps(
            {
                "event": "start",
                "sessions": len(docs),
                "samples": len({d[1] for d in docs}),
                "chat_model": args.chat_model,
                "embedding_model": args.embedding_model,
                "cue_index": not args.no_cue_index,
                "episodic": args.enable_episodic,
                "persist_path": args.persist_path,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    for i, (user_id, sample_id, session, doc, timestamp, turn_count) in enumerate(docs, 1):
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
                "segment_topic": session,
                "segment_index": i - 1,
            },
        )
        elapsed = time.perf_counter() - started
        after = client.count()
        rec = AddRecord(
            user_id=user_id,
            sample_id=sample_id,
            session=session,
            turns=turn_count,
            words=len(doc.split()),
            extracted_entries=len(entries),
            memory_count_before=before,
            memory_count_after=after,
            wall_time=elapsed,
        )
        records.append(rec)
        print(
            json.dumps(
                {
                    "event": "indexed_session",
                    "n": i,
                    "of": len(docs),
                    "sample_id": sample_id,
                    "session": session,
                    "turns": turn_count,
                    "entries": len(entries),
                    "count_delta": after - before,
                    "wall_time": round(elapsed, 3),
                    "llm_calls": stats.calls,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    total_elapsed = time.perf_counter() - total_started
    total_memories = sum(client.count() for client in clients.values())
    summary = {
        "data_path": str(data_path),
        "sessions_indexed": len(records),
        "samples_indexed": len({r.sample_id for r in records}),
        "total_turns": sum(r.turns for r in records),
        "total_words": sum(r.words for r in records),
        "total_extracted_entries_returned": sum(r.extracted_entries for r in records),
        "total_chroma_records": total_memories,
        "total_wall_time_sec": total_elapsed,
        "avg_wall_time_per_session_sec": total_elapsed / len(records),
        "llm": asdict(stats),
        "config": {
            "chat_endpoint": args.endpoint,
            "chat_model": args.chat_model,
            "embedding_model": args.embedding_model,
            "cue_index": not args.no_cue_index,
            "episodic": args.enable_episodic,
            "persist_path": args.persist_path,
        },
    }
    output = {"summary": summary, "records": [asdict(r) for r in records]}
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"event": "done", "summary": summary, "output": str(out_path)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
