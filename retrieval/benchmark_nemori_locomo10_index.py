#!/usr/bin/env python3
"""Benchmark Nemori indexing on LoCoMo10 with local embeddings.

The Zettabyte endpoint used for the chat model accepts unauthenticated
OpenAI-compatible HTTP requests. This script uses Nemori's core MemorySystem
pipeline but replaces PostgreSQL/Qdrant with small in-memory stores so the
measurement focuses on indexing, LLM generation, and local embedding time.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import re
import sys
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any

import requests


DEFAULT_NEMORI_SRC = "/private/tmp/nemori"
DEFAULT_DATA = "baselines/AnchorMem/data/locomo10.json"
DEFAULT_ENDPOINT = (
    "http://localhost:8000/v1"
    "job-c334860c04bd-20260702172613/proxy/5000/v1/chat/completions"
)
DEFAULT_CHAT_MODEL = "../gemma-4-31B-it"
DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass
class HTTPStats:
    calls: int = 0
    wall_time: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    json_repairs: int = 0
    json_parse_failures: int = 0


@dataclass
class EmbeddingStats:
    calls: int = 0
    batch_calls: int = 0
    texts: int = 0
    wall_time: float = 0.0


@dataclass
class IndexRecord:
    user_id: str
    sample_id: str
    session: str
    turns: int
    words: int
    episodes_created: int
    semantic_created: int
    fallback_episodes: int
    total_episodes_after: int
    total_semantic_after: int
    wall_time: float
    llm_calls_after: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nemori-src", default=DEFAULT_NEMORI_SRC)
    parser.add_argument("--data-path", default=DEFAULT_DATA)
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--chat-model", default=DEFAULT_CHAT_MODEL)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBED_MODEL)
    parser.add_argument("--output", default="baselines/AnchorMem/outputs_nemori_locomo10_index_timing.json")
    parser.add_argument("--agent-id", default="locomo10_nemori")
    parser.add_argument("--limit-samples", type=int, default=0)
    parser.add_argument("--limit-sessions", type=int, default=0)
    parser.add_argument("--request-timeout", type=float, default=300.0)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--respect-request-temperature", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--llm-max-concurrent", type=int, default=1)
    parser.add_argument("--batch-threshold", type=int, default=20)
    parser.add_argument("--episode-min-messages", type=int, default=2)
    parser.add_argument("--episode-max-messages", type=int, default=25)
    parser.add_argument("--no-batch-segmentation", action="store_true")
    parser.add_argument("--no-semantic", action="store_true")
    parser.add_argument("--prediction-correction", action="store_true")
    parser.add_argument("--episode-merging", action="store_true")
    return parser.parse_args()


def strip_json_object(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)

    decoder = json.JSONDecoder()
    for idx, char in enumerate(stripped):
        if char == "{":
            obj, _ = decoder.raw_decode(stripped[idx:])
            return json.dumps(obj, ensure_ascii=False)
    raise ValueError(f"No JSON object found in response: {text[:300]}")


class LocalHTTPChatProvider:
    supports_usage_tracking = True

    def __init__(
        self,
        endpoint: str,
        default_temperature: float,
        respect_request_temperature: bool,
        default_max_tokens: int,
        request_timeout: float,
        seed: int | None,
        stats: HTTPStats,
    ) -> None:
        self._endpoint = endpoint
        self._default_temperature = default_temperature
        self._respect_request_temperature = respect_request_temperature
        self._default_max_tokens = default_max_tokens
        self._request_timeout = request_timeout
        self._seed = seed
        self._stats = stats

    async def complete(self, messages: list[dict], **kwargs: object) -> str:
        content, _usage = await self.complete_with_usage(messages, **kwargs)
        return content

    async def complete_with_usage(
        self, messages: list[dict], **kwargs: object
    ) -> tuple[str, dict[str, int]]:
        return await asyncio.to_thread(self._complete_sync, messages, dict(kwargs))

    def _complete_sync(
        self, messages: list[dict], kwargs: dict[str, object]
    ) -> tuple[str, dict[str, int]]:
        temperature = (
            float(kwargs.get("temperature", self._default_temperature))
            if self._respect_request_temperature
            else self._default_temperature
        )
        max_tokens = int(kwargs.get("max_tokens", self._default_max_tokens))
        response_format = kwargs.get("response_format")
        payload: dict[str, Any] = {
            "model": str(kwargs.get("model") or DEFAULT_CHAT_MODEL),
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if self._seed is not None:
            payload["seed"] = self._seed
        if isinstance(response_format, dict):
            payload["response_format"] = response_format

        started = time.perf_counter()
        response = requests.post(
            self._endpoint,
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload, ensure_ascii=False),
            timeout=self._request_timeout,
        )
        elapsed = time.perf_counter() - started
        self._stats.calls += 1
        self._stats.wall_time += elapsed
        response.raise_for_status()

        body = response.json()
        usage = body.get("usage") or {}
        prompt_tokens = int(usage.get("prompt_tokens") or 0)
        completion_tokens = int(usage.get("completion_tokens") or 0)
        total_tokens = int(usage.get("total_tokens") or (prompt_tokens + completion_tokens))
        self._stats.prompt_tokens += prompt_tokens
        self._stats.completion_tokens += completion_tokens
        self._stats.total_tokens += total_tokens

        content = body["choices"][0]["message"]["content"]
        if isinstance(response_format, dict):
            try:
                normalized = strip_json_object(content)
                if normalized != content.strip():
                    self._stats.json_repairs += 1
                content = normalized
            except ValueError:
                self._stats.json_parse_failures += 1
                raise

        return content, {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }


class TimedMiniLMEmbedding:
    def __init__(self, model_name: str, stats: EmbeddingStats) -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name, local_files_only=True)
        self._stats = stats

    async def embed(self, text: str) -> list[float]:
        started = time.perf_counter()
        vector = await asyncio.to_thread(
            self._model.encode,
            [text],
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        elapsed = time.perf_counter() - started
        self._stats.calls += 1
        self._stats.texts += 1
        self._stats.wall_time += elapsed
        return vector[0].astype(float).tolist()

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        started = time.perf_counter()
        vectors = await asyncio.to_thread(
            self._model.encode,
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        elapsed = time.perf_counter() - started
        self._stats.batch_calls += 1
        self._stats.texts += len(texts)
        self._stats.wall_time += elapsed
        return vectors.astype(float).tolist()


class InMemoryBufferStore:
    def __init__(self) -> None:
        self._messages: dict[tuple[str, str], list[Any]] = {}
        self._next_id = 1

    async def push(self, user_id: str, agent_id: str, messages: list[Any]) -> None:
        key = (user_id, agent_id)
        bucket = self._messages.setdefault(key, [])
        for message in messages:
            message.metadata = dict(message.metadata)
            message.metadata["buffer_id"] = self._next_id
            self._next_id += 1
            bucket.append(message)

    async def get_unprocessed(self, user_id: str, agent_id: str) -> list[Any]:
        return list(self._messages.get((user_id, agent_id), []))

    async def mark_processed(self, user_id: str, agent_id: str, message_ids: list[int]) -> None:
        ids = set(message_ids)
        key = (user_id, agent_id)
        self._messages[key] = [
            message for message in self._messages.get(key, [])
            if message.metadata.get("buffer_id") not in ids
        ]

    async def count_unprocessed(self, user_id: str, agent_id: str) -> int:
        return len(self._messages.get((user_id, agent_id), []))


class InMemoryEpisodeStore:
    def __init__(self) -> None:
        self.items: dict[str, Any] = {}

    async def save(self, episode: Any) -> None:
        self.items[episode.id] = episode

    async def get(self, episode_id: str, user_id: str, agent_id: str) -> Any | None:
        episode = self.items.get(episode_id)
        if episode and episode.user_id == user_id and episode.agent_id == agent_id:
            return episode
        return None

    async def list_by_user(
        self, user_id: str, agent_id: str, limit: int = 100, offset: int = 0
    ) -> list[Any]:
        episodes = [
            ep for ep in self.items.values()
            if ep.user_id == user_id and ep.agent_id == agent_id
        ]
        episodes.sort(key=lambda ep: ep.created_at)
        return episodes[offset: offset + limit]

    async def delete(self, episode_id: str, user_id: str, agent_id: str) -> None:
        episode = await self.get(episode_id, user_id, agent_id)
        if episode:
            self.items.pop(episode_id, None)

    async def delete_by_user(self, user_id: str, agent_id: str) -> None:
        for episode_id, episode in list(self.items.items()):
            if episode.user_id == user_id and episode.agent_id == agent_id:
                self.items.pop(episode_id, None)

    async def search_by_text(
        self, user_id: str, agent_id: str, query: str, top_k: int
    ) -> list[Any]:
        query_lower = query.lower()
        hits = [
            ep for ep in self.items.values()
            if ep.user_id == user_id
            and ep.agent_id == agent_id
            and query_lower in f"{ep.title} {ep.content}".lower()
        ]
        return hits[:top_k]

    async def get_batch(
        self, episode_ids: list[str], user_id: str, agent_id: str
    ) -> list[Any]:
        episodes = []
        for episode_id in episode_ids:
            episode = await self.get(episode_id, user_id, agent_id)
            if episode:
                episodes.append(episode)
        return episodes


class InMemorySemanticStore:
    def __init__(self) -> None:
        self.items: dict[str, Any] = {}

    async def save(self, memory: Any) -> None:
        self.items[memory.id] = memory

    async def save_batch(self, memories: list[Any]) -> None:
        for memory in memories:
            self.items[memory.id] = memory

    async def get(self, memory_id: str, user_id: str, agent_id: str) -> Any | None:
        memory = self.items.get(memory_id)
        if memory and memory.user_id == user_id and memory.agent_id == agent_id:
            return memory
        return None

    async def list_by_user(
        self, user_id: str, agent_id: str, memory_type: str | None = None
    ) -> list[Any]:
        memories = [
            mem for mem in self.items.values()
            if mem.user_id == user_id and mem.agent_id == agent_id
        ]
        if memory_type:
            memories = [mem for mem in memories if mem.memory_type == memory_type]
        memories.sort(key=lambda mem: mem.created_at)
        return memories

    async def delete(self, memory_id: str, user_id: str, agent_id: str) -> None:
        memory = await self.get(memory_id, user_id, agent_id)
        if memory:
            self.items.pop(memory_id, None)

    async def delete_by_user(self, user_id: str, agent_id: str) -> None:
        for memory_id, memory in list(self.items.items()):
            if memory.user_id == user_id and memory.agent_id == agent_id:
                self.items.pop(memory_id, None)

    async def search_by_text(
        self, user_id: str, agent_id: str, query: str, top_k: int
    ) -> list[Any]:
        query_lower = query.lower()
        hits = [
            mem for mem in self.items.values()
            if mem.user_id == user_id
            and mem.agent_id == agent_id
            and query_lower in mem.content.lower()
        ]
        return hits[:top_k]

    async def get_batch(
        self, memory_ids: list[str], user_id: str, agent_id: str
    ) -> list[Any]:
        memories = []
        for memory_id in memory_ids:
            memory = await self.get(memory_id, user_id, agent_id)
            if memory:
                memories.append(memory)
        return memories


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right:
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


class InMemoryVectorStore:
    def __init__(self) -> None:
        self.episodes: dict[str, tuple[str, str, list[float]]] = {}
        self.semantic: dict[str, tuple[str, str, list[float]]] = {}

    def upsert_episode(self, episode_id: str, user_id: str, agent_id: str, embedding: list[float]) -> None:
        self.episodes[episode_id] = (user_id, agent_id, embedding)

    def upsert_semantic(self, memory_id: str, user_id: str, agent_id: str, embedding: list[float]) -> None:
        self.semantic[memory_id] = (user_id, agent_id, embedding)

    def search_episodes(
        self, user_id: str, agent_id: str, embedding: list[float], top_k: int
    ) -> list[dict[str, Any]]:
        return self._search(self.episodes, user_id, agent_id, embedding, top_k)

    def search_semantic(
        self, user_id: str, agent_id: str, embedding: list[float], top_k: int
    ) -> list[dict[str, Any]]:
        return self._search(self.semantic, user_id, agent_id, embedding, top_k)

    def _search(
        self,
        store: dict[str, tuple[str, str, list[float]]],
        user_id: str,
        agent_id: str,
        embedding: list[float],
        top_k: int,
    ) -> list[dict[str, Any]]:
        hits = []
        for item_id, (item_user, item_agent, item_embedding) in store.items():
            if item_user != user_id or item_agent != agent_id:
                continue
            score = cosine_similarity(embedding, item_embedding)
            hits.append({"id": item_id, "score": score})
        hits.sort(key=lambda hit: hit["score"], reverse=True)
        return hits[:top_k]

    def delete_episode(self, episode_id: str) -> None:
        self.episodes.pop(episode_id, None)

    def delete_semantic(self, memory_id: str) -> None:
        self.semantic.pop(memory_id, None)

    def delete_episodes_by_user(self, user_id: str, agent_id: str) -> None:
        for episode_id, (item_user, item_agent, _embedding) in list(self.episodes.items()):
            if item_user == user_id and item_agent == agent_id:
                self.episodes.pop(episode_id, None)

    def delete_semantic_by_user(self, user_id: str, agent_id: str) -> None:
        for memory_id, (item_user, item_agent, _embedding) in list(self.semantic.items()):
            if item_user == user_id and item_agent == agent_id:
                self.semantic.pop(memory_id, None)


class DummySearch:
    async def search(self, **_kwargs: Any) -> Any:
        return None


class TimeoutLLMOrchestrator:
    """Small wrapper that raises Nemori's per-request timeout for this benchmark."""

    def __init__(self, inner: Any, timeout: float) -> None:
        self._inner = inner
        self._timeout = timeout

    async def execute(self, request: Any) -> Any:
        request = replace(request, timeout=self._timeout)
        return await self._inner.execute(request)

    async def execute_batch(self, requests: list[Any]) -> list[Any]:
        return await self._inner.execute_batch([
            replace(request, timeout=self._timeout) for request in requests
        ])

    @property
    def stats(self) -> Any:
        return self._inner.stats


def parse_locomo_timestamp(text: str) -> datetime:
    cleaned = re.sub(r"\s+on\s+", " ", text.strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r"\b(am|pm)\b", lambda m: m.group(1).upper(), cleaned, flags=re.IGNORECASE)
    for fmt in ("%I:%M %p %d %B, %Y", "%I:%M%p %d %B, %Y"):
        try:
            return datetime.strptime(cleaned, fmt)
        except ValueError:
            pass
    return datetime.now()


def session_sort_key(key: str) -> int:
    match = re.fullmatch(r"session_(\d+)", key)
    return int(match.group(1)) if match else 1_000_000


def session_documents(samples: list[dict[str, Any]], limit_samples: int, limit_sessions: int):
    emitted = 0
    sample_iter = samples[:limit_samples] if limit_samples else samples
    for sample_idx, sample in enumerate(sample_iter):
        sample_id = str(sample.get("sample_id") or f"sample_{sample_idx}")
        user_id = f"locomo10_{sample_id}"
        conversation = sample.get("conversation", {})
        session_keys = [
            key for key, value in conversation.items()
            if re.fullmatch(r"session_\d+", key) and isinstance(value, list) and value
        ]
        for session in sorted(session_keys, key=session_sort_key):
            turns = conversation[session]
            timestamp_text = str(conversation.get(f"{session}_date_time", ""))
            timestamp = parse_locomo_timestamp(timestamp_text) if timestamp_text else datetime.now()
            messages = []
            words = 0
            for turn in turns:
                speaker = str(turn.get("speaker") or "speaker").strip()
                content = str(turn.get("text") or "").strip()
                if turn.get("blip_caption"):
                    content += f"\n[Image Description: {turn['blip_caption']}]"
                words += len(content.split())
                messages.append(
                    Message(
                        role=speaker,
                        content=content,
                        timestamp=timestamp,
                        metadata={
                            "sample_id": sample_id,
                            "session": session,
                            "timestamp_text": timestamp_text,
                            "dia_id": turn.get("dia_id"),
                        },
                    )
                )
            yield user_id, sample_id, session, messages, len(turns), words
            emitted += 1
            if limit_sessions and emitted >= limit_sessions:
                return


async def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    sys.path.insert(0, str(Path(args.nemori_src).resolve()))

    global Message
    from nemori.config import MemoryConfig
    from nemori.core.memory_system import MemorySystem
    from nemori.domain.models import Message
    from nemori.llm.generators.episode import EpisodeGenerator
    from nemori.llm.generators.merger import EpisodeMerger
    from nemori.llm.generators.semantic import SemanticGenerator
    from nemori.llm.orchestrator import LLMOrchestrator
    from nemori.services.event_bus import EventBus

    data_path = Path(args.data_path)
    samples = json.loads(data_path.read_text(encoding="utf-8"))
    docs = list(session_documents(samples, args.limit_samples, args.limit_sessions))
    if not docs:
        raise RuntimeError("No non-empty LoCoMo10 sessions found.")

    http_stats = HTTPStats()
    embedding_stats = EmbeddingStats()
    llm_provider = LocalHTTPChatProvider(
        endpoint=args.endpoint,
        default_temperature=args.temperature,
        respect_request_temperature=args.respect_request_temperature,
        default_max_tokens=args.max_tokens,
        request_timeout=args.request_timeout,
        seed=args.seed,
        stats=http_stats,
    )
    inner_orchestrator = LLMOrchestrator(
        provider=llm_provider,
        default_model=args.chat_model,
        max_concurrent=args.llm_max_concurrent,
    )
    orchestrator = TimeoutLLMOrchestrator(inner_orchestrator, timeout=args.request_timeout)
    embedding = TimedMiniLMEmbedding(args.embedding_model, embedding_stats)

    config = MemoryConfig(
        agent_id=args.agent_id,
        llm_model=args.chat_model,
        llm_timeout=args.request_timeout,
        llm_retries=3,
        llm_max_concurrent=args.llm_max_concurrent,
        embedding_model=args.embedding_model,
        embedding_dimension=384,
        buffer_size_min=1_000_000,
        buffer_size_max=1_000_001,
        enable_batch_segmentation=not args.no_batch_segmentation,
        batch_threshold=args.batch_threshold,
        episode_min_messages=args.episode_min_messages,
        episode_max_messages=args.episode_max_messages,
        enable_semantic_memory=not args.no_semantic,
        enable_prediction_correction=args.prediction_correction,
        enable_episode_merging=args.episode_merging,
    )

    episode_store = InMemoryEpisodeStore()
    semantic_store = InMemorySemanticStore()
    buffer_store = InMemoryBufferStore()
    qdrant = InMemoryVectorStore()
    episode_generator = EpisodeGenerator(orchestrator=orchestrator, embedding=embedding)
    semantic_generator = SemanticGenerator(
        orchestrator=orchestrator,
        embedding=embedding,
        enable_prediction_correction=config.enable_prediction_correction,
    )
    merger = (
        EpisodeMerger(
            orchestrator=orchestrator,
            embedding=embedding,
            episode_store=episode_store,
            qdrant=qdrant,
            similarity_threshold=config.merge_similarity_threshold,
            merge_top_k=config.merge_top_k,
        )
        if config.enable_episode_merging
        else None
    )
    system = MemorySystem(
        config=config,
        agent_id=args.agent_id,
        db=None,
        episode_store=episode_store,
        semantic_store=semantic_store,
        buffer_store=buffer_store,
        orchestrator=orchestrator,
        embedding=embedding,
        episode_generator=episode_generator,
        semantic_generator=semantic_generator,
        event_bus=EventBus(),
        search=DummySearch(),
        merger=merger,
        qdrant=qdrant,
    )

    print(
        json.dumps(
            {
                "event": "start",
                "system": "nemori",
                "sessions": len(docs),
                "samples": len({doc[1] for doc in docs}),
                "chat_model": args.chat_model,
                "embedding_model": args.embedding_model,
                "batch_segmentation": config.enable_batch_segmentation,
                "semantic": config.enable_semantic_memory,
                "prediction_correction": config.enable_prediction_correction,
                "episode_merging": config.enable_episode_merging,
                "temperature": args.temperature,
                "respect_request_temperature": args.respect_request_temperature,
                "seed": args.seed,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    records: list[IndexRecord] = []
    total_started = time.perf_counter()
    for i, (user_id, sample_id, session, messages, turns, words) in enumerate(docs, 1):
        episodes_before = len(episode_store.items)
        semantic_before = len(semantic_store.items)
        started = time.perf_counter()
        await system.add_messages(user_id, messages)
        episodes = await system.flush(user_id)
        elapsed = time.perf_counter() - started
        episodes_after = len(episode_store.items)
        semantic_after = len(semantic_store.items)
        fallback_episodes = sum(1 for episode in episodes if episode.metadata.get("fallback"))
        record = IndexRecord(
            user_id=user_id,
            sample_id=sample_id,
            session=session,
            turns=turns,
            words=words,
            episodes_created=episodes_after - episodes_before,
            semantic_created=semantic_after - semantic_before,
            fallback_episodes=fallback_episodes,
            total_episodes_after=episodes_after,
            total_semantic_after=semantic_after,
            wall_time=elapsed,
            llm_calls_after=http_stats.calls,
        )
        records.append(record)
        print(
            json.dumps(
                {
                    "event": "indexed_session",
                    "n": i,
                    "of": len(docs),
                    "sample_id": sample_id,
                    "session": session,
                    "turns": turns,
                    "episodes": record.episodes_created,
                    "semantic": record.semantic_created,
                    "fallback_episodes": fallback_episodes,
                    "wall_time": round(elapsed, 3),
                    "llm_calls": http_stats.calls,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    total_elapsed = time.perf_counter() - total_started
    orchestrator_stats = inner_orchestrator.stats
    summary = {
        "data_path": str(data_path),
        "sessions_indexed": len(records),
        "samples_indexed": len({record.sample_id for record in records}),
        "total_turns": sum(record.turns for record in records),
        "total_words": sum(record.words for record in records),
        "total_episodes": len(episode_store.items),
        "total_semantic_memories": len(semantic_store.items),
        "fallback_episodes": sum(record.fallback_episodes for record in records),
        "total_wall_time_sec": total_elapsed,
        "avg_wall_time_per_session_sec": total_elapsed / len(records),
        "llm": {
            **asdict(http_stats),
            "successful_requests": orchestrator_stats.total_requests,
            "orchestrator_errors": orchestrator_stats.total_errors,
            "avg_latency_ms": orchestrator_stats.avg_latency_ms,
            "requests_by_phase": orchestrator_stats.requests_by_phase,
            "tokens_by_phase": orchestrator_stats.tokens_by_phase,
        },
        "embedding": asdict(embedding_stats),
        "config": {
            "chat_endpoint": args.endpoint,
            "chat_model": args.chat_model,
            "embedding_model": args.embedding_model,
            "agent_id": args.agent_id,
            "batch_segmentation": config.enable_batch_segmentation,
            "batch_threshold": config.batch_threshold,
            "episode_min_messages": config.episode_min_messages,
            "semantic": config.enable_semantic_memory,
            "prediction_correction": config.enable_prediction_correction,
            "episode_merging": config.enable_episode_merging,
            "temperature": args.temperature,
            "respect_request_temperature": args.respect_request_temperature,
            "seed": args.seed,
            "llm_max_concurrent": args.llm_max_concurrent,
        },
    }
    # Serialize episode and semantic stores for post-hoc retrieval.
    episodes_export = []
    for ep_id, ep in episode_store.items.items():
        episodes_export.append({
            "id": ep.id,
            "user_id": ep.user_id,
            "agent_id": ep.agent_id,
            "title": ep.title,
            "content": ep.content,
            "source_messages": ep.source_messages,
            "metadata": ep.metadata,
            "created_at": ep.created_at.isoformat() if hasattr(ep.created_at, "isoformat") else str(ep.created_at),
        })
    semantic_export = []
    for m_id, mem in semantic_store.items.items():
        semantic_export.append({
            "id": mem.id,
            "user_id": mem.user_id,
            "agent_id": mem.agent_id,
            "content": mem.content,
            "memory_type": mem.memory_type,
            "source_episode_id": mem.source_episode_id,
            "confidence": mem.confidence,
            "metadata": mem.metadata,
            "created_at": mem.created_at.isoformat() if hasattr(mem.created_at, "isoformat") else str(mem.created_at),
        })
    stores_snapshot = {
        "episodes": episodes_export,
        "semantic_memories": semantic_export,
        "config": summary["config"],
    }

    return {"summary": summary, "records": [asdict(record) for record in records],
            "_stores_snapshot": stores_snapshot}


def main() -> None:
    args = parse_args()
    output = asyncio.run(run_benchmark(args))
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    stores_snapshot = output.pop("_stores_snapshot", None)
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    if stores_snapshot is not None:
        stores_path = out_path.parent / (out_path.stem + "_stores.json")
        stores_path.write_text(json.dumps(stores_snapshot, ensure_ascii=False), encoding="utf-8")
        print(f"[nemori] wrote stores snapshot to {stores_path}", flush=True)
    print(
        json.dumps(
            {"event": "done", "summary": output["summary"], "output": str(out_path)},
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
