from __future__ import annotations

import json
import hashlib
import logging
import os
import re
from dataclasses import dataclass

from .env_loader import load_project_env
from .query_embedder import embed_query

logger = logging.getLogger(__name__)

try:
    from pinecone import Pinecone
except ImportError:  # pragma: no cover - optional dependency
    Pinecone = None

HARDWARE_ALIASES = {
    "blackwell": ("blackwell", "b200", "sm100", "sm 100", "sm100a", "sm 100a"),
}

OP_ALIASES = {
    "rmsnorm": ("add rmsnorm", "rmsnorm", "rms norm", "layernorm", "layer norm"),
    "silu_mul": ("silu mul", "silu_mul", "swiglu", "gated silu"),
    "quantize": ("nvfp4", "fp4", "quant", "quantize", "quantization"),
}

EXACT_OP_PATTERNS = {
    "add_rmsnorm_fp4": (
        "add rmsnorm",
        "residual add",
        "rmsnorm",
        "fp4",
        "nvfp4",
        "quant",
    ),
}

PATTERN_ALIASES = {
    "vectorized_stores": ("vectorized stores", "vector stores", "uint4 stores", "stg 128", "stg128"),
    "vectorized_loads": ("vectorized loads", "vector loads", "uint4 loads", "ldg 128", "ldg128"),
    "register_pressure": ("register pressure", "registers", "occupancy"),
    "warp_reduction": ("warp reduction", "warp reduce", "shuffle reduction"),
}

SOURCE_QUALITY_WEIGHTS = {
    "flashinfer": 1.0,
    "vllm": 1.0,
    "sglang": 1.0,
    "cutlass": 0.95,
    "triton": 0.95,
    "pytorch": 0.9,
    "apex": 0.9,
    "lmdeploy": 0.85,
    "sakana": 0.40,  # synthetic competition data; penalise heavily to surface real production code
}


@dataclass
class PineconeMatch:
    match_id: str
    score: float
    text: str
    title: str = ""
    source: str = ""
    metadata: dict | None = None

    def to_legacy_dict(self) -> dict:
        return {
            "path": self.source,
            "content": self.text,
            "title": self.title or self.match_id,
            "metadata": self.metadata or {},
            "score": self.score,
        }


class PineconeRetriever:
    def __init__(self, config: dict | None = None):
        load_project_env()
        cfg = config or {}
        self.enabled = str(cfg.get("provider", "pinecone")).lower() == "pinecone"
        self.top_k = int(cfg.get("top_k", 4))
        self.namespace = cfg.get("namespace") or os.getenv(
            cfg.get("namespace_env", "PINECONE_NAMESPACE")
        )
        self.fields = list(
            cfg.get(
                "fields",
                [
                    "chunk_text",
                    "title",
                    "source",
                    "source_code",
                    "source_file",
                    "optimization_pattern",
                    "op_type",
                ],
            )
        )
        self.text_field = cfg.get("text_field", "chunk_text")
        self.title_field = cfg.get("title_field", "title")
        self.source_field = cfg.get("source_field", "source")
        self.api_key_env = cfg.get("api_key_env", "PINECONE_API_KEY")
        self.index_host_env = cfg.get("index_host_env", "PINECONE_INDEX_HOST")
        self.index_name_env = cfg.get("index_name_env", "PINECONE_INDEX_NAME")
        self.embed_provider_env = cfg.get("embed_provider_env", "PINECONE_EMBED_PROVIDER")
        self.embed_model_env = cfg.get("embed_model_env", "PINECONE_EMBED_MODEL")
        self.rerank_model_env = cfg.get("rerank_model_env", "PINECONE_RERANK_MODEL")
        self.default_filter = cfg.get("metadata_filter") or None
        self.index_name = cfg.get("index_name") or os.getenv(self.index_name_env)
        self.index_host = cfg.get("index_host") or None
        self.embed_provider = cfg.get("embed_provider") or os.getenv(
            self.embed_provider_env, "sentence-transformers"
        )
        self.embed_model = cfg.get("embed_model") or os.getenv(
            self.embed_model_env, "sentence-transformers/all-MiniLM-L6-v2"
        )
        self.rerank_model = cfg.get("rerank_model") or os.getenv(self.rerank_model_env, "")
        self.rerank_pool = int(cfg.get("rerank_pool", 12))
        self.candidate_pool_multiplier = int(cfg.get("candidate_pool_multiplier", 3))
        self.source_cap = int(cfg.get("source_cap", 2))
        self.format_max_chars = int(cfg.get("format_max_chars", 16000))
        self.match_max_chars = int(cfg.get("match_max_chars", 4000))
        self.last_query_mode = "uninitialized"

        self._client = None
        self._index = None
        self._init_error = ""

    def _ensure_index(self):
        if not self.enabled:
            self._init_error = "RAG provider is disabled."
            return None

        if self._index is not None:
            return self._index
        if self._init_error:
            return None
        if Pinecone is None:
            self._init_error = "The pinecone package is not installed."
            return None

        api_key = os.getenv(self.api_key_env)
        index_host = self.index_host or os.getenv(self.index_host_env)
        if not api_key or not (index_host or self.index_name):
            self._init_error = (
                f"Missing environment variables: {self.api_key_env} and either "
                f"{self.index_host_env} or {self.index_name_env}."
            )
            return None

        try:
            self._client = Pinecone(api_key=api_key)
            if index_host:
                self._index = self._client.Index(host=index_host)
            else:
                self._index = self._client.Index(self.index_name)
        except Exception as exc:  # pragma: no cover - network runtime
            self._init_error = f"Failed to initialize Pinecone index: {exc}"
            logger.warning(self._init_error)
            return None

        return self._index

    def list_indexes(self) -> list[str]:
        if Pinecone is None:
            return []
        api_key = os.getenv(self.api_key_env)
        if not api_key:
            return []
        try:
            client = self._client or Pinecone(api_key=api_key)
            listing = client.list_indexes()
        except Exception as exc:  # pragma: no cover - network runtime
            logger.warning("Pinecone list_indexes failed: %s", exc)
            return []

        names = []
        for item in listing:
            if isinstance(item, dict):
                name = item.get("name")
            else:
                name = getattr(item, "name", None)
                if name is None and hasattr(item, "to_dict"):
                    try:
                        name = item.to_dict().get("name")
                    except Exception:
                        name = None
            if name:
                names.append(str(name))
        return names

    def status(self) -> str:
        if self._ensure_index() is not None:
            return "configured"
        return self._init_error or "not configured"

    def search(
        self,
        query: str,
        top_k: int | None = None,
        namespace: str | None = None,
        metadata_filter: dict | None = None,
        exclude_source_patterns: list[str] | None = None,
    ) -> list[PineconeMatch]:
        if not query.strip():
            return []

        index = self._ensure_index()
        if index is None:
            return []

        effective_filter = metadata_filter if metadata_filter is not None else self.default_filter
        search_query = {
            "inputs": {"text": query.strip()},
            "top_k": int(top_k or self.top_k),
        }
        if effective_filter:
            search_query["filter"] = effective_filter

        try:
            response = index.search(
                namespace=namespace or self.namespace or "__default__",
                query=search_query,
                fields=self.fields,
            )
            self.last_query_mode = "text"
        except Exception as exc:  # pragma: no cover - network runtime
            if self._is_integrated_inference_error(exc):
                logger.info(
                    "Pinecone text search unavailable for %r; falling back to vector query using %s",
                    query,
                    self.embed_model,
                )
                return self._search_by_vector(
                    query=query,
                    top_k=top_k,
                    namespace=namespace,
                    metadata_filter=metadata_filter,
                    exclude_source_patterns=exclude_source_patterns,
                )
            logger.warning("Pinecone search failed for query %r: %s", query, exc)
            self.last_query_mode = "text_error"
            return []

        hits = self._extract_hits(response)
        matches = []
        for hit in hits:
            fields = hit.get("fields", {}) or {}
            metadata = hit.get("metadata", {}) or {}
            source = self._extract_source(fields=fields, metadata=metadata)
            if exclude_source_patterns and any(
                pat.lower() in source.lower() for pat in exclude_source_patterns
            ):
                continue
            text = self._extract_text(fields=fields, metadata=metadata)
            if not text:
                continue
            matches.append(
                PineconeMatch(
                    match_id=str(hit.get("_id") or hit.get("id") or "unknown"),
                    score=float(hit.get("_score") or hit.get("score") or 0.0),
                    text=text,
                    title=self._extract_title(fields=fields, metadata=metadata, hit=hit),
                    source=source,
                    metadata=metadata,
                )
            )

        return self._rerank_matches(query, matches, top_n=int(top_k or self.top_k))

    def search_many(
        self,
        queries: list[str],
        top_k: int | None = None,
        namespace: str | None = None,
        metadata_filter: dict | None = None,
        exclude_source_patterns: list[str] | None = None,
    ) -> list[PineconeMatch]:
        deduped = {}
        candidate_top_k = max(
            int(top_k or self.top_k) * self.candidate_pool_multiplier,
            self.rerank_pool,
        )
        for query in queries:
            for variant in self._expand_query_variants(query):
                for match in self.search(
                    query=variant,
                    top_k=candidate_top_k,
                    namespace=namespace,
                    metadata_filter=metadata_filter,
                    exclude_source_patterns=exclude_source_patterns,
                ):
                    existing = deduped.get(match.match_id)
                    if existing is None or match.score > existing.score:
                        deduped[match.match_id] = match
        if not deduped:
            return []

        candidates = list(deduped.values())
        if exclude_source_patterns:
            before = len(candidates)
            candidates = [
                m for m in candidates
                if not any(pat.lower() in (m.source or "").lower() for pat in exclude_source_patterns)
            ]
            logger.info(
                "RAG exclude_source_patterns %s removed %d/%d candidates",
                exclude_source_patterns, before - len(candidates), before,
            )

        if not candidates:
            return []

        combined_query = " ; ".join(str(query).strip() for query in queries if str(query).strip())
        return self._rerank_matches(
            combined_query,
            candidates,
            top_n=int(top_k or self.top_k),
        )

    def format_matches(self, matches: list[PineconeMatch], max_chars: int | None = None) -> str:
        if not matches:
            return f"No Pinecone matches. Retriever status: {self.status()}"

        max_chars = int(max_chars or self.format_max_chars)
        parts = []
        total_chars = 0
        for idx, match in enumerate(matches, start=1):
            header = f"[{idx}] {match.title or match.match_id}"
            if match.source:
                header += f" | {match.source}"

            tags = self._format_metadata_tags(match.metadata or {})
            prefix = f"{header}\nScore: {match.score:.3f}\n"
            if tags:
                prefix += f"Tags: {tags}\n"
            remaining = max_chars - total_chars
            if remaining <= len(prefix):
                break

            body_budget = min(remaining - len(prefix), self.match_max_chars)
            body = self._truncate_text(match.text.strip().replace("\r", ""), body_budget)
            if not body:
                continue

            block = f"{prefix}```cuda\n{body}\n```"
            parts.append(block)
            total_chars += len(block)

        return "\n\n".join(parts)

    def get_top_k(self, query: str, k: int = 1):
        return [match.to_legacy_dict() for match in self.search(query, top_k=k)]

    def _search_by_vector(
        self,
        query: str,
        top_k: int | None = None,
        namespace: str | None = None,
        metadata_filter: dict | None = None,
        exclude_source_patterns: list[str] | None = None,
    ) -> list[PineconeMatch]:
        index = self._ensure_index()
        if index is None:
            return []

        try:
            vector = embed_query(
                text=query.strip(),
                provider=self.embed_provider,
                model_name=self.embed_model,
            )
        except Exception as exc:  # pragma: no cover - optional dependency/runtime
            logger.warning("Query embedding failed for %r: %s", query, exc)
            self.last_query_mode = "vector_error"
            return []

        try:
            response = index.query(
                vector=vector,
                top_k=int(top_k or self.top_k),
                namespace=namespace or self.namespace or "__default__",
                include_metadata=True,
                include_values=False,
                filter=metadata_filter if metadata_filter is not None else self.default_filter,
            )
            self.last_query_mode = f"vector:{self.embed_model}"
        except Exception as exc:  # pragma: no cover - network runtime
            logger.warning("Pinecone vector query failed for %r: %s", query, exc)
            self.last_query_mode = "vector_error"
            return []

        hits = self._extract_hits(response)
        matches = []
        for hit in hits:
            metadata = hit.get("metadata", {}) or {}
            source = self._extract_source(fields={}, metadata=metadata)
            if exclude_source_patterns and any(
                pat.lower() in source.lower() for pat in exclude_source_patterns
            ):
                continue
            text = self._extract_text(fields={}, metadata=metadata)
            if not text:
                continue
            matches.append(
                PineconeMatch(
                    match_id=str(hit.get("_id") or hit.get("id") or "unknown"),
                    score=float(hit.get("_score") or hit.get("score") or 0.0),
                    text=text,
                    title=self._extract_title(fields={}, metadata=metadata, hit=hit),
                    source=source,
                    metadata=metadata,
                )
            )
        return self._rerank_matches(query, matches, top_n=int(top_k or self.top_k))

    def _extract_text(self, fields: dict, metadata: dict) -> str:
        for key in self._text_candidates():
            value = fields.get(key) or metadata.get(key)
            if value:
                return str(value).strip()
        return ""

    def _extract_title(self, fields: dict, metadata: dict, hit: dict) -> str:
        for key in self._title_candidates():
            value = fields.get(key) or metadata.get(key)
            if value:
                return str(value).strip()
        return str(hit.get("_id") or hit.get("id") or "").strip()

    def _extract_source(self, fields: dict, metadata: dict) -> str:
        for key in self._source_candidates():
            value = fields.get(key) or metadata.get(key)
            if value:
                return str(value).strip()
        return ""

    def _text_candidates(self) -> list[str]:
        return [
            self.text_field,
            "source_code",
            "chunk_text",
            "text",
            "content",
            "body",
            "code",
        ]

    def _title_candidates(self) -> list[str]:
        return [
            self.title_field,
            "title",
            "source_file",
            "op_type",
            "optimization_pattern",
        ]

    def _source_candidates(self) -> list[str]:
        return [
            self.source_field,
            "source",
            "source_file",
        ]

    def _format_metadata_tags(self, metadata: dict) -> str:
        tags = []
        for key in ("hardware_target", "op_type", "optimization_pattern"):
            value = metadata.get(key)
            if value:
                tags.append(f"{key}={value}")
        return ", ".join(tags[:3])

    def _truncate_text(self, text: str, max_len: int) -> str:
        if max_len <= 0:
            return ""
        if len(text) <= max_len:
            return text
        if max_len <= 3:
            return text[:max_len]
        return text[: max_len - 3].rstrip() + "..."

    def _rerank_matches(
        self,
        query: str,
        matches: list[PineconeMatch],
        top_n: int,
    ) -> list[PineconeMatch]:
        if not matches:
            return []

        reranked = self._pinecone_rerank(query, matches, top_n=top_n)
        if reranked is not None:
            logger.info(
                "RAG FULL RANKING (%d candidates → top %d) for query %r:",
                len(matches), top_n, query[:80],
            )
            for rank, m in enumerate(reranked, start=1):
                flag = " ← RETURNED" if rank <= top_n else ""
                logger.info(
                    "  [%2d] score=%.4f  %-50s  %s%s",
                    rank, m.score, (m.title or m.match_id)[:50], m.source[:40] if m.source else "", flag,
                )
            return reranked

        scored = sorted(
            matches,
            key=lambda match: self._heuristic_rank_score(query, match),
            reverse=True,
        )
        logger.info(
            "RAG HEURISTIC RANKING (%d candidates → top %d) for query %r:",
            len(scored), top_n, query[:80],
        )
        for rank, m in enumerate(scored, start=1):
            flag = " ← RETURNED" if rank <= top_n else ""
            logger.info(
                "  [%2d] score=%.4f  %-50s  %s%s",
                rank, self._heuristic_rank_score(query, m), (m.title or m.match_id)[:50],
                m.source[:40] if m.source else "", flag,
            )
        scored = self._diversify_matches(scored, top_n=top_n)
        if self.last_query_mode != "uninitialized" and "+heuristic" not in self.last_query_mode:
            self.last_query_mode = f"{self.last_query_mode}+heuristic+diverse"
        return scored[:top_n]

