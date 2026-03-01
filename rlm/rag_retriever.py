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

