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
