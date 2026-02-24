from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

KERNEL_QUERY_CONTEXT = {
    "add_rmsnorm": {
        "operation": "fused add rmsnorm fp4 quantization",
        "aliases": ["rmsnorm", "residual add", "fp4 quantization", "layernorm"],
    },
    "silu_mul": {
        "operation": "fused silu mul fp4 quantization",
        "aliases": ["silu", "swiglu", "gated silu", "fp4 quantization"],
    },
    "nvfp4_quantize": {
        "operation": "nvfp4 block quantization",
        "aliases": ["fp4 quantization", "nvfp4", "bf16 to fp4", "packing"],
    },
}

OBSERVATION_COMPILER_KEYS = (
    "registers_per_thread",
    "spill_stores_bytes",
    "spill_loads_bytes",
    "static_smem_bytes",
    "cmem_bytes",
    "stack_frame_bytes",
)


@dataclass
class SandboxFeedback:
    status: str
    stage: str
    route: str
    confidence: float
    speedup: float
    parent_speedup: float
    uncertainty: str
    next_action: str
    action_type: str
    evidence: list[dict] = field(default_factory=list)
    observations: dict = field(default_factory=dict)
    hypothesis_test: dict = field(default_factory=dict)
    memory: dict = field(default_factory=dict)
    rag_queries: list[str] = field(default_factory=list)
    rag_filters: dict = field(default_factory=dict)
    preserve: list[str] = field(default_factory=list)
    revert: list[str] = field(default_factory=list)
    avoid: list[str] = field(default_factory=list)
    focus: list[str] = field(default_factory=list)
    success_criteria: list[str] = field(default_factory=list)
    abort_if: list[str] = field(default_factory=list)
    error: str = ""

