"""
engine.py — RLM core engine.
Orchestrates root LLM decomposition, parallel sub-LLM beam generation, and refinement.
"""

from __future__ import annotations
import asyncio
import json
import logging
import re
from pathlib import Path

import anthropic
from anthropic import AsyncAnthropic

from .coder import build_coder_prompt
from .environment import RLMEnvironment, KernelCandidate
from .feedback import build_sandbox_feedback
from .feedback import _kernel_aliases, _kernel_operation_phrase
from .fixer import build_fixer_prompt
from .planner import (
    build_initial_plan_prompt,
    build_tree_plan_prompt,
    fallback_branches,
    parse_plan_response,
)
from .rag_retriever import init_knowledge_base
from .root_prompts import SYSTEM_PROMPT, combine_prompt
from .reflector import (
    _get_launch_signature,
    _format_profile_section,
)

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
ALLOWED_READ_ROOTS = (
    (PROJECT_ROOT / "kernels" / "common").resolve(),
    (PROJECT_ROOT / "kernels" / "reference").resolve(),
)


# ── Refinement: tool-use agent loop ────────────────────────────────────────

MAX_INNER_TURNS = 5

SUBMIT_KERNEL_TOOL = {
    "name": "submit_kernel",
    "description": (
        "Submit optimized CUDA kernel for compilation, correctness checking, "
        "and profiling.\n\n"
        "Returns one of:\n"
        "- COMPILE ERROR: first error with file:line plus surrounding context\n"
        "- CORRECTNESS FAILURE: max error magnitude and which check failed\n"
        "- Result verdict (IMPROVED / REGRESSION / NO CHANGE) with:\n"
        "  timing_us, speedup vs baseline, SM occupancy,\n"
        "  compiler resource usage, register spills,\n"
        "  delta from your previous submission,\n"
        "  remaining optimization suggestions from profiler data"
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "cuda_code": {
                "type": "string",
                "description": (
                    "Complete .cu file content with all #includes, "
                    "kernel functions, and the launch_* wrapper."
                ),
            }
        },
        "required": ["cuda_code"],
    },
}

READ_FILE_TOOL = {
    "name": "read_file",
    "description": (
        "Read a source file from the project. Available files:\n"
        "- kernels/common/nvfp4_utils.cuh — FP4/FP8 quantization helpers, pack/unpack\n"
        "- kernels/common/b200_intrinsics.cuh — Blackwell TMA, TMEM, pipeline wrappers\n"
        "- kernels/reference/add_rmsnorm.cu — Naive Add+RMSNorm+FP4 reference kernel\n"
        "- kernels/reference/silu_mul.cu — Naive SiLU*Mul+FP4 reference kernel\n"
        "- kernels/reference/nvfp4_quantize.cu — Naive BF16→FP4 reference kernel\n"
        "Costs no submit_kernel turn."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Relative path from project root (e.g. 'kernels/common/nvfp4_utils.cuh')",
            }
        },
        "required": ["path"],
    },
}

SEARCH_DOCS_TOOL = {
    "name": "search_docs",
    "description": (
        "Search CUDA intrinsic documentation. Query by keyword to find correct "
        "function signatures, headers, and usage examples. Covers: FP4/FP8 conversion "
        "(cuda_fp4.h, cuda_fp8.h), warp intrinsics (shuffle, reduction), fast math "
        "(SFU), memory intrinsics (ldg, stcg, async copy), bfloat16/half operations.\n"
        "Example queries: 'fp4 convert float', 'fp8 e4m3 to float', 'warp reduction', "
        "'fast reciprocal sqrt', 'bfloat16 pair load'\n"
        "Costs no submit_kernel turn."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search keywords (e.g. 'fp4 quantize float', 'e4m3 convert', 'warp reduce')",
            }
        },
        "required": ["query"],
    },
}

SEARCH_PINECONE_TOOL = {
    "name": "search_pinecone",
    "description": (
        "Search the Pinecone knowledge index for CUDA optimization notes, prior "
        "experiments, compiler pitfalls, and kernel-specific guidance that the user "
        "already stored there. Use this when the local docs are not enough. "
        "Costs no submit_kernel turn."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Semantic search query to run against Pinecone.",
            },
            "top_k": {
                "type": "integer",
                "description": "Optional number of matches to return.",
            },
        },
        "required": ["query"],
    },
}

REFINE_TOOLS = [
    SUBMIT_KERNEL_TOOL,
    READ_FILE_TOOL,
    SEARCH_DOCS_TOOL,
]

REFINE_SYSTEM_PROMPT = f"""\
You are a CUDA kernel optimization agent. You have {{turns}} submit_kernel calls.

Your speedup is measured against FlashInfer, a production GPU library.
You target a single GPU (B200, sm_100a) and a single problem shape — use this to your advantage.

Available tools (only submit_kernel counts toward your turn limit):
- submit_kernel: compile, test correctness, and benchmark your kernel
- read_file: read project header files (nvfp4_utils.cuh, b200_intrinsics.cuh) or reference kernels
- search_docs: look up CUDA intrinsic signatures and usage (fp4, fp8, warp, fast math, memory)

Target hardware — NVIDIA B200 (sm_100a, Blackwell):
- HBM3e: 8 TB/s bandwidth, 192 GB
- L2 cache: 126 MB — benchmark uses L2 cache cycling (data is COLD every iteration)
- 148 SMs, 228 KB shared memory per SM, 255 registers per thread
- 128-bit load/store = uint4 = 8 bf16 values per transaction
- Use read_file to check available hardware intrinsics in the project headers

Before EVERY submit_kernel call, explain in 2-3 sentences:
1. What the latest measured result confirms or leaves uncertain
2. What specific code change you will make and why you expect it to help

Rules:
{{constraint}}
- NEVER put __syncthreads() inside if/else branches (deadlock).
- The launch_* function signature must match exactly.
- Output must match reference within atol=1e-2.
- Keep original #include directives, not expanded headers.
- Do not use torch headers (torch/extension.h, ATen, c10).
"""

