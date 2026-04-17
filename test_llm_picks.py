"""
test_llm_picks.py — Test whether the LLM can pick the right strategies for each kernel.

Two modes:
  menu:     LLM picks 4 from a predefined list of 11 strategies
  freeform: LLM proposes 4 optimizations from scratch, no menu at all

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    python3 test_llm_picks.py --model sonnet                    # menu mode (default)
    python3 test_llm_picks.py --model sonnet --mode freeform    # no menu, open-ended
    python3 test_llm_picks.py --model all --mode freeform       # compare all 3 freeform
"""

import argparse
import json
import re
import time
from pathlib import Path

import anthropic

PROJECT_ROOT = Path(__file__).parent

# ── Model IDs ────────────────────────────────────────────────────────────────

MODELS = {
    "haiku":  "claude-haiku-4-5-20251001",
    "sonnet": "claude-sonnet-4-6",
    "opus":   "claude-opus-4-6",
}

# ── Strategy descriptions (for menu mode) ────────────────────────────────────

STRATEGY_DESCRIPTIONS = {
    "vectorize_loads": {
        "desc": "Replace scalar loads with 128-bit float4/uint4 vectorized transactions",
        "applicable": ["add_rmsnorm", "silu_mul", "nvfp4_quantize"],
    },
    "tma_prefetch": {
        "desc": "Use Blackwell TMA engine for async bulk copy with double buffering. "
                "Best for 2D/3D tiled access patterns, NOT simple linear access",
        "applicable": [],
    },
    "warp_reduction": {
        "desc": "Replace __syncthreads block reductions with __shfl_xor_sync warp shuffles",
        "applicable": ["add_rmsnorm"],
    },
    "fuse_passes": {
        "desc": "Combine multiple global memory passes into a single kernel loop. "
                "Only useful when kernel reads the same data in multiple passes",
        "applicable": ["add_rmsnorm"],
    },
    "register_tiling": {
        "desc": "Process multiple elements per thread using register arrays for ILP",
        "applicable": ["add_rmsnorm", "silu_mul"],
    },
    "async_pipeline": {
        "desc": "Overlap memory and compute using cp.async with double buffering. "
                "Best for kernels with loop-carried dependencies to overlap",
        "applicable": [],
    },
    "fp4_lut": {
        "desc": "Replace arithmetic FP4 quantization with a lookup table. "
                "FP4 has only 16 possible output values — a LUT eliminates all quantize math",
        "applicable": ["add_rmsnorm", "silu_mul", "nvfp4_quantize"],
    },
    "fast_math_expf": {
        "desc": "Replace slow expf/logf libcalls (~20 cycles) with hardware SFU intrinsics "
                "__expf/__logf (~4 cycles). Precision loss invisible when output goes to FP4",
        "applicable": ["silu_mul"],
    },
    "thread_coarsening": {
        "desc": "Increase work per thread (multiple quant blocks or rows per thread). "
                "Amortizes thread launch overhead when per-thread work is too small",
        "applicable": ["add_rmsnorm", "silu_mul", "nvfp4_quantize"],
    },
    "ldg_readonly": {
        "desc": "Route read-only inputs through L1 texture cache using __ldg(). "
                "Uses a separate cache path, freeing normal L1 for read-write data",
        "applicable": ["add_rmsnorm", "silu_mul", "nvfp4_quantize"],
    },
    "vectorized_stores": {
        "desc": "Replace byte-by-byte packed FP4 output stores with uint2/uint4 writes. "
                "Reduces store instruction count by 8x",
        "applicable": ["nvfp4_quantize", "silu_mul", "add_rmsnorm"],
    },
}

# ── Kernel sources ────────────────────────────────────────────────────────────

KERNELS = {
    "add_rmsnorm": PROJECT_ROOT / "kernels" / "reference" / "add_rmsnorm.cu",
    "silu_mul":    PROJECT_ROOT / "kernels" / "reference" / "silu_mul.cu",
    "nvfp4_quantize": PROJECT_ROOT / "kernels" / "reference" / "nvfp4_quantize.cu",
}

# ── Prompts ───────────────────────────────────────────────────────────────────

