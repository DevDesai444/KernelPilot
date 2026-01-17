"""
flashinfer_ref.py — FlashInfer-based reference outputs and baseline timing.

Uses FlashInfer's production CUDA kernels on B200 as the ground-truth
reference for WaferBench NVFP4, matching the official evaluation methodology.

Matches KernelArena bench_sustained convention:
  - 500 warmup, 100 timed reps, L2 cache cycling via input buffer rotation
  - silu_mul uses fused silu_and_mul_scaled_nvfp4_experts_quantize (not 2 calls)
"""

from __future__ import annotations
import logging
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

try:
    import torch
    import flashinfer
    _HAS_FLASHINFER = True
except ImportError:
    _HAS_FLASHINFER = False
    logger.warning("torch/flashinfer not available — will fall back to reference kernels")

# Check for the fused silu_mul expert API (KernelArena reference)
_HAS_FUSED_SILU = False
if _HAS_FLASHINFER:
    _HAS_FUSED_SILU = hasattr(flashinfer, "silu_and_mul_scaled_nvfp4_experts_quantize")
    if not _HAS_FUSED_SILU:
        logger.warning(
            "flashinfer.silu_and_mul_scaled_nvfp4_experts_quantize not found — "
            "silu_mul baseline will use unfused fallback (NOT comparable to KernelArena)"
        )

# Timing constants matching ThunderKittens 2.0 / WaferBench convention
_WARMUP_ITERS = 500
_BENCH_ITERS  = 100


