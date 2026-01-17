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


def _compute_l2_cycle_bufs(input_bytes: int, max_groups: int = 256) -> int:
    """Compute number of input groups to exceed 3x L2 cache (KernelArena methodology)."""
    if not _HAS_FLASHINFER:
        return 4
    props = torch.cuda.get_device_properties(0)
    l2_size = props.L2_cache_size  # bytes
    if input_bytes >= l2_size * 3:
        return 1
    n = int(l2_size * 3 / input_bytes) + 1
    return min(n, max_groups)


def available() -> bool:
    return _HAS_FLASHINFER and torch.cuda.is_available()


_jit_warmed_up = False

def _jit_warmup():
    """Trigger FlashInfer JIT compilation before any timing.

    FlashInfer uses Triton/TVM backends that JIT-compile on first call.
    This inflates the first baseline measurement if not pre-warmed.
    """
    global _jit_warmed_up
    if _jit_warmed_up:
        return
    logger.info("FlashInfer JIT warmup (first call triggers compilation)...")
    try:
        # Small tensors to trigger JIT with minimal time
        x = torch.randn(4, 256, dtype=torch.bfloat16, device="cuda")
        r = torch.randn(4, 256, dtype=torch.bfloat16, device="cuda")
        w = torch.ones(256, dtype=torch.bfloat16, device="cuda")
        flashinfer.add_rmsnorm_fp4quant(x, r, w, eps=1e-6)
        gs = torch.tensor([1.0], dtype=torch.float32, device="cuda")
        flashinfer.fp4_quantize(x, global_scale=gs)
        # Warm fused silu_mul expert API (KernelArena reference)
        if _HAS_FUSED_SILU:
            x3d = torch.randn(1, 4, 512, dtype=torch.bfloat16, device="cuda")
            mask = torch.full((1,), 4, dtype=torch.int64, device="cuda")
            flashinfer.silu_and_mul_scaled_nvfp4_experts_quantize(x3d, mask, gs)
        else:
            combined = torch.cat([x, r], dim=-1)
            flashinfer.silu_and_mul(combined)
        torch.cuda.synchronize()
    except Exception as e:
        logger.warning("JIT warmup partial failure (ok): %s", e)
    _jit_warmed_up = True
    logger.info("FlashInfer JIT warmup complete")


