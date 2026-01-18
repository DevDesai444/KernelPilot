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


def measure_baseline(kernel_type: str, shape: tuple) -> Optional[float]:
    """Measure FlashInfer baseline timing in microseconds.

    Returns None if FlashInfer is not available.
    """
    if not available():
        return None

    _jit_warmup()

    try:
        if kernel_type == "add_rmsnorm":
            return _baseline_add_rmsnorm(shape)
        elif kernel_type == "silu_mul":
            return _baseline_silu_mul(shape)
        elif kernel_type == "nvfp4_quantize":
            return _baseline_nvfp4_quantize(shape)
        else:
            logger.warning("Unknown kernel_type for FlashInfer baseline: %s", kernel_type)
            return None
    except Exception as e:
        logger.warning("FlashInfer baseline measurement failed: %s", e)
        return None


def measure_baseline_with_source(kernel_type: str, shape: tuple) -> tuple[Optional[float], str]:
    """Measure the official baseline and report where it came from."""
    baseline = measure_baseline(kernel_type, shape)
    if baseline is not None:
        return baseline, "flashinfer"
    return None, "unavailable"


def generate_reference(kernel_type: str, shape: tuple, seed: int = 42) -> Optional[dict]:
    """Generate reference output tensors using FlashInfer.

    Returns dict with input/output torch tensors, or None if unavailable.
    """
    if not available():
        return None

    try:
        if kernel_type == "add_rmsnorm":
            return _reference_add_rmsnorm(shape, seed)
        elif kernel_type == "silu_mul":
            return _reference_silu_mul(shape, seed)
        elif kernel_type == "nvfp4_quantize":
            return _reference_nvfp4_quantize(shape, seed)
        else:
            return None
    except Exception as e:
        logger.warning("FlashInfer reference generation failed: %s", e)
        return None


# ── Add + RMSNorm + FP4 Quant ────────────────────────────────────────────────

def _baseline_add_rmsnorm(shape: tuple) -> float:
    rows, hidden = shape
    input_bytes = rows * hidden * 2 * 2 + hidden * 2  # (input+residual)*bf16 + weight
    nbufs = _compute_l2_cycle_bufs(input_bytes)
    inps, ress = [], []
    for i in range(nbufs):
        torch.manual_seed(i)
        inps.append(torch.randn(rows, hidden, dtype=torch.bfloat16, device="cuda"))
        ress.append(torch.randn(rows, hidden, dtype=torch.bfloat16, device="cuda"))
    w = torch.ones(hidden, dtype=torch.bfloat16, device="cuda")

    def run(buf_idx):
        flashinfer.add_rmsnorm_fp4quant(inps[buf_idx], ress[buf_idx], w, eps=1e-6)

    return _time_fn(run, nbufs=nbufs)


def _reference_add_rmsnorm(shape: tuple, seed: int) -> dict:
    rows, hidden = shape
    torch.manual_seed(seed)
    inp = torch.randn(rows, hidden, dtype=torch.bfloat16, device="cuda")
    res = torch.randn(rows, hidden, dtype=torch.bfloat16, device="cuda")
    w   = torch.ones(hidden, dtype=torch.bfloat16, device="cuda") + \
          torch.rand(hidden, dtype=torch.bfloat16, device="cuda") * 0.1

    # FlashInfer's fused_add_rmsnorm modifies inp in-place (residual_out = inp + res)
    inp_copy = inp.clone()
    res_copy = res.clone()
    flashinfer.fused_add_rmsnorm(inp_copy, res_copy, w, eps=1e-6)

    return {
        "input": inp, "residual": res, "weight": w,
        "residual_out": res_copy, "norm_out": inp_copy,
    }


# ── SiLU × Mul + FP4 Quant ──────────────────────────────────────────────────

def _baseline_silu_mul(shape: tuple) -> float:
    b, m, k = shape
    global_scale = torch.tensor([1.0], dtype=torch.float32, device="cuda")

    if _HAS_FUSED_SILU:
        # KernelArena reference: single fused kernel (3D expert-batched, swizzled)
        input_bytes = b * m * 2 * k * 2  # (B, M, 2*K) bf16
        nbufs = _compute_l2_cycle_bufs(input_bytes)
        xs = []
        for i in range(nbufs):
            torch.manual_seed(i)
            xs.append(torch.randn(b, m, 2 * k, dtype=torch.bfloat16, device="cuda"))
        mask = torch.full((b,), m, dtype=torch.int64, device="cuda")

        def run(buf_idx):
            flashinfer.silu_and_mul_scaled_nvfp4_experts_quantize(
                xs[buf_idx], mask, global_scale
            )

        return _time_fn(run, nbufs=nbufs)
    else:
        # Fallback: two separate calls (NOT comparable to KernelArena)
        logger.warning("Using unfused silu_mul baseline — speedup NOT comparable to KernelArena")
        input_bytes = b * m * k * 2 * 2  # gate + up bf16
        nbufs = _compute_l2_cycle_bufs(input_bytes)
        combineds = []
        for i in range(nbufs):
            torch.manual_seed(i)
            gate = torch.randn(b * m, k, dtype=torch.bfloat16, device="cuda")
            up   = torch.randn(b * m, k, dtype=torch.bfloat16, device="cuda")
            combineds.append(torch.cat([gate, up], dim=-1))

        def run(buf_idx):
            out = flashinfer.silu_and_mul(combineds[buf_idx])
            flashinfer.fp4_quantize(out, global_scale=global_scale)

        return _time_fn(run, nbufs=nbufs)


