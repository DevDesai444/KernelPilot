"""
benchmark.py — Timing harness for WaferBench problem shapes.

Follows ThunderKittens 2.0 convention:
  - 500 warmup iterations
  - 100 timed reps via CUDA events
  - L2 cache input cycling to prevent cache hits inflating performance

CRITICAL: Both baseline (FlashInfer, via flashinfer_ref.py) and candidate
kernels are timed through Python/PyTorch dispatch to ensure symmetric
measurement overhead. This matches KernelArena's methodology where both
reference and candidate go through the same `bench_sustained` path.

Without this symmetry, the Python dispatch overhead (~5-15us per call)
inflates the baseline measurement while the C++ binary has near-zero
overhead, producing fake speedups of 3-4x on ~4us kernels.
"""

from __future__ import annotations
import hashlib
import logging
import math
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Optional
import time

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).parent.parent
DEBUG_BENCHMARK_KERNEL_SRC = os.getenv("RLM_DEBUG_BENCHMARK_KERNEL_SRC", "").strip().lower() in {
    "1", "true", "yes",
}

# Maximum input buffer copies for L2 cache cycling (KernelArena methodology)
_MAX_L2_CYCLE_BUFS = 256

# Timing constants matching ThunderKittens 2.0 / WaferBench convention
_WARMUP_ITERS = 500
_BENCH_ITERS  = 100


def geometric_mean(values: list) -> float:
    if not values:
        return 1.0
    log_sum = sum(math.log(v) for v in values if v > 0)
    return math.exp(log_sum / len(values))


class Benchmarker:

    def __init__(self, config: dict, kernel_type: str = "add_rmsnorm"):
        self.warmup      = config["eval"]["benchmark_warmup"]
        self.iters       = config["eval"]["benchmark_iters"]
        self.kernel_type = kernel_type
        self.include_dirs = [str(PROJECT_ROOT / 'kernels' / 'common'), str(PROJECT_ROOT)]

    @staticmethod
    def _compute_l2_cycle_bufs(input_bytes: int) -> int:
        """Compute nbufs to exceed 3x L2 cache (KernelArena methodology)."""
        import torch
        props = torch.cuda.get_device_properties(0)
        l2_size = props.L2_cache_size
        if input_bytes >= l2_size * 3:
            return 1
        n = int(l2_size * 3 / input_bytes) + 1
        return min(n, _MAX_L2_CYCLE_BUFS)

    @staticmethod
    def _cuda_harness_prelude() -> str:
        return r"""
#define CHECK_CUDA(call) do { \
    cudaError_t err__ = (call); \
    if (err__ != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err__)); \
        return 2; \
    } \
} while (0)
#define cudaMalloc(...) CHECK_CUDA(cudaMalloc(__VA_ARGS__))
#define cudaMemcpy(...) CHECK_CUDA(cudaMemcpy(__VA_ARGS__))
#define cudaMemset(...) CHECK_CUDA(cudaMemset(__VA_ARGS__))
#define cudaFree(...) CHECK_CUDA(cudaFree(__VA_ARGS__))
#define cudaStreamCreate(...) CHECK_CUDA(cudaStreamCreate(__VA_ARGS__))
#define cudaStreamSynchronize(...) CHECK_CUDA(cudaStreamSynchronize(__VA_ARGS__))
#define cudaDeviceSynchronize(...) CHECK_CUDA(cudaDeviceSynchronize())
#define cudaEventCreate(...) CHECK_CUDA(cudaEventCreate(__VA_ARGS__))
#define cudaEventRecord(...) CHECK_CUDA(cudaEventRecord(__VA_ARGS__))
#define cudaEventElapsedTime(...) CHECK_CUDA(cudaEventElapsedTime(__VA_ARGS__))
#define cudaEventDestroy(...) CHECK_CUDA(cudaEventDestroy(__VA_ARGS__))
#define cudaDeviceGetAttribute(...) CHECK_CUDA(cudaDeviceGetAttribute(__VA_ARGS__))
"""

    def _input_bytes(self, shape: tuple) -> int:
        """Compute input bytes for L2 cycling calculation."""
        if self.kernel_type == "add_rmsnorm":
            rows, hidden = shape
            return rows * hidden * 2 * 2 + hidden * 2
        elif self.kernel_type == "silu_mul":
            b, m, k = shape
            return b * m * k * 2 * 2
        elif self.kernel_type == "nvfp4_quantize":
            m, k = shape
            return m * k * 2
        return 4096  # fallback

    # ── Primary: Python-dispatch timing (symmetric with FlashInfer baseline) ──

    def _compile_and_time(self, kernel_src: str, shape: tuple) -> Optional[float]:
        """Compile candidate and time through Python dispatch.

        Uses torch.utils.cpp_extension.load_inline so both candidate and
        FlashInfer baseline go through the same PyTorch dispatch path,
        ensuring symmetric measurement overhead.
        """
        try:
            return self._time_via_extension(kernel_src, shape)
        except Exception as e:
            logger.warning("PyTorch extension timing failed (%s), trying C++ fallback", e)
            return self._time_via_binary(kernel_src, shape)

