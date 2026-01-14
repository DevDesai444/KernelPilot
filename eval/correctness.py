"""
correctness.py — Numerical correctness verification.

Primary: validate against FlashInfer reference (production code path on B200).
Fallback: validate against hand-written CUDA reference kernel.
"""

from __future__ import annotations
import logging
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

from eval.hack_detector import is_clean
from eval import flashinfer_ref

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).parent.parent


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
#define cudaEventCreate(...) CHECK_CUDA(cudaEventCreate(__VA_ARGS__))
#define cudaEventRecord(...) CHECK_CUDA(cudaEventRecord(__VA_ARGS__))
#define cudaEventElapsedTime(...) CHECK_CUDA(cudaEventElapsedTime(__VA_ARGS__))
#define cudaEventDestroy(...) CHECK_CUDA(cudaEventDestroy(__VA_ARGS__))
"""


def _generate_flashinfer_harness(kernel_type: str, shape: tuple,
                                  ref_data_dir: str, atol: float, rtol: float) -> str:
    """Generate CUDA harness that loads FlashInfer reference outputs and compares."""
    if kernel_type == "add_rmsnorm":
        return _flashinfer_harness_add_rmsnorm(shape, ref_data_dir, atol, rtol)
    elif kernel_type == "nvfp4_quantize":
        return _flashinfer_harness_nvfp4_quantize(shape, ref_data_dir, atol, rtol)
    elif kernel_type == "silu_mul":
        return _flashinfer_harness_silu_mul(shape, ref_data_dir, atol, rtol)
    return None


