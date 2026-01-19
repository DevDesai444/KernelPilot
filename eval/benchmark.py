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


