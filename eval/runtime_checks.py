"""
runtime_checks.py — Post-compilation runtime hack detection.

Five checks that static regex cannot catch (per Herdora field guide):
  R1. no_op       — kernel writes nothing; output stays NaN sentinel
  R2. identity    — kernel copies input → output without normalizing
  R3. cache       — kernel caches results keyed by output pointer address
  R4. nondeter    — non-deterministic output (shared memory overflow / race)
  R5. stream_inj  — near-zero CUDA event time on a non-trivial shape

Each check compiles and runs a small CUDA harness (rows=32, hidden=256)
and parses structured output lines: "RTCHECK <name>: PASS|FAIL <detail>"
"""

from __future__ import annotations
import logging
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).parent.parent

# Small shape for speed — enough data to be meaningful, fast to compile/run
_ROWS   = 32
_HIDDEN = 256


def _cuda_harness_prelude() -> str:
    return r"""
#define CHECK_CUDA(call) do { \
    cudaError_t err__ = (call); \
    if (err__ != cudaSuccess) { \
        fprintf(stderr, "RTCHECK internal CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err__)); \
        exit(2); \
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


@dataclass
class RuntimeCheckResult:
    passed:     bool
    check_name: str
    detail:     str

    def __repr__(self):
        status = "CLEAN" if self.passed else f"HACK:{self.check_name}"
        return f"RuntimeCheckResult({status}, {self.detail!r})"


