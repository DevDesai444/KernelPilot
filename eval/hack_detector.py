"""
hack_detector.py — Detect LLM-generated kernels that cheat on benchmarks.

Static checks (regex, pre-compilation):
  1. early_exit        — kernel skips work based on iteration count / env var
  2. input_poison      — writes sentinel values to inputs (corrupts reference data)
  3. null_output       — outputs zeros / constants without reading inputs
  4. cache_thrash      — flushes cache between timing calls to inflate speedup
  5. timing_hook       — reads clock/timer intrinsics to skip work during measurement
  6. nan_inf_output    — produces NaN/Inf (would pass if correctness check is skipped)
  7. pointer_alias     — copies output pointer back to input (in-place aliasing hack)
  8. precision_downgrade — computes in fp16 then upcasts (trades accuracy for speed)
  9. library_call      — delegates to cuBLAS/cuDNN/thrust instead of custom kernel

Runtime checks (compiled harness, post-compilation) live in runtime_checks.py:
  R1. no_op       — kernel writes nothing; output stays NaN sentinel
  R2. identity    — kernel copies input to output without normalizing
  R3. cache       — pointer-keyed caching (new malloc, re-run, compare)
  R4. nondeter    — non-deterministic output (shared memory overflow / race)
  R5. stream_inj  — near-zero CUDA event time (work deferred to another stream)
"""

from __future__ import annotations
import re
import logging

logger = logging.getLogger(__name__)


class HackDetectionResult:
    def __init__(self, passed: bool, hack_type: str, detail: str):
        self.passed    = passed      # True = clean, False = hack detected
        self.hack_type = hack_type   # e.g. "early_exit"
        self.detail    = detail      # human-readable explanation

    def __repr__(self):
        status = "CLEAN" if self.passed else f"HACK:{self.hack_type}"
        return f"HackDetectionResult({status}, {self.detail!r})"


def _check_early_exit(src: str) -> HackDetectionResult:
    """Kernel conditionally skips computation based on iteration/env state."""
    patterns = [
        r"getenv\s*\(",
        r"blockIdx\.x\s*==\s*0\s*&&\s*threadIdx\.x\s*==\s*0.*return",
        r"static\s+\w+\s+__call_count",
        r"__call_count\s*[+][+]",
        r"if\s*\(.*call_count.*\)\s*return",
        r"cudaGetDeviceProperties.*skipWork",
    ]
    for p in patterns:
        if re.search(p, src, re.DOTALL):
            return HackDetectionResult(False, "early_exit",
                f"Suspicious early-exit pattern: {p}")
    return HackDetectionResult(True, "early_exit", "clean")


def _check_input_poison(src: str) -> HackDetectionResult:
    """Kernel overwrites its own input buffers with constants."""
    # Writing to a __restrict__ input pointer is a red flag
    patterns = [
        r"input\s*\[.*\]\s*=",
        r"residual\s*\[.*\]\s*=\s*(__float2bfloat16\s*\(\s*0|0\.0|1\.0)",
        r"memset\s*\(\s*(input|residual|in|res)\b",
        r"cudaMemset\s*\(\s*(di|d_in|input)\b",
    ]
    for p in patterns:
        if re.search(p, src):
            return HackDetectionResult(False, "input_poison",
                f"Input buffer write detected: {p}")
    return HackDetectionResult(True, "input_poison", "clean")


