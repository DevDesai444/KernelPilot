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


def _check_null_output(src: str) -> HackDetectionResult:
    """Kernel writes a constant (0, 1) to output without reading inputs."""
    # Output filled with constant and no bfloat16 conversion from input
    writes_zero = bool(re.search(
        r"output\s*\[.*\]\s*=\s*(__float2bfloat16\s*\(\s*0\.0|0\s*;)", src))
    reads_input = bool(re.search(
        r"__bfloat162float\s*\(\s*(input|in|residual|res)\s*\[", src))
    if writes_zero and not reads_input:
        return HackDetectionResult(False, "null_output",
            "Output filled with constant and input never read")
    return HackDetectionResult(True, "null_output", "clean")


def _check_cache_thrash(src: str) -> HackDetectionResult:
    """Kernel explicitly flushes caches between iterations to inflate speedup."""
    patterns = [
        r"cudaDeviceSynchronize\s*\(\s*\).*cudaMemset",
        r"__threadfence_system",
        r"asm\s*volatile.*\"discard\"",
        r"cudaMemPrefetchAsync.*cudaCpuDeviceId",
    ]
    for p in patterns:
        if re.search(p, src, re.DOTALL):
            return HackDetectionResult(False, "cache_thrash",
                f"Cache flush pattern detected: {p}")
    return HackDetectionResult(True, "cache_thrash", "clean")


def _check_timing_hook(src: str) -> HackDetectionResult:
    """Kernel reads GPU clock to detect when it's being timed and skips work."""
    patterns = [
        r"\bclock\s*\(\s*\)",
        r"\bclock64\s*\(\s*\)",
        r"__clock\b",
        r"globaltimer",
        r"asm.*\"mov.*%clock\"",
        r"asm.*\"mov.*%clock64\"",
    ]
    for p in patterns:
        if re.search(p, src):
            return HackDetectionResult(False, "timing_hook",
                f"GPU clock intrinsic detected: {p}")
    return HackDetectionResult(True, "timing_hook", "clean")


def _check_nan_inf_output(src: str) -> HackDetectionResult:
    """Kernel intentionally writes NaN or Inf (exploit if correctness skipped)."""
    patterns = [
        r"\bNAN\b",
        r"\bINFINITY\b",
        r"\bHUGE_VALF\b",
        r"__int_as_float\s*\(\s*0x7f800000\s*\)",   # +Inf bit pattern
        r"__int_as_float\s*\(\s*0x7fc00000\s*\)",   # NaN bit pattern
        r"1\.0f\s*/\s*0\.0f",
    ]
    for p in patterns:
        if re.search(p, src):
            return HackDetectionResult(False, "nan_inf_output",
                f"NaN/Inf literal detected: {p}")
    return HackDetectionResult(True, "nan_inf_output", "clean")


def _check_pointer_alias(src: str) -> HackDetectionResult:
    """Kernel copies output pointer to input to create false in-place aliasing."""
    patterns = [
        r"(input|in)\s*=\s*(output|out)\s*;",
        r"(residual|res)\s*=\s*(output|out)\s*;",
        r"const_cast\s*<.*>\s*\(\s*(output|out)\s*\)",
    ]
    for p in patterns:
        if re.search(p, src):
            return HackDetectionResult(False, "pointer_alias",
                f"Pointer aliasing detected: {p}")
    return HackDetectionResult(True, "pointer_alias", "clean")


def _check_precision_downgrade(src: str) -> HackDetectionResult:
    """Kernel downcasts to fp16 mid-computation then upcasts back to bf16/fp32."""
    patterns = [
        r"__float2half\s*\(",
        r"__half2float\s*\(",
        r"\bhalf\b.*=.*\bfloat\b",
        r"__nv_half",
        r"fp16.*accumul",
        r"half_t",
    ]
    for p in patterns:
        if re.search(p, src):
            return HackDetectionResult(False, "precision_downgrade",
                f"fp16 intermediate detected (precision downgrade): {p}")
    return HackDetectionResult(True, "precision_downgrade", "clean")


def _check_library_call(src: str) -> HackDetectionResult:
    """Kernel delegates to optimized library instead of implementing the op."""
    patterns = [
        r"\bcublas\w+\s*\(",
        r"\bcublasLt\w+\s*\(",
        r"\bcudnn\w+\s*\(",
        r"\bcufft\w+\s*\(",
        r"\bthrust::",
        r"#include\s*[<\"]cublas",
        r"#include\s*[<\"]cudnn",
        r"#include\s*[<\"]cufft",
    ]
    for p in patterns:
        if re.search(p, src):
            return HackDetectionResult(False, "library_call",
                f"External library call detected: {p}")
    return HackDetectionResult(True, "library_call", "clean")


# All 9 static checks in order
_CHECKS = [
    _check_early_exit,
    _check_input_poison,
    _check_null_output,
    _check_cache_thrash,
    _check_timing_hook,
    _check_nan_inf_output,
    _check_pointer_alias,
    _check_precision_downgrade,
    _check_library_call,
]


def detect_hacks(kernel_src: str) -> list[HackDetectionResult]:
    """Run all 7 checks. Returns list of results (one per check)."""
    results = [check(kernel_src) for check in _CHECKS]
    failed  = [r for r in results if not r.passed]
    if failed:
        for r in failed:
            logger.warning("HACK DETECTED [%s]: %s", r.hack_type, r.detail)
    return results


def is_clean(kernel_src: str) -> tuple[bool, str]:
    """
    Convenience wrapper. Returns (clean, hack_type_or_empty).
    hack_type is empty string if clean.
    """
    results = detect_hacks(kernel_src)
    failed  = [r for r in results if not r.passed]
    if failed:
        return False, failed[0].hack_type
    return True, ""
