"""
combiner.py — Merge top-K beam variants into a single optimized kernel.
Used as fallback when LLM-based combination fails.
"""

from __future__ import annotations
import re
import logging
from rlm.environment import KernelCandidate

logger = logging.getLogger(__name__)


def extract_kernel_body(cuda_src: str) -> str:
    match = re.search(r"__global__[^{]+\{", cuda_src, re.DOTALL)
    if not match:
        return cuda_src
    start = match.end()
    depth = 1
    pos   = start
    while pos < len(cuda_src) and depth > 0:
        if cuda_src[pos] == "{":   depth += 1
        elif cuda_src[pos] == "}": depth -= 1
        pos += 1
    return cuda_src[start:pos-1].strip()


def extract_shared_memory_decls(cuda_src: str) -> list:
    return re.findall(r"__shared__[^;]+;", cuda_src)


def naive_merge(variant_a: KernelCandidate, variant_b: KernelCandidate) -> str:
    """Conservative fallback: keep the better parent rather than emit broken merged CUDA."""
    winner = variant_a if variant_a.speedup >= variant_b.speedup else variant_b
    loser = variant_b if winner is variant_a else variant_a
    logger.warning(
        "Naive merge fallback selected better parent instead of attempting an unsafe textual merge: %s over %s",
        winner.strategy,
        loser.strategy,
    )
    return (
        f"// Combine fallback kept the better parent unchanged.\n"
        f"// Selected: {winner.strategy} ({winner.speedup:.3f}x)\n"
        f"// Rejected merge target: {loser.strategy} ({loser.speedup:.3f}x)\n"
        f"// Reason: naive textual merging is unsafe for nontrivial CUDA kernels.\n\n"
        f"{winner.code}"
    )
