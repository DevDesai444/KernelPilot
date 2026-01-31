"""
strategy_bank.py — Predefined optimization strategies with metadata.
Includes kernel-type-aware selection to pick the best 4 per kernel.
"""

from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class Strategy:
    name: str
    display_name: str
    description: str
    targets_bottleneck: list
    priority: int
    applicable_kernels: list = field(default_factory=list)  # empty = all
    requires: list = field(default_factory=list)
    conflicts: list = field(default_factory=list)

    def applies_to(self, bottleneck: str) -> bool:
        return bottleneck in self.targets_bottleneck or not self.targets_bottleneck

    def applies_to_kernel(self, kernel_type: str) -> bool:
        return not self.applicable_kernels or kernel_type in self.applicable_kernels


STRATEGY_BANK: dict = {
    # ── Original strategies (kept, with kernel applicability) ────────────────
    "vectorize_loads": Strategy(
        name="vectorize_loads",
        display_name="Float4 Vectorized Loads",
        description="Replace scalar loads with 128-bit float4/uint4 vectorized transactions",
        targets_bottleneck=["memory_bound", "latency_bound"],
        priority=10,
        applicable_kernels=["add_rmsnorm", "silu_mul", "nvfp4_quantize"],
    ),
    "tma_prefetch": Strategy(
        name="tma_prefetch",
        display_name="TMA Async Prefetch",
        description="Use Blackwell TMA engine for async bulk copy with double buffering",
        targets_bottleneck=["memory_bound", "latency_bound"],
        priority=4,
        applicable_kernels=[],  # not useful for any of the 3 kernel types
    ),
    "warp_reduction": Strategy(
        name="warp_reduction",
        display_name="Warp Shuffle Reduction",
        description="Replace __syncthreads reductions with __shfl_xor_sync",
        targets_bottleneck=["sync_bound"],
        priority=5,
        applicable_kernels=["add_rmsnorm"],  # only kernel with a reduction
    ),
    "fuse_passes": Strategy(
        name="fuse_passes",
        display_name="Fuse Memory Passes",
        description="Combine multiple global memory passes into a single kernel loop",
        targets_bottleneck=["memory_bound"],
        priority=9,
        applicable_kernels=["add_rmsnorm"],  # only kernel with 2 passes
        conflicts=["async_pipeline"],
    ),
    "register_tiling": Strategy(
        name="register_tiling",
        display_name="Register-Level Tiling",
        description="Process multiple elements per thread using register arrays for ILP",
        targets_bottleneck=["compute_bound", "latency_bound"],
        priority=6,
        applicable_kernels=["add_rmsnorm", "silu_mul"],
    ),
    "async_pipeline": Strategy(
        name="async_pipeline",
        display_name="Async Software Pipeline",
        description="Overlap memory and compute using cp.async with double buffering",
        targets_bottleneck=["latency_bound", "memory_bound"],
        priority=3,
        applicable_kernels=[],  # not useful for simple elementwise kernels
        conflicts=["fuse_passes"],
    ),
    # ── New strategies ───────────────────────────────────────────────────────
    "fp4_lut": Strategy(
        name="fp4_lut",
        display_name="FP4 Quantization Lookup Table",
        description="Replace arithmetic FP4 encode with a precomputed lookup table. "
                    "FP4 has only 16 output values — a 256-entry __constant__ LUT indexed "
                    "by the top 8 bits of the scaled value eliminates all quantize math",
        targets_bottleneck=["compute_bound", "latency_bound", "memory_bound"],
        priority=10,
        applicable_kernels=["add_rmsnorm", "silu_mul", "nvfp4_quantize"],
    ),
    "fast_math_expf": Strategy(
        name="fast_math_expf",
        display_name="Fast Math Intrinsics",
        description="Replace slow expf/logf libcalls with hardware SFU intrinsics "
                    "(__expf, __logf, __frcp_rn). ~4 cycles vs ~20 cycles. "
                    "Precision loss is invisible when output goes to FP4",
        targets_bottleneck=["compute_bound", "latency_bound"],
        priority=9,
        applicable_kernels=["silu_mul"],  # only kernel using expf (SiLU)
    ),
    "thread_coarsening": Strategy(
        name="thread_coarsening",
        display_name="Thread Coarsening",
        description="Increase work per thread (multiple quant blocks or rows per thread). "
                    "Amortizes thread launch/scheduling overhead and improves SM utilization "
                    "when per-thread work is too small",
        targets_bottleneck=["latency_bound", "compute_bound", "memory_bound"],
        priority=8,
        applicable_kernels=["add_rmsnorm", "silu_mul", "nvfp4_quantize"],
    ),
    "ldg_readonly": Strategy(
        name="ldg_readonly",
        display_name="Read-Only Cache (__ldg)",
        description="Route read-only inputs through L1 texture cache using __ldg() "
                    "intrinsic. Combined with __restrict__ and const, enables the "
                    "compiler to reorder loads and use a separate cache path",
        targets_bottleneck=["memory_bound", "latency_bound"],
        priority=7,
        applicable_kernels=["add_rmsnorm", "silu_mul", "nvfp4_quantize"],
    ),
    "vectorized_stores": Strategy(
        name="vectorized_stores",
        display_name="Vectorized Stores",
        description="Replace byte-by-byte packed FP4 output stores with uint2/uint4 "
                    "vectorized writes. Reduces store instruction count by 8x",
        targets_bottleneck=["memory_bound"],
        priority=8,
        applicable_kernels=["nvfp4_quantize", "silu_mul", "add_rmsnorm"],
    ),
    "cache_streaming_stores": Strategy(
        name="cache_streaming_stores",
        display_name="Cache-Streaming PTX Stores",
        description="Bypass the L2 cache allocation for pure write-outs (like quantization blocks) "
                    "using inline PTX cache-streaming modifiers. This prevents L2 cache thrashing "
                    "when outputs are never read back by the same kernel.",
        targets_bottleneck=["memory_bound"],
        priority=12,
        applicable_kernels=["nvfp4_quantize", "silu_mul", "add_rmsnorm"],
    ),
    # ── Hackathon-winning strategies (from B200 NVFP4 competition analysis) ──
    "launch_bounds_low_regs": Strategy(
        name="launch_bounds_low_regs",
        display_name="__launch_bounds__ Low Register Budget",
        description="Add __launch_bounds__(BLOCK_SIZE, MIN_BLOCKS_PER_SM) to constrain "
                    "the compiler to 32-48 registers per thread. On memory-bound kernels, "
                    "higher occupancy from fewer registers beats per-thread efficiency. "
                    "Hackathon winners used 32-45 regs vs 80+ for naive kernels",
        targets_bottleneck=["memory_bound", "latency_bound"],
        priority=11,
        applicable_kernels=["add_rmsnorm", "silu_mul", "nvfp4_quantize"],
    ),
    "hardware_fp4_intrinsics": Strategy(
        name="hardware_fp4_intrinsics",
        display_name="Hardware FP4/FP8 Intrinsics",
        description="Replace manual bit manipulation for E4M3/FP4 conversion with "
                    "hardware intrinsics: __nv_cvt_float_to_fp8 for E4M3 scales, "
                    "__nv_fp4x2_e2m1x2 for FP4 packing. Eliminates ~20 instructions "
                    "per quantization block. On B200 these are single-cycle operations",
        targets_bottleneck=["compute_bound", "latency_bound", "memory_bound"],
        priority=11,
        applicable_kernels=["add_rmsnorm", "silu_mul", "nvfp4_quantize"],
    ),
    "shape_specialized_unroll": Strategy(
        name="shape_specialized_unroll",
        display_name="Shape-Specialized Full Unroll",
        description="Hard-code the exact problem dimensions as compile-time constants. "
                    "Use constexpr/template parameters for hidden_size, rows, K. "
                    "Compiler fully unrolls loops, eliminates bounds checks and branches. "
                    "This is the #1 advantage over general-purpose libraries like FlashInfer",
        targets_bottleneck=["memory_bound", "latency_bound", "compute_bound"],
        priority=12,
        applicable_kernels=["add_rmsnorm", "silu_mul", "nvfp4_quantize"],
    ),
    "multi_row_processing": Strategy(
        name="multi_row_processing",
        display_name="Multi-Row (Thread Coarsening) Processing",
        description="Process multiple rows within a single thread block instead of just one row. "
                    "This amortizes shared data loads (like shared weight vectors or scales) "
                    "across rows because the L1 cache retains the fetched weights for all warps "
                    "handling the parallel rows, drastically cutting L1/L2 read bandwidth.",
        targets_bottleneck=["memory_bound"],
        priority=10,
        applicable_kernels=["add_rmsnorm", "silu_mul", "nvfp4_quantize"],
    ),
}

# ── Kernel-type-specific ideal strategies ────────────────────────────────────
# These are the best 4 strategies for each kernel type, ordered by impact.
# Used by select_for_kernel() when the system needs to pick strategies
# without relying on keyword grep.

KERNEL_IDEAL_STRATEGIES: dict = {
    "add_rmsnorm": [
        "fuse_passes",                # eliminate residual_out reread first
        "hardware_fp4_intrinsics",    # replace scalar FP4 branch chain
        "vectorize_loads",            # 128-bit coalesced bf16 loads
        "launch_bounds_low_regs",     # preserve 32-reg / 100% occ regime
        "shape_specialized_unroll",   # exploit fixed 128x2048 shape
        "multi_row_processing",       # only after the main structure is fixed
        "warp_reduction",             # secondary unless paired with bigger changes
    ],
    "silu_mul": [
        "shape_specialized_unroll",   # hard-code B, M, K dimensions
        "launch_bounds_low_regs",     # maximize occupancy for memory-bound
        "hardware_fp4_intrinsics",    # replace manual FP4 bit packing
        "vectorize_loads",            # 2 input arrays loaded scalar-by-scalar
        "fast_math_expf",             # SiLU uses slow expf
        "thread_coarsening",          # more work per thread
    ],
    "nvfp4_quantize": [
        "shape_specialized_unroll",   # hard-code M, K dimensions
        "launch_bounds_low_regs",     # maximize occupancy for memory-bound
        "hardware_fp4_intrinsics",    # kernel is ALL quantization — biggest win
        "vectorize_loads",            # + vectorized stores for both sides
        "thread_coarsening",          # more quant blocks per thread
        "vectorized_stores",          # fused packed output writes
    ],
}


def select_for_kernel(
    kernel_type: str,
    tried: list,
    beam_width: int = 4,
) -> list:
    """Select the best strategies for a specific kernel type.

    Uses the kernel-ideal mapping first, then falls back to scoring
    all applicable strategies from the bank.
    """
    # Start with ideal strategies for this kernel type
    ideal = KERNEL_IDEAL_STRATEGIES.get(kernel_type, [])
    selected = [s for s in ideal if s not in tried]

    # If we need more, score remaining applicable strategies
    if len(selected) < beam_width:
        remaining = []
        for name, s in STRATEGY_BANK.items():
            if name in selected or name in tried:
                continue
            if not s.applies_to_kernel(kernel_type):
                continue
            remaining.append((s.priority, name))
        remaining.sort(key=lambda x: -x[0])
        for _, name in remaining:
            if len(selected) >= beam_width:
                break
            selected.append(name)

    return selected[:beam_width]


def select_strategies(
    bottleneck: str,
    missing_opts: list,
    tried: list,
    beam_width: int = 4,
) -> list:
    """Legacy scorer: rank by bottleneck match + priority."""
    candidates = []
    for name, s in STRATEGY_BANK.items():
        if name in tried:
            continue
        if name not in missing_opts and missing_opts:
            continue
        score = s.priority
        if s.applies_to(bottleneck):
            score += 5
        candidates.append((score, name))
    candidates.sort(key=lambda x: -x[0])
    return [name for _, name in candidates[:beam_width]]


def get_strategy(name: str) -> Strategy:
    if name not in STRATEGY_BANK:
        raise ValueError(f"Unknown strategy: {name}")
    return STRATEGY_BANK[name]
