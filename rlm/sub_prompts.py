"""
sub_prompts.py — Strategy-specific prompts for sub-LLMs.
Sub-LLMs receive the full kernel source and must return a complete compilable .cu file.
"""

from __future__ import annotations


def vectorize_loads_prompt(kernel_slice: str, hw_spec: dict, current_metrics: dict = None) -> str:
    mem_bw = hw_spec["memory"]["hbm_bandwidth_tbs"]
    metrics_str = ""
    if current_metrics:
        metrics_str = (
            f"\nCurrent profiler metrics:\n"
            f"  Kernel timing: {current_metrics.get('duration_us', 'N/A')} us\n"
            f"  DRAM stall rate: {current_metrics.get('stall_memory', 'N/A')}%\n"
            f"  L2 hit rate: {current_metrics.get('l2_hit_rate', 'N/A')}%\n"
        )
    return f"""\
## Optimization Task: Vectorize Global Memory Loads

Hardware: NVIDIA B200 ({mem_bw} TB/s HBM3e bandwidth)
{metrics_str}
Full kernel source (complete .cu file):
```cuda
{kernel_slice}
```

Apply float4 / uint4 vectorized loads (128-bit transactions):

Requirements:
1. Replace scalar `float` loads with `float4` where 16-byte aligned
2. For bfloat16: use `uint4` loads then reinterpret as `__nv_bfloat162 x4`
3. Ensure pointer alignment with `__builtin_assume_aligned(ptr, 16)`
4. Same output semantics — no change to computed values
5. Do NOT add new __syncthreads() calls
6. Add `#pragma unroll 4` hints where appropriate

CRITICAL: Return the COMPLETE .cu file (all #includes, ALL kernel functions, and the launch_* wrapper function) in a single ```cuda code block. The file must compile standalone with nvcc. No explanations — just the code block.
"""


def tma_prefetch_prompt(kernel_slice: str, hw_spec: dict, current_metrics: dict = None) -> str:
    metrics_str = ""
    if current_metrics:
        metrics_str = (
            f"\nCurrent profiler metrics:\n"
            f"  Kernel timing: {current_metrics.get('duration_us', 'N/A')} us\n"
            f"  Long scoreboard stalls: {current_metrics.get('stall_memory', 'N/A')}%\n"
        )
    return f"""\
## Optimization Task: TMA Async Prefetch

Hardware: NVIDIA B200 (Blackwell) with TMA support
{metrics_str}
Full kernel source (complete .cu file):
```cuda
{kernel_slice}
```

Add TMA async prefetch with double-buffering:

Requirements:
1. Use `tma_load_1d()` from b200_intrinsics.cuh for async copy
2. Double-buffered shared memory: two ping-pong buffers
3. Initialize mbarrier with `mbar_init()`, wait with `mbar_wait()`
4. Issue prefetch 1 tile ahead
5. Elect threadIdx.x == 0 to issue TMA
6. __syncthreads() or mbar_wait() before consuming prefetched data

Shared memory layout:
  __shared__ T smem_buf[2][TILE_SIZE];  // double buffer
  __shared__ uint64_t mbar[2];

CRITICAL: Return the COMPLETE .cu file (all #includes, ALL kernel functions, and the launch_* wrapper function) in a single ```cuda code block. The file must compile standalone with nvcc. No explanations — just the code block.
"""


def warp_reduction_prompt(kernel_slice: str, hw_spec: dict, current_metrics: dict = None) -> str:
    metrics_str = ""
    if current_metrics:
        metrics_str = (
            f"\nCurrent profiler metrics:\n"
            f"  Barrier stall rate: {current_metrics.get('stall_barrier', 'N/A')}%\n"
            f"  SM occupancy: {current_metrics.get('sm_occupancy', 'N/A')}%\n"
        )
    return f"""\
## Optimization Task: Replace Block Reduction with Warp Shuffles

Hardware: NVIDIA B200 (sm_100a, __shfl_xor_sync supported)
{metrics_str}
Full kernel source (complete .cu file, contains __syncthreads-based reduction):
```cuda
{kernel_slice}
```

Replace shared-memory block reduction with warp-level shuffles:

Requirements:
1. Use `warp_reduce_sum()` from nvfp4_utils.cuh (__shfl_xor_sync internally)
2. Pattern: each warp reduces locally → warp leaders write to shared mem
   → warp 0 reduces the warp partial sums
3. Only ONE __syncthreads() instead of O(log N) naive syncs
4. For RMSNorm: accumulate `x*x` locally, warp reduce, then combine warps
5. Use `0xFFFFFFFF` as the warp mask (full warp participation)
6. Preserve numerical equivalence

CRITICAL: Return the COMPLETE .cu file (all #includes, ALL kernel functions, and the launch_* wrapper function) in a single ```cuda code block. The file must compile standalone with nvcc. No explanations — just the code block.
"""


