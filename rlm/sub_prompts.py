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


def fuse_passes_prompt(kernel_slice: str, hw_spec: dict, current_metrics: dict = None) -> str:
    metrics_str = ""
    if current_metrics:
        metrics_str = (
            f"\nCurrent profiler metrics:\n"
            f"  Kernel timing: {current_metrics.get('duration_us', 'N/A')} us\n"
            f"  Achieved occupancy: {current_metrics.get('sm_occupancy', 'N/A')}%\n"
        )
    return f"""\
## Optimization Task: Fuse Multiple Memory Passes into One

Hardware: NVIDIA B200 ({hw_spec['memory']['hbm_bandwidth_tbs']} TB/s HBM3e)
{metrics_str}
Full kernel source (complete .cu file, currently makes multiple passes over data):
```cuda
{kernel_slice}
```

Fuse separate read passes into a single loop:

Requirements:
1. Identify values computed in Pass 1 needed in Pass 2
2. Use registers or shared memory to carry values across phases
3. Single loop: load → compute all phases → store results
4. Register budget: B200 has 255 registers/thread — use ~64
5. Use `__ldg()` for read-only data (L1 texture cache)
6. Mark read-only pointers with `__restrict__` and `const`

CRITICAL: Return the COMPLETE .cu file (all #includes, ALL kernel functions, and the launch_* wrapper function) in a single ```cuda code block. The file must compile standalone with nvcc. No explanations — just the code block.
"""


def register_tiling_prompt(kernel_slice: str, hw_spec: dict, current_metrics: dict = None) -> str:
    metrics_str = ""
    if current_metrics:
        metrics_str = (
            f"\nCurrent profiler metrics:\n"
            f"  Registers/thread: {current_metrics.get('_compiler', {}).get('registers_per_thread', 'N/A')}\n"
            f"  Achieved occupancy: {current_metrics.get('sm_occupancy', 'N/A')}%\n"
        )
    return f"""\
## Optimization Task: Register-Level Tiling for Compute ILP

Hardware: NVIDIA B200 (255 registers/thread, 4-wide SIMD fp32)
{metrics_str}
Full kernel source (complete .cu file):
```cuda
{kernel_slice}
```

Apply register tiling to increase ILP:

Requirements:
1. Unroll inner loop by 4x — process 4 independent elements per iteration
2. Use separate register variables (avoid array subscripts in hot path)
3. Interleave independent computations to hide latency
4. For RMSNorm: compute 4 partial sums simultaneously before reducing
5. Preserve #pragma unroll for the compiler
6. Do not exceed 96 registers/thread

CRITICAL: Return the COMPLETE .cu file (all #includes, ALL kernel functions, and the launch_* wrapper function) in a single ```cuda code block. The file must compile standalone with nvcc. No explanations — just the code block.
"""


def async_pipeline_prompt(kernel_slice: str, hw_spec: dict, current_metrics: dict = None) -> str:
    metrics_str = ""
    if current_metrics:
        metrics_str = (
            f"\nCurrent profiler metrics:\n"
            f"  Long scoreboard stalls: {current_metrics.get('stall_memory', 'N/A')}%\n"
            f"  Kernel timing: {current_metrics.get('duration_us', 'N/A')} us\n"
        )
    return f"""\
## Optimization Task: Async Software Pipeline (cp.async)

Hardware: NVIDIA B200 (Blackwell, cp.async.bulk supported)
{metrics_str}
Full kernel source (complete .cu file):
```cuda
{kernel_slice}
```

Implement a software pipeline using cp.async:

Requirements:
1. Use `cuda::pipeline` from <cuda/pipeline>
2. Pattern: issue cp.async for next tile → compute on current tile → commit + wait
3. Pipeline depth = 2 (double buffer)
4. Shared memory: allocate 2x tile_size for ping-pong
5. Use `__pipeline_memcpy_async()` or TMA for bulk async copy
6. Commit with `__pipeline_commit()`
7. Wait with `__pipeline_wait_prior(1)` — allow 1 outstanding stage
8. Handle prologue (first tile) and epilogue (drain) correctly

CRITICAL: Return the COMPLETE .cu file (all #includes, ALL kernel functions, and the launch_* wrapper function) in a single ```cuda code block. The file must compile standalone with nvcc. No explanations — just the code block.
"""


def fp4_lut_prompt(kernel_slice: str, hw_spec: dict, current_metrics: dict = None) -> str:
    metrics_str = ""
    if current_metrics:
        metrics_str = (
            f"\nCurrent profiler metrics:\n"
            f"  Registers/thread: {current_metrics.get('_compiler', {}).get('registers_per_thread', 'N/A')}\n"
            f"  Kernel timing: {current_metrics.get('duration_us', 'N/A')} us\n"
        )
    return f"""\
## Optimization Task: FP4 Quantization Lookup Table

Hardware: NVIDIA B200 (sm_100a)
{metrics_str}
Full kernel source (complete .cu file):
```cuda
{kernel_slice}
```

Replace the arithmetic FP4 quantization with a precomputed lookup table:

Background:
NVFP4 has only 16 possible output values (4 bits). The current quantize_block_nvfp4()
does per-element: find absmax → compute scale → clamp → round → bit-pack. This is
expensive arithmetic for a function with only 16 possible outputs.

Requirements:
1. Precompute a __constant__ or __device__ lookup table that maps scaled float values
   to their nearest FP4 encoding (4-bit code)
2. The quantization per block becomes:
   a. Find absmax of the 16-element block (keep this)
   b. Compute E4M3 scale from absmax (keep this)
   c. For each element: multiply by (1/scale), clamp to FP4 range,
      use a LUT indexed by the quantized bin to get the 4-bit code
3. Pack two 4-bit codes per byte as before
4. The LUT approach eliminates the per-element float-to-fp4 conversion arithmetic
5. Ensure the LUT covers both positive and negative values (sign bit handled separately)
6. Keep the E4M3 scale computation identical to the original

Hint: FP4 positive values are [0, 0.5, 1, 1.5, 2, 3, 4, 6]. You can build a
boundary table [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0] and use binary search or
linear scan to find the nearest bin. With only 7 boundaries, a linear scan in
registers is faster than any branching approach.

CRITICAL: Return the COMPLETE .cu file (all #includes, ALL kernel functions, and the launch_* wrapper function) in a single ```cuda code block. The file must compile standalone with nvcc. No explanations — just the code block.
"""


