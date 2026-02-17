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


def fast_math_expf_prompt(kernel_slice: str, hw_spec: dict, current_metrics: dict = None) -> str:
    metrics_str = ""
    if current_metrics:
        metrics_str = (
            f"\nCurrent profiler metrics:\n"
            f"  Registers/thread: {current_metrics.get('_compiler', {}).get('registers_per_thread', 'N/A')}\n"
            f"  SM occupancy: {current_metrics.get('sm_occupancy', 'N/A')}%\n"
        )
    return f"""\
## Optimization Task: Fast Math Intrinsics for SiLU

Hardware: NVIDIA B200 (sm_100a, SFU unit available)
{metrics_str}
Full kernel source (complete .cu file):
```cuda
{kernel_slice}
```

Replace slow math library calls with hardware SFU intrinsics:

Background:
SiLU(x) = x / (1 + exp(-x)). The `expf()` call compiles to a multi-instruction
software routine (~20 cycles). The hardware SFU can compute `__expf()` in ~4 cycles
with slightly reduced precision. Since the output goes to FP4 (only 16 possible
values), the precision difference is completely invisible.

Requirements:
1. Replace `expf(-g)` with `__expf(-g)` in the SiLU computation
2. Replace `1.0f / (1.0f + ...)` with `__frcp_rn(1.0f + ...)` (hardware reciprocal)
3. The optimized SiLU becomes: `x * __frcp_rn(1.0f + __expf(-x))`
4. If rsqrtf is used anywhere, replace with `__frsqrt_rn()`
5. Do NOT change any other logic — only math intrinsic substitutions
6. Preserve all function signatures and launch wrappers

CRITICAL: Return the COMPLETE .cu file (all #includes, ALL kernel functions, and the launch_* wrapper function) in a single ```cuda code block. The file must compile standalone with nvcc. No explanations — just the code block.
"""


def thread_coarsening_prompt(kernel_slice: str, hw_spec: dict, current_metrics: dict = None) -> str:
    sm_count = hw_spec["sm"]["count"]
    metrics_str = ""
    if current_metrics:
        metrics_str = (
            f"\nCurrent profiler metrics:\n"
            f"  SM occupancy: {current_metrics.get('sm_occupancy', 'N/A')}%\n"
            f"  Registers/thread: {current_metrics.get('_compiler', {}).get('registers_per_thread', 'N/A')}\n"
            f"  Kernel timing: {current_metrics.get('duration_us', 'N/A')} us\n"
        )
    return f"""\
## Optimization Task: Thread Coarsening

Hardware: NVIDIA B200 ({sm_count} SMs, 2048 threads/SM max)
{metrics_str}
Full kernel source (complete .cu file):
```cuda
{kernel_slice}
```

Increase the amount of work each thread performs:

Background:
Currently each thread processes very little data (e.g., 1 quantization block = 16
elements, or a small slice of a row). Thread scheduling, register allocation, and
warp launch overhead dominate. By having each thread process 4-8x more data, we
amortize this overhead and improve instruction-level parallelism.

Requirements:
1. Identify the per-thread work unit (quantization block, row elements, etc.)
2. Increase it by 4x:
   - For quantize kernels: each thread processes 4 quant blocks (64 elements)
   - For add_rmsnorm: each block processes 2-4 rows instead of 1
   - Use a grid-stride loop if the total work exceeds grid capacity
3. Adjust grid dimensions: divide grid size by the coarsening factor
4. Use `#pragma unroll` on the coarsened inner loop
5. Keep register usage under 96 per thread to maintain occupancy
6. Handle the tail case: when total work isn't divisible by coarsening factor,
   add a bounds check for the last iterations

CRITICAL: Return the COMPLETE .cu file (all #includes, ALL kernel functions, and the launch_* wrapper function) in a single ```cuda code block. The file must compile standalone with nvcc. No explanations — just the code block.
"""


def ldg_readonly_prompt(kernel_slice: str, hw_spec: dict, current_metrics: dict = None) -> str:
    metrics_str = ""
    if current_metrics:
        metrics_str = (
            f"\nCurrent profiler metrics:\n"
            f"  Kernel timing: {current_metrics.get('duration_us', 'N/A')} us\n"
            f"  L2 hit rate: {current_metrics.get('l2_hit_rate', 'N/A')}%\n"
            f"  DRAM stall rate: {current_metrics.get('stall_memory', 'N/A')}%\n"
        )
    return f"""\
## Optimization Task: Read-Only Cache Routing (__ldg)

Hardware: NVIDIA B200 (sm_100a, separate L1 texture cache path)
{metrics_str}
Full kernel source (complete .cu file):
```cuda
{kernel_slice}
```

Route read-only memory accesses through the L1 texture cache:

Background:
The GPU has a separate read-only data cache (texture/L1 cache) that doesn't
participate in coherence traffic. For data that is only read (never written),
using __ldg() routes loads through this cache, freeing up the normal L1 for
read-write data. Combined with __restrict__, the compiler can also reorder
loads more aggressively.

Requirements:
1. Identify all input pointers that are read-only (never written to):
   - input, residual, weight/rms_weight, gate, up — these are all read-only
2. Mark them with `const __restrict__` if not already done
3. Replace direct reads like `input[idx]` with `__ldg(&input[idx])`
4. For bf16: `__ldg()` works with `__nv_bfloat16` directly
5. Do NOT apply __ldg to output pointers (quant_out, residual_out, scales)
6. Do NOT change any computation logic — only memory access routing

CRITICAL: Return the COMPLETE .cu file (all #includes, ALL kernel functions, and the launch_* wrapper function) in a single ```cuda code block. The file must compile standalone with nvcc. No explanations — just the code block.
"""


def vectorized_stores_prompt(kernel_slice: str, hw_spec: dict, current_metrics: dict = None) -> str:
    metrics_str = ""
    if current_metrics:
        metrics_str = (
            f"\nCurrent profiler metrics:\n"
            f"  Kernel timing: {current_metrics.get('duration_us', 'N/A')} us\n"
            f"  Registers/thread: {current_metrics.get('_compiler', {}).get('registers_per_thread', 'N/A')}\n"
        )
    return f"""\
## Optimization Task: Vectorized Stores for Packed Output

Hardware: NVIDIA B200 (128-bit store transactions)
{metrics_str}
Full kernel source (complete .cu file):
```cuda
{kernel_slice}
```

Replace byte-by-byte stores of packed FP4 output with vectorized writes:

Background:
The packed FP4 output is stored byte-by-byte in a loop:
  for (j = 0; j < 8; ++j) quant_out[base + j] = packed_out[j];
Each 1-byte store is a separate memory transaction. A single uint2 store (8 bytes)
replaces all 8 stores with one 64-bit write. Similarly, scale stores can be batched.

Requirements:
1. Accumulate the 8 packed bytes into a uint2 (or two uint32_t values)
2. Store with a single `*reinterpret_cast<uint2*>(&quant_out[base]) = packed_val;`
3. Ensure the output pointer is 8-byte aligned (it should be, since quant blocks
   are 8 bytes each and base addresses are cudaMalloc'd)
4. Also vectorize loads where possible (input bf16 → load as uint4 for 8 bf16 values)
5. Preserve all output semantics — packed byte order must be identical
6. Keep the scale store as-is (single byte, not worth vectorizing alone)

CRITICAL: Return the COMPLETE .cu file (all #includes, ALL kernel functions, and the launch_* wrapper function) in a single ```cuda code block. The file must compile standalone with nvcc. No explanations — just the code block.
"""


STRATEGY_PROMPTS = {
    "vectorize_loads":   vectorize_loads_prompt,
    "tma_prefetch":      tma_prefetch_prompt,
    "warp_reduction":    warp_reduction_prompt,
    "fuse_passes":       fuse_passes_prompt,
    "register_tiling":   register_tiling_prompt,
    "async_pipeline":    async_pipeline_prompt,
    "fp4_lut":           fp4_lut_prompt,
    "fast_math_expf":    fast_math_expf_prompt,
    "thread_coarsening": thread_coarsening_prompt,
    "ldg_readonly":      ldg_readonly_prompt,
    "vectorized_stores": vectorized_stores_prompt,
}


def get_prompt_for_strategy(
    strategy: str,
    kernel_slice: str,
    hw_spec: dict,
    current_metrics: dict = None,
) -> str:
    if strategy not in STRATEGY_PROMPTS:
        raise ValueError(f"Unknown strategy: {strategy}. Available: {list(STRATEGY_PROMPTS)}")
    return STRATEGY_PROMPTS[strategy](kernel_slice, hw_spec, current_metrics)
