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


