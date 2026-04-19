"""
test_llm_codegen.py — Test if LLMs can write compilable, correct CUDA kernels.

Measures: compile pass@1, correctness pass@1 (optional, requires GPU), and cost.

Usage:
    python3 test_llm_codegen.py --model sonnet              # compile-only
    python3 test_llm_codegen.py --model sonnet --check       # compile + correctness (needs GPU)
    python3 test_llm_codegen.py --model all
    python3 test_llm_codegen.py --model opus --kernel silu_mul
"""

import argparse
import json
import os
import re
import subprocess
import tempfile
import time
from pathlib import Path

import anthropic

PROJECT_ROOT = Path(__file__).parent

MODELS = {
    "haiku":  "claude-haiku-4-5-20251001",
    "sonnet": "claude-sonnet-4-6",
    "opus":   "claude-opus-4-6",
}

PRICING = {
    "haiku":  {"in": 0.25, "out": 1.25},
    "sonnet": {"in": 3.0,  "out": 15.0},
    "opus":   {"in": 15.0, "out": 75.0},
}

# ── Kernel sources ────────────────────────────────────────────────────────────

KERNELS = {
    "add_rmsnorm":    PROJECT_ROOT / "kernels" / "reference" / "add_rmsnorm.cu",
    "silu_mul":       PROJECT_ROOT / "kernels" / "reference" / "silu_mul.cu",
    "nvfp4_quantize": PROJECT_ROOT / "kernels" / "reference" / "nvfp4_quantize.cu",
}

COMMON_DIR = PROJECT_ROOT / "kernels" / "common"

# ── Optimizations to test (from Sonnet freeform results — the best picks) ────

TASKS = {
    "add_rmsnorm": [
        {
            "name": "vectorized_loads",
            "instruction": (
                "Replace all scalar bfloat16 loads with 128-bit vectorized loads. "
                "Use uint4 loads and reinterpret as __nv_bfloat162 pairs to load "
                "8 bf16 elements per transaction. Apply to input, residual, weight, "
                "and residual_out arrays in both Phase 1 and Phase 2 loops. "
                "Adjust loop stride accordingly (tid*8 instead of tid)."
            ),
        },
        {
            "name": "fuse_passes",
            "instruction": (
                "Eliminate the global memory round-trip between Phase 1 and Phase 2. "
                "Currently Phase 1 writes residual_out[base+i] to HBM, then Phase 2 "
                "re-reads it. Instead, keep the added values (a+r) in registers or "
                "shared memory during Phase 1, compute the RMS inverse, then immediately "
                "normalize and quantize without re-reading from global memory."
            ),
        },
        {
            "name": "vectorized_stores",
            "instruction": (
                "Replace the byte-by-byte quant_out writes with vectorized stores. "
                "The loop `for (j=0; j<8; ++j) quant_out[base+j] = packed_out[j]` "
                "does 8 separate byte stores. Pack the 8 bytes into a uint2 and write "
                "with a single 64-bit store: `*(uint2*)&quant_out[base] = packed_val`. "
                "Ensure 8-byte alignment."
            ),
        },
    ],
    "silu_mul": [
        {
            "name": "vectorized_loads",
            "instruction": (
                "Replace scalar bfloat16 loads of gate and up arrays with 128-bit "
                "vectorized loads. Use uint4 to load 8 bf16 values at once from each "
                "array. Reinterpret as __nv_bfloat162 for conversion to float. "
                "Each thread should still process one 16-element NVFP4 block but "
                "load in two 8-element vector transactions."
            ),
        },
        {
            "name": "fast_math_expf",
            "instruction": (
                "Replace the slow expf() call in silu_f32() with __expf() hardware "
                "intrinsic. Also replace the reciprocal with __frcp_rn(). The optimized "
                "SiLU becomes: x * __frcp_rn(1.0f + __expf(-x)). This uses the SFU "
                "unit (~4 cycles vs ~20 for software expf). Precision loss is invisible "
                "since output goes to FP4."
            ),
        },
        {
            "name": "thread_coarsening",
            "instruction": (
                "Have each thread process 4 NVFP4 blocks (64 elements) instead of 1. "
                "Use a loop over 4 blocks per thread. Divide the grid size by 4. "
                "Keep all values in registers. Add bounds checking for the last "
                "iteration when num_quant_blocks is not divisible by 4."
            ),
        },
    ],
    "nvfp4_quantize": [
        {
            "name": "vectorized_loads",
            "instruction": (
                "Replace the scalar bfloat16 loads with 128-bit vectorized loads. "
                "Use uint4 to load 8 bf16 values at once (16 bytes). Each NVFP4 block "
                "is 16 elements, so 2 vector loads per block. Reinterpret the uint4 "
                "components as __nv_bfloat162 pairs for conversion to float."
            ),
        },
        {
            "name": "vectorized_stores",
            "instruction": (
                "Replace byte-by-byte packed output stores with a single 64-bit store. "
                "The 8 packed bytes per NVFP4 block should be accumulated into a uint2 "
                "and written with: *(uint2*)&packed[packed_base] = packed_val. This "
                "replaces 8 byte stores with 1 instruction."
            ),
        },
        {
            "name": "thread_coarsening",
            "instruction": (
                "Have each thread process 4 quantization blocks instead of 1. "
                "Loop over 4 blocks sequentially, keeping all values in registers. "
                "Reduce grid size by 4x. Add bounds check: if (block_id + i*stride >= "
                "num_blocks) break. Use #pragma unroll on the 4-iteration outer loop."
            ),
        },
    ],
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def expand_includes(src: str, src_path: Path) -> str:
    """Expand local #include directives so the LLM sees helper functions."""
    lines = src.split("\n")
    result = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith('#include "') and stripped.endswith('"'):
            rel_path = stripped[len('#include "'):-1]
            header = src_path.parent / rel_path
            if header.exists():
                result.append(f"// === expanded from {rel_path} ===")
                result.append(header.read_text())
                result.append(f"// === end {rel_path} ===")
                continue
        result.append(line)
    return "\n".join(result)


def extract_cuda_code(text: str) -> str:
    """Extract CUDA code from LLM response."""
    for pattern in [r"```cuda\s*\n(.*?)```", r"```cpp\s*\n(.*?)```",
                    r"```c\s*\n(.*?)```", r"```\s*\n(.*?)```"]:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
    if "__global__" in text or "#include" in text:
        lines = text.split("\n")
        for i, line in enumerate(lines):
            s = line.strip()
            if s.startswith("#include") or s.startswith("__global__") or s.startswith("//"):
                return "\n".join(lines[i:]).strip()
    return ""


def detect_cuda_arch() -> str:
    """Detect best CUDA arch that nvcc can compile (for compile-only tests)."""
    nvcc = _find_nvcc()
    test_code = '''\
#include <cuda_runtime.h>
#include <cuda_fp8.h>
__device__ float test() {
    __nv_fp8_storage_t x = 0;
    return __nv_cvt_fp8_to_float(x, __NV_E4M3);
}
'''
    with tempfile.TemporaryDirectory() as tmpdir:
        src = os.path.join(tmpdir, "test.cu")
        obj = os.path.join(tmpdir, "test.o")
        with open(src, "w") as f:
            f.write(test_code)
        for arch in ["sm_100a", "sm_90a", "sm_89", "sm_80"]:
            try:
                result = subprocess.run(
                    [nvcc, "-c", f"-arch={arch}", "-o", obj, src],
                    capture_output=True, text=True, timeout=30,
                )
                if result.returncode == 0:
                    return arch
            except (FileNotFoundError, subprocess.TimeoutExpired):
                pass
    return "sm_80"


def detect_gpu_arch() -> str:
    """Detect actual GPU compute capability (for running kernels)."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            cap = result.stdout.strip().split("\n")[0].strip()
            major, minor = cap.split(".")
            return f"sm_{major}{minor}"
    except Exception:
        pass
    return None


def has_gpu() -> bool:
    """Check if an NVIDIA GPU is available."""
    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True, timeout=5)
        return result.returncode == 0
    except Exception:
        return False


