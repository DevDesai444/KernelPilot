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


def _find_nvcc() -> str:
    cuda_home = os.environ.get("CUDA_HOME", "/usr/local/cuda")
    nvcc = os.path.join(cuda_home, "bin", "nvcc")
    return nvcc if os.path.exists(nvcc) else "nvcc"


# ── Stub headers (for systems without CUDA 12.8+ / Blackwell) ────────────────

STUB_NVFP4 = '''\
#pragma once
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdint.h>

// === Stub: FP8 type and conversion (requires CUDA 12.8+) ===
typedef unsigned char __nv_fp8_storage_t;
__device__ __forceinline__ __nv_fp8_storage_t float_to_e4m3(float x) {
    // Minimal E4M3 stub: clamp to [0,448], encode sign+exponent+mantissa
    unsigned int bits;
    memcpy(&bits, &x, 4);
    unsigned int sign = (bits >> 24) & 0x80;
    float ax = fabsf(x);
    if (ax < 1.95313e-03f) return sign;                   // zero / tiny
    if (ax > 448.0f) return sign | 0x7E;                  // max normal
    int exp = 0;
    float m = frexpf(ax, &exp);                            // 0.5 <= m < 1.0
    exp += 6;                                              // bias=7, frexp gives 2^(exp-1)
    if (exp < 0) exp = 0;
    if (exp > 15) exp = 15;
    int mant = (int)((m * 2.0f - 1.0f) * 4.0f + 0.5f);   // 2-bit mantissa
    if (mant > 3) mant = 3;
    return (uint8_t)(sign | (exp << 2) | mant);
}
__device__ __forceinline__ float e4m3_to_float(__nv_fp8_storage_t x) {
    unsigned int sign = (x & 0x80) ? 1 : 0;
    unsigned int exp  = (x >> 2) & 0xF;
    unsigned int mant = x & 0x3;
    float val;
    if (exp == 0) {
        val = (mant / 4.0f) * powf(2.0f, -6.0f);         // subnormal
    } else {
        val = (1.0f + mant / 4.0f) * powf(2.0f, (float)exp - 7.0f);
    }
    return sign ? -val : val;
}

// === Real definitions (copied from nvfp4_utils.cuh) ===
#define NVFP4_MANTISSA_BITS 2
#define NVFP4_EXPONENT_BITS 1
#define NVFP4_BIAS          1
#define NVFP4_BLOCK_SIZE    16
#define NVFP4_PER_BYTE      2
#define NVFP4_PER_INT32     8

__device__ __constant__ float kNVFP4LUT[16] = {
    0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
   -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
};

__device__ __forceinline__ uint8_t float_to_nvfp4(float x) {
    uint8_t sign_bit = (x < 0.0f) ? 0x8u : 0x0u;
    float ax = fabsf(x);
    ax = fminf(ax, 6.0f);
    uint8_t code;
    if      (ax < 0.25f) code = 0;
    else if (ax < 0.75f) code = 1;
    else if (ax < 1.25f) code = 2;
    else if (ax < 1.75f) code = 3;
    else if (ax < 2.5f)  code = 4;
    else if (ax < 3.5f)  code = 5;
    else if (ax < 5.0f)  code = 6;
    else                 code = 7;
    return sign_bit | code;
}

__device__ __forceinline__ float nvfp4_to_float(uint8_t code) {
    return kNVFP4LUT[code & 0xF];
}

__device__ __forceinline__ void quantize_block_nvfp4(
    const float* __restrict__ x,
    uint8_t* __restrict__ packed,
    __nv_fp8_storage_t* __restrict__ scale)
{
    float amax = 0.0f;
    #pragma unroll
    for (int i = 0; i < NVFP4_BLOCK_SIZE; ++i)
        amax = fmaxf(amax, fabsf(x[i]));

    const float inv_max_repr = 1.0f / 6.0f;
    float s = (amax > 0.0f) ? (amax * inv_max_repr) : 1.0f;
    *scale = float_to_e4m3(s);
    float inv_s = (amax > 0.0f) ? (6.0f / amax) : 1.0f;

    #pragma unroll
    for (int i = 0; i < NVFP4_BLOCK_SIZE / 2; ++i) {
        uint8_t lo = float_to_nvfp4(x[2*i]   * inv_s);
        uint8_t hi = float_to_nvfp4(x[2*i+1] * inv_s);
        packed[i] = (hi << 4) | (lo & 0xF);
    }
}

__device__ __forceinline__ void dequantize_block_nvfp4(
    const uint8_t* __restrict__ packed,
    __nv_fp8_storage_t scale,
    float* __restrict__ out)
{
    float s = e4m3_to_float(scale);
    #pragma unroll
    for (int i = 0; i < NVFP4_BLOCK_SIZE / 2; ++i) {
        uint8_t byte = packed[i];
        out[2*i]   = nvfp4_to_float(byte & 0xF) * s;
        out[2*i+1] = nvfp4_to_float(byte >> 4)  * s;
    }
}

__device__ __forceinline__ uint32_t pack8_nvfp4(const uint8_t codes[8]) {
    uint32_t result = 0;
    #pragma unroll
    for (int i = 0; i < 8; ++i)
        result |= ((uint32_t)(codes[i] & 0xF)) << (i * 4);
    return result;
}

__device__ __forceinline__ void unpack8_nvfp4(uint32_t packed, uint8_t codes[8]) {
    #pragma unroll
    for (int i = 0; i < 8; ++i)
        codes[i] = (packed >> (i * 4)) & 0xF;
}

__device__ __forceinline__ float warp_absmax(float val) {
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1)
        val = fmaxf(val, fabsf(__shfl_xor_sync(0xFFFFFFFF, val, mask)));
    return val;
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1)
        val += __shfl_xor_sync(0xFFFFFFFF, val, mask);
    return val;
}
'''

STUB_B200 = '''\
#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>

// === TMA stubs (require sm_100a PTX) ===
struct TMADescriptor1D {
    uint64_t tensor_map;
    uint32_t box_size;
    uint32_t element_stride;
};

__device__ __forceinline__ void tma_load_1d(
    void* smem_dst, const void* gmem_src, uint64_t* mbar, uint32_t num_bytes) {}
__device__ __forceinline__ void mbar_init(uint64_t* mbar, uint32_t num_transactions) {}
__device__ __forceinline__ void mbar_wait(uint64_t* mbar, uint32_t phase) {}
__device__ __forceinline__ void mbar_arrive(uint64_t* mbar) {}

struct PipelineState {
    uint32_t phase = 0;
    uint32_t stage = 0;
    __device__ __forceinline__ void issue_prefetch(
        void* smem_buf, const void* gmem_src, uint32_t num_bytes, uint64_t* mbar) {}
    __device__ __forceinline__ void wait(uint64_t* mbar) {}
    __device__ __forceinline__ void swap() { stage ^= 1; phase ^= 1; }
};

// === Vectorized load/store (these work on sm_80+) ===
__device__ __forceinline__ float4 load_float4(const float* ptr) {
    return *reinterpret_cast<const float4*>(ptr);
}
__device__ __forceinline__ void store_float4(float* ptr, float4 val) {
    *reinterpret_cast<float4*>(ptr) = val;
}
__device__ __forceinline__ uint4 load_uint4(const uint32_t* ptr) {
    return *reinterpret_cast<const uint4*>(ptr);
}
__device__ __forceinline__ void load_bf16x8(
    const __nv_bfloat16* ptr,
    __nv_bfloat162& a, __nv_bfloat162& b,
    __nv_bfloat162& c, __nv_bfloat162& d)
{
    uint4 raw = *reinterpret_cast<const uint4*>(ptr);
    a = *reinterpret_cast<__nv_bfloat162*>(&raw.x);
    b = *reinterpret_cast<__nv_bfloat162*>(&raw.y);
    c = *reinterpret_cast<__nv_bfloat162*>(&raw.z);
    d = *reinterpret_cast<__nv_bfloat162*>(&raw.w);
}

// === TMEM stubs (require sm_100a PTX) ===
__device__ __forceinline__ uint32_t tmem_alloc(uint32_t num_bytes) { return 0; }
__device__ __forceinline__ void tmem_free(uint32_t tmem_addr) {}
__device__ __forceinline__ uint32_t tmem_load(uint32_t tmem_addr) { return 0; }
__device__ __forceinline__ void tmem_store(uint32_t tmem_addr, uint32_t val) {}

__device__ __forceinline__ bool is_warp_leader() { return (threadIdx.x % 32) == 0; }
__device__ __forceinline__ bool is_block_leader() {
    return threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0;
}

__device__ __forceinline__ float load_streaming(const float* ptr) {
    return *ptr;
}
__device__ __forceinline__ void prefetch_l2(const void* ptr) {}
'''


