"""
correctness.py — Numerical correctness verification.

Primary: validate against FlashInfer reference (production code path on B200).
Fallback: validate against hand-written CUDA reference kernel.
"""

from __future__ import annotations
import logging
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

from eval.hack_detector import is_clean
from eval import flashinfer_ref

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).parent.parent


def _cuda_harness_prelude() -> str:
    return r"""
#define CHECK_CUDA(call) do { \
    cudaError_t err__ = (call); \
    if (err__ != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err__)); \
        return 2; \
    } \
} while (0)
#define cudaMalloc(...) CHECK_CUDA(cudaMalloc(__VA_ARGS__))
#define cudaMemcpy(...) CHECK_CUDA(cudaMemcpy(__VA_ARGS__))
#define cudaMemset(...) CHECK_CUDA(cudaMemset(__VA_ARGS__))
#define cudaFree(...) CHECK_CUDA(cudaFree(__VA_ARGS__))
#define cudaStreamCreate(...) CHECK_CUDA(cudaStreamCreate(__VA_ARGS__))
#define cudaStreamSynchronize(...) CHECK_CUDA(cudaStreamSynchronize(__VA_ARGS__))
#define cudaEventCreate(...) CHECK_CUDA(cudaEventCreate(__VA_ARGS__))
#define cudaEventRecord(...) CHECK_CUDA(cudaEventRecord(__VA_ARGS__))
#define cudaEventElapsedTime(...) CHECK_CUDA(cudaEventElapsedTime(__VA_ARGS__))
#define cudaEventDestroy(...) CHECK_CUDA(cudaEventDestroy(__VA_ARGS__))
"""


def _generate_flashinfer_harness(kernel_type: str, shape: tuple,
                                  ref_data_dir: str, atol: float, rtol: float) -> str:
    """Generate CUDA harness that loads FlashInfer reference outputs and compares."""
    if kernel_type == "add_rmsnorm":
        return _flashinfer_harness_add_rmsnorm(shape, ref_data_dir, atol, rtol)
    elif kernel_type == "nvfp4_quantize":
        return _flashinfer_harness_nvfp4_quantize(shape, ref_data_dir, atol, rtol)
    elif kernel_type == "silu_mul":
        return _flashinfer_harness_silu_mul(shape, ref_data_dir, atol, rtol)
    return None


def _flashinfer_harness_add_rmsnorm(shape: tuple, ref_dir: str,
                                     atol: float, rtol: float) -> str:
    rows, hidden = shape
    n = rows * hidden
    nb = n // 16
    qblocks_per_row = hidden // 16
    return f"""
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
{_cuda_harness_prelude()}

void launch_fused_add_rmsnorm_nvfp4(
    const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*,
    __nv_bfloat16*, unsigned char*, __nv_fp8_storage_t*, int, int, cudaStream_t);

/* FP4 decode LUT: codes 0-7 positive, 8-15 negative */
static const float kFP4LUT[16] = {{
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
   -0.0f,-0.5f,-1.0f,-1.5f,-2.0f,-3.0f,-4.0f,-6.0f
}};

static float decode_e4m3(unsigned char x) {{
    unsigned int sign = (x & 0x80u);
    unsigned int exp  = (x >> 3) & 0xFu;
    unsigned int mant = x & 0x7u;
    float val;
    if (exp == 0u) val = (mant / 8.0f) * (1.0f / 64.0f);
    else           val = (1.0f + mant / 8.0f) * powf(2.0f, (float)exp - 7.0f);
    return sign ? -val : val;
}}

static void load_bf16(const char* path, __nv_bfloat16* dst, int n) {{
    FILE* f = fopen(path, "rb");
    if (!f) {{ fprintf(stderr, "Cannot open %s\\n", path); exit(1); }}
    fread(dst, 2, n, f); fclose(f);
}}

int main() {{
    const int rows={rows}, hidden={hidden}, N={n}, NB={nb};
    const int qb_per_row = {qblocks_per_row};

    __nv_bfloat16 *h_in  = (__nv_bfloat16*)malloc(N*2);
    __nv_bfloat16 *h_res = (__nv_bfloat16*)malloc(N*2);
    __nv_bfloat16 *h_w   = (__nv_bfloat16*)malloc(hidden*2);
    __nv_bfloat16 *h_ref_ro = (__nv_bfloat16*)malloc(N*2);
    __nv_bfloat16 *h_ref_no = (__nv_bfloat16*)malloc(N*2);

    load_bf16("{ref_dir}/input.bin",        h_in,     N);
    load_bf16("{ref_dir}/residual.bin",     h_res,    N);
    load_bf16("{ref_dir}/weight.bin",       h_w,      hidden);
    load_bf16("{ref_dir}/residual_out.bin", h_ref_ro, N);
    load_bf16("{ref_dir}/norm_out.bin",     h_ref_no, N);

    __nv_bfloat16 *di, *dr, *dw, *dro;
    unsigned char *dq; __nv_fp8_storage_t *ds;
    cudaMalloc(&di, N*2); cudaMemcpy(di, h_in, N*2, cudaMemcpyHostToDevice);
    cudaMalloc(&dr, N*2); cudaMemcpy(dr, h_res, N*2, cudaMemcpyHostToDevice);
    cudaMalloc(&dw, hidden*2); cudaMemcpy(dw, h_w, hidden*2, cudaMemcpyHostToDevice);
    cudaMalloc(&dro, N*2);
    cudaMalloc(&dq, N/2);
    cudaMalloc(&ds, NB);

    cudaStream_t s; cudaStreamCreate(&s);
    launch_fused_add_rmsnorm_nvfp4(di, dr, dw, dro, dq, ds, rows, hidden, s);
    cudaStreamSynchronize(s);

    /* --- Check 1: residual_out (bf16 exact) --- */
    __nv_bfloat16 *h_out = (__nv_bfloat16*)malloc(N*2);
    cudaMemcpy(h_out, dro, N*2, cudaMemcpyDeviceToHost);

    float maxe_ro = 0.f; int miss_ro = 0;
    for (int i = 0; i < N; ++i) {{
        float ref = __bfloat162float(h_ref_ro[i]);
        float can = __bfloat162float(h_out[i]);
        float e = fabsf(ref - can);
        if (e > maxe_ro) maxe_ro = e;
        if (e > {atol}f + {rtol}f * fabsf(ref)) miss_ro++;
    }}

    if (miss_ro > 0) {{
        printf("CORRECTNESS: FAIL (max_abs_err=%.6f mismatches=%d/%d residual_out mismatch)\\n",
               maxe_ro, miss_ro, N);
        return 1;
    }}

    /* --- Check 2: FP4 quantized output (dequant vs norm_out) --- */
    unsigned char *h_qo = (unsigned char*)malloc(N/2);
    unsigned char *h_sc = (unsigned char*)malloc(NB);
    cudaMemcpy(h_qo, dq, N/2, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_sc, ds, NB, cudaMemcpyDeviceToHost);

    float maxe_q = 0.f; int miss_q = 0; int zero_blocks = 0;

    for (int idx = 0; idx < NB; ++idx) {{
        int r  = idx / qb_per_row;
        int qb = idx % qb_per_row;
        float scale = decode_e4m3(h_sc[idx]);
        float tol = fmaxf(scale * 1.5f, 0.01f);
        int packed_base = idx * 8;
        int elem_base   = r * hidden + qb * 16;

        int all_zero = 1;
        for (int j = 0; j < 8; ++j) {{
            unsigned char byte = h_qo[packed_base + j];
            float lo = kFP4LUT[byte & 0xF] * scale;
            float hi = kFP4LUT[byte >> 4]  * scale;
            if (lo != 0.0f || hi != 0.0f) all_zero = 0;

            float ref_lo = __bfloat162float(h_ref_no[elem_base + 2*j]);
            float ref_hi = __bfloat162float(h_ref_no[elem_base + 2*j + 1]);
            float e_lo = fabsf(ref_lo - lo);
            float e_hi = fabsf(ref_hi - hi);
            if (e_lo > maxe_q) maxe_q = e_lo;
            if (e_hi > maxe_q) maxe_q = e_hi;
            if (e_lo > tol) miss_q++;
            if (e_hi > tol) miss_q++;
        }}
        if (all_zero) zero_blocks++;
    }}

    float zero_frac = (float)zero_blocks / NB;
    if (zero_frac > 0.5f) {{
        printf("CORRECTNESS: FAIL (max_abs_err=%.6f quant_zero_blocks=%.0f%% — FP4 output is empty)\\n",
               maxe_q, zero_frac * 100.f);
        return 1;
    }}

    float miss_q_frac = (float)miss_q / N;
    if (miss_q_frac > 0.05f) {{
        printf("CORRECTNESS: FAIL (max_abs_err=%.6f quant_mismatches=%d/%d (%.1f%%) — FP4 output wrong)\\n",
               maxe_q, miss_q, N, miss_q_frac * 100.f);
        return 1;
    }}

    printf("CORRECTNESS: PASS (max_abs_err=%.6f N=%d quant_err=%.4f quant_zero=%d/%d)\\n",
           maxe_ro, N, maxe_q, zero_blocks, NB);
    return 0;
}}
"""


def _flashinfer_harness_silu_mul(shape: tuple, ref_dir: str,
                                  atol: float, rtol: float) -> str:
    b, m, k = shape
    n = b * m * k
    nb = n // 16
    return f"""
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
{_cuda_harness_prelude()}

void launch_silu_mul_fp4quant(
    const __nv_bfloat16*, const __nv_bfloat16*,
    uint8_t*, __nv_fp8_storage_t*, int, cudaStream_t);

/* FP4 decode LUT */
static const float kFP4LUT[16] = {{
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
   -0.0f,-0.5f,-1.0f,-1.5f,-2.0f,-3.0f,-4.0f,-6.0f
}};

static float decode_e4m3(unsigned char x) {{
    unsigned int sign = (x & 0x80u);
    unsigned int exp  = (x >> 3) & 0xFu;
    unsigned int mant = x & 0x7u;
    float val;
    if (exp == 0u) val = (mant / 8.0f) * (1.0f / 64.0f);
    else           val = (1.0f + mant / 8.0f) * powf(2.0f, (float)exp - 7.0f);
    return sign ? -val : val;
}}

static void load_bf16(const char* path, __nv_bfloat16* dst, int n) {{
    FILE* f = fopen(path, "rb");
    if (!f) {{ fprintf(stderr, "Cannot open %s\\n", path); exit(1); }}
    fread(dst, 2, n, f); fclose(f);
}}

int main() {{
    const int N={n}, NB={nb};

    __nv_bfloat16 *h_gate = (__nv_bfloat16*)malloc(N*2);
    __nv_bfloat16 *h_up   = (__nv_bfloat16*)malloc(N*2);
    load_bf16("{ref_dir}/gate.bin", h_gate, N);
    load_bf16("{ref_dir}/up.bin",   h_up,   N);

    __nv_bfloat16 *dg, *du;
    uint8_t *dq; __nv_fp8_storage_t *ds;
    cudaMalloc(&dg, N*2); cudaMemcpy(dg, h_gate, N*2, cudaMemcpyHostToDevice);
    cudaMalloc(&du, N*2); cudaMemcpy(du, h_up,   N*2, cudaMemcpyHostToDevice);
    cudaMalloc(&dq, N/2);
    cudaMalloc(&ds, NB);

    cudaStream_t s; cudaStreamCreate(&s);
    launch_silu_mul_fp4quant(dg, du, dq, ds, N, s);
    cudaStreamSynchronize(s);

    /* Copy back quantized output */
    unsigned char *h_packed = (unsigned char*)malloc(N/2);
    unsigned char *h_scales = (unsigned char*)malloc(NB);
    cudaMemcpy(h_packed, dq, N/2, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_scales, ds, NB, cudaMemcpyDeviceToHost);

    /* Dequant FP4 output and compare against expected silu(gate)*up */
    float maxe = 0.0f; int miss = 0; int zero_blocks = 0;

    for (int blk = 0; blk < NB; ++blk) {{
        float scale = decode_e4m3(h_scales[blk]);
        float tol = fmaxf(scale * 1.5f, 0.01f);
        int packed_base = blk * 8;
        int elem_base   = blk * 16;

        int all_zero = 1;
        for (int j = 0; j < 8; ++j) {{
            unsigned char byte = h_packed[packed_base + j];
            float dq_lo = kFP4LUT[byte & 0xF] * scale;
            float dq_hi = kFP4LUT[byte >> 4]  * scale;
            if (dq_lo != 0.0f || dq_hi != 0.0f) all_zero = 0;

            /* Expected: silu(gate) * up = gate / (1 + exp(-gate)) * up */
            float g0 = __bfloat162float(h_gate[elem_base + 2*j]);
            float u0 = __bfloat162float(h_up[elem_base + 2*j]);
            float exp0 = g0 / (1.0f + expf(-g0)) * u0;

            float g1 = __bfloat162float(h_gate[elem_base + 2*j + 1]);
            float u1 = __bfloat162float(h_up[elem_base + 2*j + 1]);
            float exp1 = g1 / (1.0f + expf(-g1)) * u1;

            float e0 = fabsf(exp0 - dq_lo);
            float e1 = fabsf(exp1 - dq_hi);
            if (e0 > maxe) maxe = e0;
            if (e1 > maxe) maxe = e1;
            if (e0 > tol) miss++;
            if (e1 > tol) miss++;
        }}
        if (all_zero) zero_blocks++;
    }}

    float zero_frac = (float)zero_blocks / NB;
    if (zero_frac > 0.5f) {{
        printf("CORRECTNESS: FAIL (max_abs_err=%.6f zero_blocks=%.0f%% — FP4 output is empty)\\n",
               maxe, zero_frac * 100.f);
        return 1;
    }}

    float miss_frac = (float)miss / N;
    if (miss_frac > 0.05f) {{
        printf("CORRECTNESS: FAIL (max_abs_err=%.6f mismatches=%d/%d (%.1f%%) — silu*mul+quant mismatch)\\n",
               maxe, miss, N, miss_frac * 100.f);
        return 1;
    }}

    printf("CORRECTNESS: PASS (max_abs_err=%.6f N=%d zero_blocks=%d/%d)\\n",
           maxe, N, zero_blocks, NB);
    return 0;
}}
"""


