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


def _flashinfer_harness_nvfp4_quantize(shape: tuple, ref_dir: str,
                                        atol: float, rtol: float) -> str:
    m, k = shape
    n = m * k
    nb = n // 16
    return f"""
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <stdint.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
{_cuda_harness_prelude()}

void launch_nvfp4_quantize_bf16(
    const __nv_bfloat16*, uint8_t*, __nv_fp8_storage_t*, int, cudaStream_t);

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

    __nv_bfloat16 *h_in = (__nv_bfloat16*)malloc(N*2);
    load_bf16("{ref_dir}/input.bin", h_in, N);

    __nv_bfloat16 *din; uint8_t *dpk; __nv_fp8_storage_t *dsc;
    cudaMalloc(&din, N*2); cudaMemcpy(din, h_in, N*2, cudaMemcpyHostToDevice);
    cudaMalloc(&dpk, N/2);
    cudaMalloc(&dsc, NB);

    cudaStream_t s; cudaStreamCreate(&s);
    launch_nvfp4_quantize_bf16(din, dpk, dsc, N, s);
    cudaStreamSynchronize(s);

    /* Copy back quantized output */
    unsigned char *h_packed = (unsigned char*)malloc(N/2);
    unsigned char *h_scales = (unsigned char*)malloc(NB);
    cudaMemcpy(h_packed, dpk, N/2, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_scales, dsc, NB, cudaMemcpyDeviceToHost);

    /* Dequant FP4 and compare against original input (round-trip check) */
    float maxe = 0.0f; int miss = 0; int zero_blocks = 0;

    for (int blk = 0; blk < NB; ++blk) {{
        float scale = decode_e4m3(h_scales[blk]);
        /* FP4 max quantization error = scale * 1.0 (half the largest step) */
        float tol = fmaxf(scale * 1.5f, 0.01f);
        int packed_base = blk * 8;
        int elem_base   = blk * 16;

        int all_zero = 1;
        for (int j = 0; j < 8; ++j) {{
            unsigned char byte = h_packed[packed_base + j];
            float dq_lo = kFP4LUT[byte & 0xF] * scale;
            float dq_hi = kFP4LUT[byte >> 4]  * scale;
            if (dq_lo != 0.0f || dq_hi != 0.0f) all_zero = 0;

            float ref_lo = __bfloat162float(h_in[elem_base + 2*j]);
            float ref_hi = __bfloat162float(h_in[elem_base + 2*j + 1]);
            float e_lo = fabsf(ref_lo - dq_lo);
            float e_hi = fabsf(ref_hi - dq_hi);
            if (e_lo > maxe) maxe = e_lo;
            if (e_hi > maxe) maxe = e_hi;
            if (e_lo > tol) miss++;
            if (e_hi > tol) miss++;
        }}
        if (all_zero) zero_blocks++;
    }}

    /* Reject if >50% of blocks are all-zero (likely no-op kernel) */
    float zero_frac = (float)zero_blocks / NB;
    if (zero_frac > 0.5f) {{
        printf("CORRECTNESS: FAIL (max_abs_err=%.6f zero_blocks=%.0f%% — FP4 output is empty)\\n",
               maxe, zero_frac * 100.f);
        return 1;
    }}

    float miss_frac = (float)miss / N;
    if (miss_frac > 0.05f) {{
        printf("CORRECTNESS: FAIL (max_abs_err=%.6f mismatches=%d/%d (%.1f%%) — quantization mismatch)\\n",
               maxe, miss, N, miss_frac * 100.f);
        return 1;
    }}

    printf("CORRECTNESS: PASS (max_abs_err=%.6f N=%d zero_blocks=%d/%d)\\n",
           maxe, N, zero_blocks, NB);
    return 0;
}}
"""


def _generate_cuda_reference_harness(rows: int, hidden: int, atol: float, rtol: float) -> str:
    """Fallback: hand-written CUDA reference (used when FlashInfer is unavailable)."""
    return f"""
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
{_cuda_harness_prelude()}

__device__ float _wref_sum(float v) {{
    for (int m = 16; m > 0; m >>= 1) v += __shfl_xor_sync(0xFFFFFFFF, v, m);
    return v;
}}

__global__ void reference_kernel(
    const __nv_bfloat16* in, const __nv_bfloat16* res,
    const __nv_bfloat16* w, __nv_bfloat16* ro,
    unsigned char* qo, __nv_fp8_storage_t* sc,
    int hidden, float eps)
{{
    extern __shared__ float sm[];
    int row = blockIdx.x, tid = threadIdx.x;
    int wid = tid/32, lane = tid%32, nw = blockDim.x/32;
    int base = row * hidden;
    float ss = 0.0f;
    for (int i = tid; i < hidden; i += blockDim.x) {{
        float a = __bfloat162float(in[base+i]) + __bfloat162float(res[base+i]);
        ro[base+i] = __float2bfloat16(a);
        ss += a*a;
    }}
    ss = _wref_sum(ss);
    if (lane == 0) sm[wid] = ss;
    __syncthreads();
    if (wid == 0) {{ float v = (lane<nw)?sm[lane]:0.f; v=_wref_sum(v); if(!lane) sm[0]=v; }}
    __syncthreads();
    float ri = rsqrtf(sm[0]/hidden + eps);
    int nb = hidden/16;
    for (int qb = tid; qb < nb; qb += blockDim.x) {{
        int eb = qb*16;
        float amax = 0.f, vals[16];
        for (int j=0;j<16;++j) {{
            float x = __bfloat162float(ro[base+eb+j]) * ri * __bfloat162float(w[eb+j]);
            vals[j]=x; amax=fmaxf(amax,fabsf(x));
        }}
        float s = (amax>0)?amax/6.f:1.f;
        sc[row*nb+qb] = __nv_cvt_float_to_fp8(s, __NV_SATFINITE, __NV_E4M3);
        float is = (amax>0)?6.f/amax:1.f;
        int pb = (row*nb+qb)*8;
        for (int j=0;j<8;++j) {{
            int lo = (int)fmaxf(0.f,fminf(15.f,(vals[2*j]*is/6.f+1.f)*7.5f));
            int hi = (int)fmaxf(0.f,fminf(15.f,(vals[2*j+1]*is/6.f+1.f)*7.5f));
            qo[pb+j] = (unsigned char)((hi<<4)|lo);
        }}
    }}
}}

void launch_fused_add_rmsnorm_nvfp4(
    const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*,
    __nv_bfloat16*, unsigned char*, __nv_fp8_storage_t*, int, int, cudaStream_t);

int main() {{
    const int rows={rows}, hidden={hidden}, N=rows*hidden, nb=N/16;
    __nv_bfloat16 *hi,*hr,*hw, *di,*dr,*dw,*rr,*rc;
    __nv_fp8_storage_t *sr,*sc;
    unsigned char *qr,*qc;
    hi=(__nv_bfloat16*)malloc(N*2); hr=(__nv_bfloat16*)malloc(N*2); hw=(__nv_bfloat16*)malloc(hidden*2);
    srand(42);
    for(int i=0;i<N;++i){{ hi[i]=__float2bfloat16((float)rand()/RAND_MAX-0.5f); hr[i]=__float2bfloat16((float)rand()/RAND_MAX-0.5f); }}
    for(int i=0;i<hidden;++i) hw[i]=__float2bfloat16(1.f+(float)rand()/RAND_MAX*0.1f);
    cudaMalloc(&di,N*2); cudaMemcpy(di,hi,N*2,cudaMemcpyHostToDevice);
    cudaMalloc(&dr,N*2); cudaMemcpy(dr,hr,N*2,cudaMemcpyHostToDevice);
    cudaMalloc(&dw,hidden*2); cudaMemcpy(dw,hw,hidden*2,cudaMemcpyHostToDevice);
    cudaMalloc(&rr,N*2); cudaMalloc(&rc,N*2);
    cudaMalloc(&qr,N/2); cudaMalloc(&qc,N/2);
    cudaMalloc(&sr,nb); cudaMalloc(&sc,nb);
    reference_kernel<<<rows,128,(128/32)*sizeof(float)>>>(di,dr,dw,rr,qr,sr,hidden,1e-6f);
    cudaStream_t s; cudaStreamCreate(&s);
    launch_fused_add_rmsnorm_nvfp4(di,dr,dw,rc,qc,sc,rows,hidden,s);
    cudaStreamSynchronize(s);
    __nv_bfloat16 *h_rr=(__nv_bfloat16*)malloc(N*2), *h_rc=(__nv_bfloat16*)malloc(N*2);
    cudaMemcpy(h_rr,rr,N*2,cudaMemcpyDeviceToHost);
    cudaMemcpy(h_rc,rc,N*2,cudaMemcpyDeviceToHost);
    float maxe=0.f; int miss=0;
    for(int i=0;i<N;++i){{
        float ref=__bfloat162float(h_rr[i]), can=__bfloat162float(h_rc[i]);
        float e=fabsf(ref-can); if(e>maxe) maxe=e;
        if(e>{atol}f+{rtol}f*fabsf(ref)) miss++;
    }}
    if(miss==0) {{ printf("CORRECTNESS: PASS (max_abs_err=%.6f N=%d)\\n",maxe,N); return 0; }}
    else {{ printf("CORRECTNESS: FAIL (max_abs_err=%.6f mismatches=%d/%d)\\n",maxe,miss,N); return 1; }}
}}
"""


class CorrectnessChecker:

    def __init__(self, config: dict):
        self.atol = config["eval"]["correctness_atol"]
        self.rtol = config["eval"]["correctness_rtol"]
        seeds_cfg = config["eval"].get("correctness_seeds", [42, 123, 999])
        if isinstance(seeds_cfg, int):
            self.correctness_seeds = [42 + idx for idx in range(max(1, seeds_cfg))]
        else:
            self.correctness_seeds = [int(seed) for seed in seeds_cfg] or [42]
        self.nvcc_flags = [
            "-O2", "-arch=sm_100a", "-std=c++17",
            f"-I{PROJECT_ROOT / 'kernels' / 'common'}",
            f"-I{PROJECT_ROOT}",
        ]

    def check(self, candidate_src: str, problem_shape: tuple,
              kernel_type: str = "add_rmsnorm") -> tuple:
        """Returns (passed, max_abs_err, message)."""
        clean, hack_type = is_clean(candidate_src)
        if not clean:
            return False, float("inf"), f"Hack detected: {hack_type}"

        # Try FlashInfer reference first
        if flashinfer_ref.available():
            return self._check_with_flashinfer(candidate_src, problem_shape, kernel_type)

        # Fallback to CUDA reference (add_rmsnorm only)
        if kernel_type == "add_rmsnorm":
            return self._check_with_cuda_ref(candidate_src, problem_shape)

        return False, float("inf"), f"No reference available for {kernel_type} (install flashinfer)"

    def _check_with_flashinfer(self, candidate_src: str, shape: tuple,
                                kernel_type: str) -> tuple:
        """Check correctness using FlashInfer-generated reference data."""
        import torch

        last_err = float("inf")
        last_msg = "FlashInfer reference unavailable"
        for seed in self.correctness_seeds:
            ref_data = flashinfer_ref.generate_reference(kernel_type, shape, seed=seed)
            if ref_data is None:
                logger.warning("FlashInfer reference generation failed for seed %s", seed)
                if kernel_type == "add_rmsnorm":
                    return self._check_with_cuda_ref(candidate_src, shape)
                return False, float("inf"), "FlashInfer reference unavailable"

            with tempfile.TemporaryDirectory() as tmpdir:
                for name, tensor in ref_data.items():
                    if isinstance(tensor, torch.Tensor):
                        with open(f"{tmpdir}/{name}.bin", "wb") as bf:
                            bf.write(
                                tensor.cpu().contiguous().numpy(force=True).tobytes()
                                if tensor.dtype not in (torch.bfloat16, torch.float8_e4m3fn)
                                else bytes(tensor.cpu().contiguous().untyped_storage())
                            )

                harness = _generate_flashinfer_harness(kernel_type, shape, tmpdir,
                                                        self.atol, self.rtol)
                if harness is None:
                    return False, float("inf"), f"No harness for {kernel_type}"

                passed, err, msg = self._compile_and_run(candidate_src, harness, tmpdir)
                if not passed:
                    return False, err, f"seed={seed} {msg}"
                last_err = err
                last_msg = msg

        return True, last_err, f"{last_msg} seeds={self.correctness_seeds}"

    def _check_with_cuda_ref(self, candidate_src: str, shape: tuple) -> tuple:
        """Fallback: check using hand-written CUDA reference kernel."""
        rows, hidden = shape
        harness = _generate_cuda_reference_harness(rows, hidden, self.atol, self.rtol)

        with tempfile.TemporaryDirectory() as tmpdir:
            return self._compile_and_run(candidate_src, harness, tmpdir)

    def _compile_and_run(self, candidate_src: str, harness: str, tmpdir: str) -> tuple:
        """Compile candidate + harness and run correctness check."""
        combined_src = candidate_src + "\n\n" + harness
        src_file = Path(tmpdir) / "correctness_check.cu"
        bin_file = Path(tmpdir) / "correctness_check"
        src_file.write_text(combined_src)

        cmd = ["nvcc"] + self.nvcc_flags + [str(src_file), "-o", str(bin_file)]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        except subprocess.TimeoutExpired:
            return False, float("inf"), "Compile timed out after 120s"
        if r.returncode != 0:
            return False, float("inf"), f"Compile failed: {r.stderr[:300]}"

        try:
            r2 = subprocess.run([str(bin_file)], capture_output=True, text=True, timeout=30)
        except subprocess.TimeoutExpired:
            return False, float("inf"), "Kernel hung (timeout) — likely deadlock or infinite loop"
        output = r2.stdout.strip()

        if "PASS" in output:
            m = re.search(r"max_abs_err=([\d.]+)", output)
            err = float(m.group(1)) if m else 0.0
            return True, err, output
        elif "FAIL" in output:
            m = re.search(r"max_abs_err=([\d.]+)", output)
            err = float(m.group(1)) if m else float("inf")
            return False, err, output
        else:
            return False, float("inf"), f"Unexpected output: {output}"
