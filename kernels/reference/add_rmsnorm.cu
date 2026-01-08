// add_rmsnorm.cu — Naive fused Add + RMSNorm + NVFP4 quantize
// Intentionally unoptimized baseline matching KernelArena's reference style.
// Operation: residual_out = input + residual
//            norm_out = RMSNorm(residual_out) * weight
//            quantize norm_out to NVFP4 (block_size=16, E4M3 scales)

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>

#if __has_include(<cuda_fp8.h>)
#include <cuda_fp8.h>
#endif

#ifndef __CUDA_FP8_TYPES_EXIST__
typedef unsigned char __nv_fp8_storage_t;
#endif

// ── E4M3 scale helpers ──────────────────────────────────────────────────────

__device__ __forceinline__ __nv_fp8_storage_t float_to_e4m3(float x) {
#if defined(__NV_E4M3) && defined(__nv_cvt_float_to_fp8)
    return __nv_cvt_float_to_fp8(x, __NV_SATFINITE, __NV_E4M3);
#else
    unsigned int bits;
    memcpy(&bits, &x, 4);
    unsigned int sign = (bits >> 24) & 0x80u;
    float ax = fabsf(x);
    if (ax < 1.9531e-03f) return (__nv_fp8_storage_t)sign;
    if (ax > 448.0f) ax = 448.0f;
    int e = 0;
    float m = frexpf(ax, &e);
    e += 6;
    if (e < 0) e = 0;
    if (e > 15) e = 15;
    int mant = (int)((m * 2.0f - 1.0f) * 8.0f + 0.5f);
    if (mant > 7) mant = 7;
    return (__nv_fp8_storage_t)(sign | (e << 3) | mant);
#endif
}

__device__ __forceinline__ float e4m3_to_float(__nv_fp8_storage_t x) {
#if defined(__NV_E4M3) && defined(__nv_cvt_fp8_to_float)
    return __nv_cvt_fp8_to_float(x, __NV_E4M3);
#else
    unsigned int exp  = (x >> 3) & 0xFu;
    unsigned int mant = x & 0x7u;
    float val;
    if (exp == 0u) {
        val = (mant / 8.0f) * (1.0f / 64.0f);
    } else {
        val = (1.0f + mant / 8.0f) * powf(2.0f, (float)exp - 7.0f);
    }
    return val;
#endif
}

// ── NVFP4 encode ────────────────────────────────────────────────────────────

#define NVFP4_BLOCK_SIZE 16

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

// ── Naive fused kernel ──────────────────────────────────────────────────────
// One block per row. Scalar bf16 loads, shared memory reduction.
// No vectorization, no warp shuffles, two passes over data.

#define BLOCK_THREADS 256

__global__ void fused_add_rmsnorm_nvfp4_kernel(
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ residual,
    const __nv_bfloat16* __restrict__ rms_weight,
    __nv_bfloat16*       __restrict__ residual_out,
    uint8_t*             __restrict__ quant_out,
    __nv_fp8_storage_t*  __restrict__ quant_scales,
    int hidden_size,
    float eps)
{
    extern __shared__ float smem[];

    int row = blockIdx.x;
    int tid = threadIdx.x;
    int base = row * hidden_size;

    // Phase 1: Add input + residual, write to residual_out, compute sum of squares
    float local_ss = 0.0f;
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        float a = __bfloat162float(input[base + i]);
        float r = __bfloat162float(residual[base + i]);
        float x = a + r;
        residual_out[base + i] = __float2bfloat16(x);
        local_ss += x * x;
    }

    // Shared memory tree reduction (naive — no warp shuffles)
    smem[tid] = local_ss;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride)
            smem[tid] += smem[tid + stride];
        __syncthreads();
    }

    float rms_inv = rsqrtf(smem[0] / hidden_size + eps);

    // Phase 2: Re-read residual_out, normalize, quantize to NVFP4
    int num_quant_blocks = hidden_size / NVFP4_BLOCK_SIZE;
    for (int qb = tid; qb < num_quant_blocks; qb += blockDim.x) {
        int elem_base = qb * NVFP4_BLOCK_SIZE;
        float block_vals[NVFP4_BLOCK_SIZE];

        // Read and normalize
        for (int j = 0; j < NVFP4_BLOCK_SIZE; ++j) {
            float x = __bfloat162float(residual_out[base + elem_base + j]);
            float w = __bfloat162float(rms_weight[elem_base + j]);
            block_vals[j] = x * rms_inv * w;
        }

        // Find block absmax
        float amax = 0.0f;
        for (int j = 0; j < NVFP4_BLOCK_SIZE; ++j)
            amax = fmaxf(amax, fabsf(block_vals[j]));

        // Compute E4M3 scale
        float s = (amax > 0.0f) ? (amax / 6.0f) : 1.0f;
        __nv_fp8_storage_t scale_e4m3 = float_to_e4m3(s);
        float inv_s = (amax > 0.0f) ? (6.0f / amax) : 1.0f;

        // Pack FP4 pairs into bytes
        int packed_base = (row * num_quant_blocks + qb) * (NVFP4_BLOCK_SIZE / 2);
        for (int j = 0; j < NVFP4_BLOCK_SIZE / 2; ++j) {
            uint8_t lo = float_to_nvfp4(block_vals[2*j]   * inv_s);
            uint8_t hi = float_to_nvfp4(block_vals[2*j+1] * inv_s);
            quant_out[packed_base + j] = (hi << 4) | (lo & 0xF);
        }
        quant_scales[row * num_quant_blocks + qb] = scale_e4m3;
    }
}

// Host launch wrapper — signature must not change
void launch_fused_add_rmsnorm_nvfp4(
    const __nv_bfloat16* input, const __nv_bfloat16* residual,
    const __nv_bfloat16* rms_weight, __nv_bfloat16* residual_out,
    uint8_t* quant_out, __nv_fp8_storage_t* quant_scales,
    int num_rows, int hidden_size, cudaStream_t stream)
{
    dim3 grid(num_rows);
    dim3 block(BLOCK_THREADS);
    size_t smem = BLOCK_THREADS * sizeof(float);
    fused_add_rmsnorm_nvfp4_kernel<<<grid, block, smem, stream>>>(
        input, residual, rms_weight, residual_out,
        quant_out, quant_scales, hidden_size, 1e-6f);
}
