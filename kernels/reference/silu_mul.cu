// silu_mul.cu — Naive fused SiLU × Mul + NVFP4 quantize kernel
// Intentionally unoptimized baseline matching KernelArena's reference style.
// Operation: output = NVFP4_quantize(SiLU(gate) * up)
// SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>

#if __has_include(<cuda_fp8.h>)
#include <cuda_fp8.h>
#endif

#ifndef __CUDA_FP8_TYPES_EXIST__
typedef unsigned char __nv_fp8_storage_t;
#endif

#define NVFP4_BLOCK_SIZE 16

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

// ── NVFP4 encode ────────────────────────────────────────────────────────────

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
// One thread per 16-element quant block. Scalar loads, no vectorization.

#define BLOCK_THREADS 128

__global__ void silu_mul_fp4quant_kernel(
    const __nv_bfloat16* __restrict__ gate,
    const __nv_bfloat16* __restrict__ up,
    uint8_t*             __restrict__ quant_out,
    __nv_fp8_storage_t*  __restrict__ quant_scales,
    int N)
{
    int num_quant_blocks = N / NVFP4_BLOCK_SIZE;
    int qb = blockIdx.x * blockDim.x + threadIdx.x;
    if (qb >= num_quant_blocks) return;

    int elem_base = qb * NVFP4_BLOCK_SIZE;
    float block_vals[NVFP4_BLOCK_SIZE];

    // Compute SiLU(gate) * up for each element
    for (int j = 0; j < NVFP4_BLOCK_SIZE; ++j) {
        float g = __bfloat162float(gate[elem_base + j]);
        float u = __bfloat162float(up[elem_base + j]);
        float silu = g / (1.0f + expf(-g));
        block_vals[j] = silu * u;
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
    int packed_base = qb * (NVFP4_BLOCK_SIZE / 2);
    for (int j = 0; j < NVFP4_BLOCK_SIZE / 2; ++j) {
        uint8_t lo = float_to_nvfp4(block_vals[2*j]   * inv_s);
        uint8_t hi = float_to_nvfp4(block_vals[2*j+1] * inv_s);
        quant_out[packed_base + j] = (hi << 4) | (lo & 0xF);
    }
    quant_scales[qb] = scale_e4m3;
}

// Host launch wrapper — signature must not change
void launch_silu_mul_fp4quant(
    const __nv_bfloat16* gate, const __nv_bfloat16* up,
    uint8_t* quant_out, __nv_fp8_storage_t* quant_scales,
    int N, cudaStream_t stream)
{
    int num_quant_blocks = N / NVFP4_BLOCK_SIZE;
    dim3 grid((num_quant_blocks + BLOCK_THREADS - 1) / BLOCK_THREADS);
    dim3 block(BLOCK_THREADS);
    silu_mul_fp4quant_kernel<<<grid, block, 0, stream>>>(
        gate, up, quant_out, quant_scales, N);
}
