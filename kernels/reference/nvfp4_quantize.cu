// nvfp4_quantize.cu — Naive BF16→NVFP4 block quantization kernel
// Intentionally unoptimized baseline matching KernelArena's reference style.
// Layout: [num_blocks * 8 bytes data] + [num_blocks * 1 byte scale]

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

// ── Naive quantize kernel ───────────────────────────────────────────────────
// One thread per 16-element quant block. Scalar loads, no vectorization.

#define BLOCK_THREADS 256

__global__ void nvfp4_quantize_bf16_kernel(
    const __nv_bfloat16* __restrict__ input,
    uint8_t*             __restrict__ packed,
    __nv_fp8_storage_t*  __restrict__ scales,
    int N)
{
    int block_id = blockIdx.x * blockDim.x + threadIdx.x;
    int num_blocks = N / NVFP4_BLOCK_SIZE;
    if (block_id >= num_blocks) return;

    int base = block_id * NVFP4_BLOCK_SIZE;
    float x[NVFP4_BLOCK_SIZE];

    // Scalar loads
    for (int i = 0; i < NVFP4_BLOCK_SIZE; ++i)
        x[i] = __bfloat162float(input[base + i]);

    // Find block absmax
    float amax = 0.0f;
    for (int i = 0; i < NVFP4_BLOCK_SIZE; ++i)
        amax = fmaxf(amax, fabsf(x[i]));

    // Compute E4M3 scale
    float s = (amax > 0.0f) ? (amax / 6.0f) : 1.0f;
    __nv_fp8_storage_t scale_e4m3 = float_to_e4m3(s);
    float inv_s = (amax > 0.0f) ? (6.0f / amax) : 1.0f;

    // Pack FP4 pairs into bytes
    int packed_base = block_id * (NVFP4_BLOCK_SIZE / 2);
    for (int i = 0; i < NVFP4_BLOCK_SIZE / 2; ++i) {
        uint8_t lo = float_to_nvfp4(x[2*i]   * inv_s);
        uint8_t hi = float_to_nvfp4(x[2*i+1] * inv_s);
        packed[packed_base + i] = (hi << 4) | (lo & 0xF);
    }
    scales[block_id] = scale_e4m3;
}

// Host launch wrapper — signature must not change
void launch_nvfp4_quantize_bf16(
    const __nv_bfloat16* input, uint8_t* packed, __nv_fp8_storage_t* scales,
    int N, cudaStream_t stream)
{
    int num_blocks = N / NVFP4_BLOCK_SIZE;
    dim3 grid((num_blocks + BLOCK_THREADS - 1) / BLOCK_THREADS);
    nvfp4_quantize_bf16_kernel<<<grid, BLOCK_THREADS, 0, stream>>>(
        input, packed, scales, N);
}
