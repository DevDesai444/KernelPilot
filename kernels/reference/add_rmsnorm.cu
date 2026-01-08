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
