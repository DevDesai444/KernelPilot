// nvfp4_utils.cuh — NVFP4 pack/unpack and quantization helpers
// NVFP4: 1-bit exponent, 2-bit mantissa, 1-bit sign
// Format: s | e | m1 | m0  (4 bits total)
// Scale: per-16-element block scaling factor (E4M3 fp8)

#pragma once
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdint.h>

// Try to include FP8/FP4 headers; provide fallback if unavailable
#if __has_include(<cuda_fp8.h>)
#include <cuda_fp8.h>
#endif
#if __has_include(<cuda_fp4.h>)
#include <cuda_fp4.h>
#endif

// ────────────────────────────────────────────────────────────────────────────
// FP8 storage type (fallback if cuda_fp8.h not available)
// ────────────────────────────────────────────────────────────────────────────
#ifndef __CUDA_FP8_TYPES_EXIST__
#ifndef _NVFP4_FP8_FALLBACK_
#define _NVFP4_FP8_FALLBACK_
typedef unsigned char __nv_fp8_storage_t;
#endif
#endif

// ────────────────────────────────────────────────────────────────────────────
// E4M3 scale helpers — scales are stored as FP8 E4M3 (1 byte per block)
// Uses hardware intrinsics when available, software fallback otherwise.
// ────────────────────────────────────────────────────────────────────────────
__device__ __forceinline__ __nv_fp8_storage_t float_to_e4m3(float x) {
#if defined(__NV_E4M3) && defined(__nv_cvt_float_to_fp8)
    return __nv_cvt_float_to_fp8(x, __NV_SATFINITE, __NV_E4M3);
#else
    // Software E4M3 encode: sign(1) + exponent(4) + mantissa(3), bias=7
    unsigned int bits;
    memcpy(&bits, &x, 4);
    unsigned int sign = (bits >> 24) & 0x80u;
    float ax = fabsf(x);
    if (ax < 1.9531e-03f) return (__nv_fp8_storage_t)sign;
    if (ax > 448.0f) ax = 448.0f;
    int e = 0;
    float m = frexpf(ax, &e);      // 0.5 <= m < 1.0, ax = m * 2^e
    e += 6;                         // bias=7, frexp gives 2^(e-1)
    if (e < 0) e = 0;
    if (e > 15) e = 15;
    int mant = (int)((m * 2.0f - 1.0f) * 8.0f + 0.5f);  // 3-bit mantissa
    if (mant > 7) mant = 7;
    return (__nv_fp8_storage_t)(sign | (e << 3) | mant);
#endif
}

__device__ __forceinline__ float e4m3_to_float(__nv_fp8_storage_t x) {
#if defined(__NV_E4M3) && defined(__nv_cvt_fp8_to_float)
    return __nv_cvt_fp8_to_float(x, __NV_E4M3);
#else
    // Software E4M3 decode
    unsigned int sign = (x & 0x80u) ? 1u : 0u;
    unsigned int exp  = (x >> 3) & 0xFu;
    unsigned int mant = x & 0x7u;
    float val;
    if (exp == 0u) {
        val = (mant / 8.0f) * powf(2.0f, -6.0f);       // subnormal
    } else {
        val = (1.0f + mant / 8.0f) * powf(2.0f, (float)exp - 7.0f);
    }
    return sign ? -val : val;
#endif
}

// ────────────────────────────────────────────────────────────────────────────
// NVFP4 representation constants
// ────────────────────────────────────────────────────────────────────────────
#define NVFP4_MANTISSA_BITS 2
#define NVFP4_EXPONENT_BITS 1
#define NVFP4_BIAS          1
#define NVFP4_BLOCK_SIZE    16      // elements per scale factor
#define NVFP4_PER_BYTE      2       // 2 fp4 values packed per byte
#define NVFP4_PER_INT32     8       // 8 fp4 values packed per int32

// ────────────────────────────────────────────────────────────────────────────
// Lookup table: all 16 NVFP4 values (sign × exp × mant)
// Positive values: 0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0
// Negative values: mirror of positive
// ────────────────────────────────────────────────────────────────────────────
__device__ __constant__ float kNVFP4LUT[16] = {
    0.0f,   0.5f,   1.0f,   1.5f,   // 0b0000 .. 0b0011  (positive, exp=0)
    2.0f,   3.0f,   4.0f,   6.0f,   // 0b0100 .. 0b0111  (positive, exp=1)
   -0.0f,  -0.5f,  -1.0f,  -1.5f,  // 0b1000 .. 0b1011  (negative, exp=0)
   -2.0f,  -3.0f,  -4.0f,  -6.0f,  // 0b1100 .. 0b1111  (negative, exp=1)
};
