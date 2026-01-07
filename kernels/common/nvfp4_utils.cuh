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

// ────────────────────────────────────────────────────────────────────────────
// Core encode: float → 4-bit code (two-step: clamp → scale → round-to-nearest)
//
// NVFP4 positive magnitudes: 0, 0.5, 1, 1.5, 2, 3, 4, 6
// The spacing between levels is non-uniform (0.5 up to 2, then 1, then 2).
// We map the scaled value to the nearest representable level using a lookup
// into 8 boundary midpoints, which avoids a 16-iteration linear scan and
// gives the compiler a branchless cmov sequence instead.
//
// Boundary midpoints between consecutive positive levels:
//   0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0  (7 thresholds for 8 levels)
// ────────────────────────────────────────────────────────────────────────────
__device__ __forceinline__ uint8_t float_to_nvfp4(float x) {
    // Extract sign and work with |x| scaled to [0, 6] range
    uint8_t sign_bit = (x < 0.0f) ? 0x8u : 0x0u;
    float   ax       = fabsf(x);

    // Clamp to representable range [0, 6]
    ax = fminf(ax, 6.0f);

    // Map magnitude to 3-bit mantissa+exponent code (0..7 → levels 0,0.5,1,1.5,2,3,4,6)
    // Threshold comparisons collapse to a sequence of SETP/SELP on B200 PTX
    uint8_t code;
    if      (ax < 0.25f) code = 0;   // → 0.0
    else if (ax < 0.75f) code = 1;   // → 0.5
    else if (ax < 1.25f) code = 2;   // → 1.0
    else if (ax < 1.75f) code = 3;   // → 1.5
    else if (ax < 2.5f)  code = 4;   // → 2.0
    else if (ax < 3.5f)  code = 5;   // → 3.0
    else if (ax < 5.0f)  code = 6;   // → 4.0
    else                 code = 7;   // → 6.0

    return sign_bit | code;
}

// ────────────────────────────────────────────────────────────────────────────
// Hardware FP4 pair conversion (sm_100a+): 2 scaled floats → 1 packed byte
// Usage: pack_fp4_pair(val0 * inv_scale, val1 * inv_scale)
// Values must be pre-scaled to [-6, 6] range before calling.
// ────────────────────────────────────────────────────────────────────────────
__device__ __forceinline__ uint8_t pack_fp4_pair(float a, float b) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
    return (uint8_t)__nv_cvt_float2_to_fp4x2(make_float2(a, b), __NV_E2M1, cudaRoundNearest);
#else
    return (float_to_nvfp4(b) << 4) | (float_to_nvfp4(a) & 0xF);
#endif
}

// ────────────────────────────────────────────────────────────────────────────
// Core decode: 4-bit code → float
// ────────────────────────────────────────────────────────────────────────────
__device__ __forceinline__ float nvfp4_to_float(uint8_t code) {
    return kNVFP4LUT[code & 0xF];
}

// ────────────────────────────────────────────────────────────────────────────
// Block quantize: 16 floats → 8 bytes (packed fp4) + 1 E4M3 scale
// Input:  float x[16]
// Output: uint8_t packed[8], __nv_fp8_storage_t scale (E4M3)
// ────────────────────────────────────────────────────────────────────────────
__device__ __forceinline__ void quantize_block_nvfp4(
    const float* __restrict__ x,
    uint8_t* __restrict__ packed,
    __nv_fp8_storage_t* __restrict__ scale)
{
    // Find block absmax for scaling
    float amax = 0.0f;
    #pragma unroll
    for (int i = 0; i < NVFP4_BLOCK_SIZE; ++i) {
        amax = fmaxf(amax, fabsf(x[i]));
    }

    // Scale so that max maps to representable NVFP4 max (6.0)
    const float inv_max_repr = 1.0f / 6.0f;
    float s = (amax > 0.0f) ? (amax * inv_max_repr) : 1.0f;
    *scale = float_to_e4m3(s);

    float inv_s = (amax > 0.0f) ? (6.0f / amax) : 1.0f;

    // Encode each element and pack 2 fp4 per byte
    #pragma unroll
    for (int i = 0; i < NVFP4_BLOCK_SIZE / 2; ++i) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
        // Hardware FP4 conversion — single instruction per pair on sm_100a
        float2 pair = make_float2(x[2*i] * inv_s, x[2*i+1] * inv_s);
        packed[i] = (uint8_t)__nv_cvt_float2_to_fp4x2(pair, __NV_E2M1, cudaRoundNearest);
#else
        uint8_t lo = float_to_nvfp4(x[2*i]   * inv_s);
        uint8_t hi = float_to_nvfp4(x[2*i+1] * inv_s);
        packed[i] = (hi << 4) | (lo & 0xF);
#endif
    }
}
