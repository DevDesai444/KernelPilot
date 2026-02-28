"""
cuda_docs.py — Local CUDA intrinsic reference for the search_docs tool.
Source: NVIDIA CUDA Math API Reference Manual 13.2
"""

CUDA_INTRINSICS_DB = [
    # ── FP4 Conversion (cuda_fp4.h) ──────────────────────────────────────────
    {
        "name": "__nv_cvt_float2_to_fp4x2",
        "signature": "__nv_fp4x2_storage_t __nv_cvt_float2_to_fp4x2(const float2 x, const __nv_fp4_interpretation_t fp4_interpretation, const enum cudaRoundMode rounding)",
        "description": "Convert two floats (packed as float2) to two FP4 values packed in one byte. Single hardware instruction on sm_100a. Note: Cast the result to (uint8_t) before storing to a uint8_t* array.",
        "header": "cuda_fp4.h",
        "example": "uint8_t packed = (uint8_t)__nv_cvt_float2_to_fp4x2(make_float2(a, b), __NV_E2M1, cudaRoundNearest);",
        "tags": ["fp4", "quantize", "convert", "float", "pack", "e2m1", "nvfp4"],
    },
    {
        "name": "__nv_cvt_float_to_fp4",
        "signature": "__nv_fp4_storage_t __nv_cvt_float_to_fp4(const float x, const __nv_fp4_interpretation_t fp4_interpretation, const enum cudaRoundMode rounding)",
        "description": "Convert single float to FP4.",
        "header": "cuda_fp4.h",
        "example": "__nv_fp4_storage_t val = __nv_cvt_float_to_fp4(x, __NV_E2M1, cudaRoundNearest);",
        "tags": ["fp4", "quantize", "convert", "float", "e2m1"],
    },
    {
        "name": "__nv_cvt_fp4_to_halfraw",
        "signature": "__half_raw __nv_cvt_fp4_to_halfraw(const __nv_fp4_storage_t x, const __nv_fp4_interpretation_t fp4_interpretation)",
        "description": "Convert FP4 value to half precision (fp16). Use __half_raw result with __half constructor.",
        "header": "cuda_fp4.h",
        "example": "__half_raw hr = __nv_cvt_fp4_to_halfraw(fp4_val, __NV_E2M1); float f = __half2float(*(__half*)&hr);",
        "tags": ["fp4", "dequantize", "convert", "half", "fp16", "e2m1"],
    },
    {
        "name": "__nv_cvt_fp4x2_to_halfraw2",
        "signature": "__half2_raw __nv_cvt_fp4x2_to_halfraw2(const __nv_fp4x2_storage_t x, const __nv_fp4_interpretation_t fp4_interpretation)",
        "description": "Convert two packed FP4 values to two half precision values.",
        "header": "cuda_fp4.h",
        "example": "__half2_raw hr2 = __nv_cvt_fp4x2_to_halfraw2(packed, __NV_E2M1);",
        "tags": ["fp4", "dequantize", "convert", "half", "fp16", "e2m1", "unpack"],
    },
    {
        "name": "__nv_cvt_bfloat16raw2_to_fp4x2",
        "signature": "__nv_fp4x2_storage_t __nv_cvt_bfloat16raw2_to_fp4x2(const __nv_bfloat162_raw x, const __nv_fp4_interpretation_t fp4_interpretation, const enum cudaRoundMode rounding)",
        "description": "Convert two bfloat16 values to two packed FP4 values. Direct bf16→fp4 without float intermediate. Note: Cast the result to (uint8_t) before storing to a uint8_t* array.",
        "header": "cuda_fp4.h",
        "example": "uint8_t packed = (uint8_t)__nv_cvt_bfloat16raw2_to_fp4x2(bf16_pair, __NV_E2M1, cudaRoundNearest);",
        "tags": ["fp4", "quantize", "convert", "bfloat16", "bf16", "e2m1", "pack"],
    },
    {
        "name": "__nv_cvt_halfraw2_to_fp4x2",
        "signature": "__nv_fp4x2_storage_t __nv_cvt_halfraw2_to_fp4x2(const __half2_raw x, const __nv_fp4_interpretation_t fp4_interpretation, const enum cudaRoundMode rounding)",
        "description": "Convert two half precision values to two packed FP4 values.",
        "header": "cuda_fp4.h",
        "example": "__nv_fp4x2_storage_t packed = __nv_cvt_halfraw2_to_fp4x2(h2_val, __NV_E2M1, cudaRoundNearest);",
        "tags": ["fp4", "quantize", "convert", "half", "fp16", "e2m1", "pack"],
    },
    {
        "name": "__nv_cvt_double2_to_fp4x2",
        "signature": "__nv_fp4x2_storage_t __nv_cvt_double2_to_fp4x2(const double2 x, const __nv_fp4_interpretation_t fp4_interpretation, const enum cudaRoundMode rounding)",
        "description": "Convert two doubles to two packed FP4 values.",
        "header": "cuda_fp4.h",
        "tags": ["fp4", "quantize", "convert", "double", "e2m1"],
    },

    # ── FP8 Conversion (cuda_fp8.h) ──────────────────────────────────────────
    {
        "name": "__nv_cvt_float_to_fp8",
        "signature": "__nv_fp8_storage_t __nv_cvt_float_to_fp8(const float x, const __nv_saturation_t saturate, const __nv_fp8_interpretation_t fp8_interpretation)",
        "description": "Convert float to FP8. Use __NV_E4M3 for e4m3 format, __NV_E5M2 for e5m2. Use __NV_SATFINITE for saturation.",
        "header": "cuda_fp8.h",
        "example": "__nv_fp8_storage_t fp8 = __nv_cvt_float_to_fp8(x, __NV_SATFINITE, __NV_E4M3);",
        "tags": ["fp8", "quantize", "convert", "float", "e4m3", "e5m2", "scale"],
    },
    {
        "name": "__nv_cvt_float2_to_fp8x2",
        "signature": "__nv_fp8x2_storage_t __nv_cvt_float2_to_fp8x2(const float2 x, const __nv_saturation_t saturate, const __nv_fp8_interpretation_t fp8_interpretation)",
        "description": "Convert two floats to two packed FP8 values.",
        "header": "cuda_fp8.h",
        "example": "__nv_fp8x2_storage_t fp8x2 = __nv_cvt_float2_to_fp8x2(make_float2(a, b), __NV_SATFINITE, __NV_E4M3);",
        "tags": ["fp8", "quantize", "convert", "float", "e4m3", "pack"],
    },
    {
        "name": "__nv_cvt_fp8_to_halfraw",
        "signature": "__half_raw __nv_cvt_fp8_to_halfraw(const __nv_fp8_storage_t x, const __nv_fp8_interpretation_t fp8_interpretation)",
        "description": "Convert FP8 to half precision. This is the correct way to convert fp8→float (go through half). There is NO __nv_cvt_fp8_to_float function.",
        "header": "cuda_fp8.h",
        "example": "__half_raw hr = __nv_cvt_fp8_to_halfraw(fp8_val, __NV_E4M3); float f = __half2float(*(__half*)&hr);",
        "tags": ["fp8", "dequantize", "convert", "half", "fp16", "e4m3", "to_float"],
    },
    {
        "name": "__nv_cvt_fp8x2_to_halfraw2",
        "signature": "__half2_raw __nv_cvt_fp8x2_to_halfraw2(const __nv_fp8x2_storage_t x, const __nv_fp8_interpretation_t fp8_interpretation)",
        "description": "Convert two packed FP8 values to two half precision values.",
        "header": "cuda_fp8.h",
        "example": "__half2_raw hr2 = __nv_cvt_fp8x2_to_halfraw2(fp8x2_val, __NV_E4M3);",
        "tags": ["fp8", "dequantize", "convert", "half", "fp16", "e4m3", "unpack"],
    },
    {
        "name": "__nv_fp8_e4m3",
        "signature": "__nv_fp8_e4m3(float val)  // constructor;  float(fp8_val)  // cast to float",
        "description": "FP8 E4M3 type with float constructor and float cast operator. Simplest way to do float↔fp8 conversion. There is NO __nv_cvt_fp8_to_float function — use float() cast instead.",
        "header": "cuda_fp8.h",
        "example": "__nv_fp8_e4m3 fp8 = __nv_fp8_e4m3(3.14f); float back = float(fp8);",
        "tags": ["fp8", "e4m3", "type", "convert", "float", "cast", "constructor"],
    },
    {
        "name": "__nv_cvt_bfloat16raw2_to_fp8x2",
        "signature": "__nv_fp8x2_storage_t __nv_cvt_bfloat16raw2_to_fp8x2(const __nv_bfloat162_raw x, const __nv_saturation_t saturate, const __nv_fp8_interpretation_t fp8_interpretation)",
        "description": "Convert two bfloat16 values to two packed FP8 values. Direct bf16→fp8.",
        "header": "cuda_fp8.h",
        "example": "__nv_fp8x2_storage_t fp8x2 = __nv_cvt_bfloat16raw2_to_fp8x2(bf16_pair, __NV_SATFINITE, __NV_E4M3);",
        "tags": ["fp8", "quantize", "convert", "bfloat16", "bf16", "e4m3"],
    },
    {
        "name": "__nv_cvt_float_to_e8m0",
        "signature": "__nv_fp8_storage_t __nv_cvt_float_to_e8m0(const float x, const __nv_saturation_t saturate, const enum cudaRoundMode rounding)",
        "description": "Convert float to E8M0 format (8-bit exponent only, no mantissa). Used for block scaling factors.",
        "header": "cuda_fp8.h",
        "tags": ["fp8", "e8m0", "scale", "exponent", "block_scale"],
    },
    {
        "name": "__nv_cvt_e8m0_to_bf16raw",
        "signature": "__nv_bfloat16_raw __nv_cvt_e8m0_to_bf16raw(const __nv_fp8_storage_t x)",
        "description": "Convert E8M0 value back to bfloat16.",
        "header": "cuda_fp8.h",
        "tags": ["fp8", "e8m0", "dequantize", "bfloat16", "bf16", "scale"],
    },
    {
        "name": "__nv_cvt_halfraw_to_fp8",
        "signature": "__nv_fp8_storage_t __nv_cvt_halfraw_to_fp8(const __half_raw x, const __nv_saturation_t saturate, const __nv_fp8_interpretation_t fp8_interpretation)",
        "description": "Convert half precision value to FP8.",
        "header": "cuda_fp8.h",
        "tags": ["fp8", "quantize", "convert", "half", "fp16", "e4m3"],
    },

    # ── Warp intrinsics ──────────────────────────────────────────────────────
    {
        "name": "__shfl_xor_sync",
        "signature": "T __shfl_xor_sync(unsigned mask, T var, int laneMask, int width=warpSize)",
        "description": "Exchange variable between threads using XOR of lane IDs. Used for parallel reductions. mask=0xffffffff for full warp.",
        "header": "<builtin>",
        "example": "float val = __shfl_xor_sync(0xffffffff, myval, 16); // swap with lane myLane^16",
        "tags": ["warp", "shuffle", "reduction", "sync", "xor"],
    },
    {
        "name": "__shfl_down_sync",
        "signature": "T __shfl_down_sync(unsigned mask, T var, unsigned delta, int width=warpSize)",
        "description": "Get value from lane with higher ID (lane + delta). Common for tree reductions.",
        "header": "<builtin>",
        "example": "for (int d=16; d>0; d>>=1) val += __shfl_down_sync(0xffffffff, val, d);",
        "tags": ["warp", "shuffle", "reduction", "sync", "down"],
    },
    {
        "name": "__redux_sync_add",
        "signature": "unsigned __redux_sync_add(unsigned mask, unsigned val)",
        "description": "Hardware single-instruction warp reduction (add). ~1 cycle vs __shfl_xor chain ~5 cycles. sm_80+. Only supports unsigned int.",
        "header": "<builtin>",
        "example": "unsigned sum = __redux_sync_add(0xffffffff, my_uint_val);",
        "tags": ["warp", "reduction", "sync", "add", "hardware", "fast"],
    },
    {
        "name": "__redux_sync_min",
        "signature": "unsigned __redux_sync_min(unsigned mask, unsigned val)",
        "description": "Hardware single-instruction warp reduction (min). sm_80+.",
        "header": "<builtin>",
        "tags": ["warp", "reduction", "sync", "min"],
    },
    {
        "name": "__redux_sync_max",
        "signature": "unsigned __redux_sync_max(unsigned mask, unsigned val)",
        "description": "Hardware single-instruction warp reduction (max). sm_80+.",
        "header": "<builtin>",
        "tags": ["warp", "reduction", "sync", "max"],
    },

    # ── Fast math (SFU) ──────────────────────────────────────────────────────
    {
        "name": "__expf",
        "signature": "float __expf(float x)",
        "description": "Fast exponential via Special Function Unit. ~4 cycles vs expf() ~20 cycles. Reduced precision.",
        "header": "<builtin>",
        "tags": ["math", "fast", "sfu", "exp", "silu"],
    },
    {
        "name": "__logf",
        "signature": "float __logf(float x)",
        "description": "Fast log via SFU. ~4 cycles.",
        "header": "<builtin>",
        "tags": ["math", "fast", "sfu", "log"],
    },
    {
        "name": "__frcp_rn",
        "signature": "float __frcp_rn(float x)",
        "description": "Fast reciprocal (1/x) with round-to-nearest. Hardware instruction.",
        "header": "<builtin>",
        "tags": ["math", "fast", "reciprocal", "division"],
    },
    {
        "name": "__frsqrt_rn",
        "signature": "float __frsqrt_rn(float x)",
        "description": "Fast reciprocal square root (1/sqrt(x)). Useful for RMSNorm. ~4 cycles.",
        "header": "<builtin>",
        "tags": ["math", "fast", "rsqrt", "rmsnorm", "normalization"],
    },
    {
        "name": "fmaf",
        "signature": "float fmaf(float x, float y, float z)",
        "description": "Fused multiply-add: x*y + z in one instruction, one rounding. Always use instead of separate mul+add.",
        "header": "<builtin>",
        "tags": ["math", "fma", "multiply", "add", "fused"],
    },
    {
        "name": "__fmaf_rn",
        "signature": "float __fmaf_rn(float x, float y, float z)",
        "description": "Fused multiply-add with round-to-nearest. Intrinsic version of fmaf.",
        "header": "<builtin>",
        "tags": ["math", "fma", "multiply", "add", "fused", "intrinsic"],
    },

    # ── Memory intrinsics ────────────────────────────────────────────────────
    {
        "name": "__ldg",
        "signature": "T __ldg(const T* ptr)",
        "description": "Load through read-only (texture) cache. Frees L1 for read-write data. Use for read-only inputs.",
        "header": "<builtin>",
        "example": "float val = __ldg(&input[idx]);",
        "tags": ["memory", "load", "cache", "readonly", "texture", "ldg"],
    },
    {
        "name": "__stcg",
        "signature": "void __stcg(T* ptr, T val)",
        "description": "Store with cache-global hint. Bypasses L1, goes direct to L2. Good for write-only outputs.",
        "header": "<builtin>",
        "tags": ["memory", "store", "cache", "streaming", "bypass"],
    },

    # ── Async copy ───────────────────────────────────────────────────────────
    {
        "name": "__pipeline_memcpy_async",
        "signature": "void __pipeline_memcpy_async(void* dst, const void* src, size_t count)",
        "description": "Async memory copy from global to shared memory. No register pressure. Use with __pipeline_commit() and __pipeline_wait_prior().",
        "header": "<cuda_pipeline.h>",
        "tags": ["memory", "async", "copy", "pipeline", "shared"],
    },

    # ── bfloat16 types ───────────────────────────────────────────────────────
    {
        "name": "__nv_bfloat162",
        "signature": "__nv_bfloat162 make_bfloat162(__nv_bfloat16 x, __nv_bfloat16 y)",
        "description": "Packed pair of bfloat16 values. Enables 2-wide operations. Load two bf16 values with one 32-bit load.",
        "header": "cuda_bf16.h",
        "example": "__nv_bfloat162 pair = *reinterpret_cast<__nv_bfloat162*>(&input[idx]); // load 2 bf16 at once",
        "tags": ["bfloat16", "bf16", "pair", "vectorize", "pack", "type"],
    },
    {
        "name": "__hadd2",
        "signature": "__half2 __hadd2(__half2 a, __half2 b)",
        "description": "Add two pairs of half precision values simultaneously. 2x throughput vs scalar.",
        "header": "cuda_fp16.h",
        "tags": ["half", "fp16", "add", "vectorize", "simd"],
    },
    {
        "name": "__hmul2",
        "signature": "__half2 __hmul2(__half2 a, __half2 b)",
        "description": "Multiply two pairs of half precision values simultaneously.",
        "header": "cuda_fp16.h",
        "tags": ["half", "fp16", "multiply", "vectorize", "simd"],
    },
    {
        "name": "__hfma2",
        "signature": "__half2 __hfma2(__half2 a, __half2 b, __half2 c)",
        "description": "Fused multiply-add on two pairs of half precision values. a*b + c.",
        "header": "cuda_fp16.h",
        "tags": ["half", "fp16", "fma", "vectorize", "simd"],
    },
    {
        "name": "__hadd2 (bfloat162)",
        "signature": "__nv_bfloat162 __hadd2(const __nv_bfloat162 a, const __nv_bfloat162 b)",
        "description": "Add two pairs of bfloat16 values simultaneously. 2x throughput vs scalar.",
        "header": "cuda_bf16.h",
        "tags": ["bfloat16", "bf16", "add", "vectorize", "simd"],
    },
    {
        "name": "__hmul2 (bfloat162)",
        "signature": "__nv_bfloat162 __hmul2(const __nv_bfloat162 a, const __nv_bfloat162 b)",
        "description": "Multiply two pairs of bfloat16 values simultaneously.",
        "header": "cuda_bf16.h",
        "tags": ["bfloat16", "bf16", "multiply", "vectorize", "simd"],
    },
    {
        "name": "__hfma2 (bfloat162)",
        "signature": "__nv_bfloat162 __hfma2(const __nv_bfloat162 a, const __nv_bfloat162 b, const __nv_bfloat162 c)",
        "description": "Fused multiply-add on two pairs of bfloat16 values. a*b + c.",
        "header": "cuda_bf16.h",
        "tags": ["bfloat16", "bf16", "fma", "vectorize", "simd"],
    },
]


def search_intrinsics(query: str, max_results: int = 8) -> str:
    """Search the CUDA intrinsics database using BM25-style fuzzy matching."""
    import re
    query_lower = query.lower()
    keywords = set(re.findall(r'\w+', query_lower))

    scored = []
    for entry in CUDA_INTRINSICS_DB:
        score = 0.0
        searchable_text = (
            entry["name"].lower() + " " +
            entry.get("description", "").lower() + " " +
            " ".join(entry.get("tags", []))
        )
        
        for kw in keywords:
            # Boost exact name matches massively
            if kw in entry["name"].lower():
                score += 15.0
            
            # Count term frequencies in the description/tags
            tf = searchable_text.count(kw)
            if tf > 0:
                # Diminishing returns for spamming the same word (BM25 principle)
                score += 3.0 * (tf / (tf + 0.5))
                
        if score > 0:
            scored.append((score, entry))

    scored.sort(key=lambda x: -x[0])
    results = scored[:max_results]

    if not results:
        return f"No results found for '{query}'. Try keywords like: fp4, fp8, e4m3, reduction, shuffle, fast math, bfloat16, fma, ldg, stcg, async"

    lines = []
    for _, entry in results:
        lines.append(f"## {entry['name']}")
        lines.append(f"Header: #include <{entry['header']}>")
        lines.append(f"```cpp\n{entry['signature']}\n```")
        lines.append(entry.get("description", ""))
        if "example" in entry:
            lines.append(f"Example: `{entry['example']}`")
        lines.append("")

    return "\n".join(lines)
