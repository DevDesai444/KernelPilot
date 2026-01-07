// b200_intrinsics.cuh — Blackwell-specific TMA, TMEM, and pipeline wrappers
// Requires CUDA 12.6+ and sm_100a or later

#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cuda/pipeline>
#include <stdint.h>

namespace cg = cooperative_groups;

// ────────────────────────────────────────────────────────────────────────────
// TMA (Tensor Memory Accelerator) — async bulk copy helpers
// These wrap cp.async.bulk / tcgen05.ld PTX for B200
// ────────────────────────────────────────────────────────────────────────────

// Descriptor for a 1D TMA copy
struct TMADescriptor1D {
    uint64_t  tensor_map;     // filled by cuTensorMapEncode*
    uint32_t  box_size;       // elements in box (copy width)
    uint32_t  element_stride; // bytes between elements (usually sizeof(T))
};

// Issue async TMA copy: global → shared memory
// Requires mbarrier for completion signaling
__device__ __forceinline__ void tma_load_1d(
    void* __restrict__ smem_dst,
    const void* __restrict__ gmem_src,
    uint64_t* __restrict__ mbar,
    uint32_t num_bytes)
{
    // Use cp.async.bulk for contiguous transfers
    asm volatile (
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes"
        " [%0], [%1], %2, [%3];"
        :
        : "r"((uint32_t)__cvta_generic_to_shared(smem_dst)),
          "l"((uint64_t)gmem_src),
          "r"(num_bytes),
          "r"((uint32_t)__cvta_generic_to_shared(mbar))
        : "memory"
    );
}

// Initialize mbarrier for TMA completion tracking
__device__ __forceinline__ void mbar_init(uint64_t* mbar, uint32_t num_transactions) {
    asm volatile (
        "mbarrier.init.shared::cta.b64 [%0], %1;"
        : : "r"((uint32_t)__cvta_generic_to_shared(mbar)), "r"(num_transactions)
        : "memory"
    );
}

// Wait for mbarrier (TMA completion)
__device__ __forceinline__ void mbar_wait(uint64_t* mbar, uint32_t phase) {
    asm volatile (
        "{\n\t"
        ".reg .pred p;\n\t"
        "WAIT:\n\t"
        "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n\t"
        "@!p bra WAIT;\n\t"
        "}"
        : : "r"((uint32_t)__cvta_generic_to_shared(mbar)), "r"(phase)
        : "memory"
    );
}

// Arrive at mbarrier (for thread-block synchronization)
__device__ __forceinline__ void mbar_arrive(uint64_t* mbar) {
    asm volatile (
        "mbarrier.arrive.shared::cta.b64 _, [%0];"
        : : "r"((uint32_t)__cvta_generic_to_shared(mbar))
        : "memory"
    );
}

// ────────────────────────────────────────────────────────────────────────────
// Software pipeline helper — double-buffered shared memory prefetch
// Usage:
//   PipelineState pipe;
//   pipe.issue_prefetch(stage0_buf, gmem_ptr, nbytes, mbar);
//   pipe.wait(mbar);
//   ... consume stage0_buf ...
//   pipe.issue_prefetch(stage1_buf, next_ptr, nbytes, mbar);
//   pipe.swap();
// ────────────────────────────────────────────────────────────────────────────
struct PipelineState {
    uint32_t phase = 0;
    uint32_t stage = 0;

    __device__ __forceinline__ void issue_prefetch(
        void* smem_buf,
        const void* gmem_src,
        uint32_t num_bytes,
        uint64_t* mbar)
    {
        mbar_init(mbar, 1);
        if (threadIdx.x == 0) {
            tma_load_1d(smem_buf, gmem_src, mbar, num_bytes);
        }
    }

    __device__ __forceinline__ void wait(uint64_t* mbar) {
        mbar_wait(mbar, phase);
    }

    __device__ __forceinline__ void swap() {
        stage ^= 1;
        phase ^= 1;
    }
};

// ────────────────────────────────────────────────────────────────────────────
// Vectorized global memory loads — 128-bit aligned
// ────────────────────────────────────────────────────────────────────────────

__device__ __forceinline__ float4 load_float4(const float* ptr) {
    return *reinterpret_cast<const float4*>(ptr);
}

__device__ __forceinline__ void store_float4(float* ptr, float4 val) {
    *reinterpret_cast<float4*>(ptr) = val;
}

__device__ __forceinline__ uint4 load_uint4(const uint32_t* ptr) {
    return *reinterpret_cast<const uint4*>(ptr);
}

// Load 16 bytes (128 bits) as __nv_bfloat162 x4
__device__ __forceinline__ void load_bf16x8(
    const __nv_bfloat16* ptr,
    __nv_bfloat162& a, __nv_bfloat162& b,
    __nv_bfloat162& c, __nv_bfloat162& d)
{
    uint4 raw = load_uint4(reinterpret_cast<const uint32_t*>(ptr));
    a = *reinterpret_cast<__nv_bfloat162*>(&raw.x);
    b = *reinterpret_cast<__nv_bfloat162*>(&raw.y);
    c = *reinterpret_cast<__nv_bfloat162*>(&raw.z);
    d = *reinterpret_cast<__nv_bfloat162*>(&raw.w);
}

// ────────────────────────────────────────────────────────────────────────────
// TMEM access helpers (Blackwell tensor memory — on-chip scratchpad)
// TMEM sits between L1 and registers; ~5 cycle latency, 512KB per SM
// ────────────────────────────────────────────────────────────────────────────

// Allocate TMEM region (must be called by elected thread in block)
__device__ __forceinline__ uint32_t tmem_alloc(uint32_t num_bytes) {
    uint32_t tmem_addr;
    asm volatile (
        "tcgen05.alloc.cta_group::1.sync.aligned.b32 %0, %1;"
        : "=r"(tmem_addr)
        : "r"(num_bytes)
    );
    return tmem_addr;
}

// Free TMEM region
__device__ __forceinline__ void tmem_free(uint32_t tmem_addr) {
    asm volatile (
        "tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0;"
        : : "r"(tmem_addr)
    );
}

// Load from TMEM into register
__device__ __forceinline__ uint32_t tmem_load(uint32_t tmem_addr) {
    uint32_t val;
    asm volatile (
        "tcgen05.ld.sync.aligned.32b.x1.b32 %0, [%1];"
        : "=r"(val) : "r"(tmem_addr)
    );
    return val;
}

// Store register to TMEM
__device__ __forceinline__ void tmem_store(uint32_t tmem_addr, uint32_t val) {
    asm volatile (
        "tcgen05.st.sync.aligned.32b.x1.b32 [%0], %1;"
        : : "r"(tmem_addr), "r"(val)
    );
}

// ────────────────────────────────────────────────────────────────────────────
// Utility: elect one thread in a warp for TMA issue
// ────────────────────────────────────────────────────────────────────────────
__device__ __forceinline__ bool is_warp_leader() {
    return (threadIdx.x % 32) == 0;
}

__device__ __forceinline__ bool is_block_leader() {
    return threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0;
}

// ────────────────────────────────────────────────────────────────────────────
// Cache control
// ────────────────────────────────────────────────────────────────────────────

// Evict from L1 (streaming load — don't pollute L1 cache)
__device__ __forceinline__ float load_streaming(const float* ptr) {
    float val;
    asm volatile ("ld.global.cs.f32 %0, [%1];" : "=f"(val) : "l"(ptr));
    return val;
}

// L2 prefetch hint
__device__ __forceinline__ void prefetch_l2(const void* ptr) {
    asm volatile ("prefetch.global.L2 [%0];" : : "l"(ptr));
}
