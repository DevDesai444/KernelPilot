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
