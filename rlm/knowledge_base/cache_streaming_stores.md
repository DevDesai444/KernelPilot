# Cache-Streaming Stores (bypassing L2 Cache on writes)

When a CUDA kernel is heavily memory-bound and writes large volumes of data (especially packed data like NVFP4 blocks or FP16 tensors), the default memory store instructions (`st.global`) will allocate lines in the L2 cache for the destination addresses before writing them to HBM.

If the written data is never read back by the same kernel (e.g., final output quantization blocks), this L2 cache allocation eviction policy simply thrashes the L2 cache. It forcefully evicts useful read-caching data (like weight vectors or input parameters) just to make room for write-only data.

## The Solution: `.cs` (Cache Streaming) modifier
You can bypass the L2 cache allocation entirely for pure write-outs by using inline PTX assembly with the `.cs` cache-streaming modifier. This instructs the hardware to stream the store directly through the memory subsystem without displacing necessary L1/L2 read caches.

### Example: Vectorized UInt4 Cache-Streaming Store

To maximize bandwidth, group your outputs into 128-bit (16-byte) packets. In C++, this is a `uint4`.

```cpp
// 1. Pack your output bytes or FP4 words into a 16-byte vector (uint4)
uint4 packed_data;
packed_data.x = ... // 4 bytes 
packed_data.y = ... // 4 bytes 
packed_data.z = ... // 4 bytes 
packed_data.w = ... // 4 bytes 

// 2. Perform the cache-streaming store via inline PTX assembly
// Address MUST be 16-byte aligned.
asm volatile(
    "st.global.cs.v4.u32 [%0], {%1, %2, %3, %4};"
    :
    : "l"(dest_ptr), 
      "r"(packed_data.x), 
      "r"(packed_data.y), 
      "r"(packed_data.z), 
      "r"(packed_data.w)
    : "memory"
);
```

### Constraints
* The destination pointer `[%0]` MUST be correctly typed as a 64-bit generic pointer (`"l"` constraint).
* The pointer must be properly aligned to the vector size (16-byte aligned for `.v4.u32`).
* You must include `"memory"` in the clobber list to prevent the compiler from reordering critical memory operations around the inline assembly.
