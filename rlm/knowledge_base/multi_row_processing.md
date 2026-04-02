# Multi-Row Processing (Amortizing Shared Weights)

When processing data where each thread row shares a common read vector (e.g., RMSNorm reading a scale weight vector, or LayerNorm reading beta/gamma), processing one row per thread block can severely restrict cache performance. 

If multiple thread blocks are launched and each processes one row, every block independently requests the shared weight vector from the L2 cache or HBM. If the weight vector is large (e.g., 2048 or 8192 floats), retrieving it repeatedly per block thrashes the L1 cache bandwidth.

## The Solution: Multi-Row per Block Mapping
To reduce redundant memory reads across thread blocks, configure the grid launch so that a single block manages **multiple rows simultaneously**. 

Instead of `blockIdx.x` pointing merely to one output row, have every block index represent a *group* of rows, and assign subsets of the warps inside the block to each individual row.

### Example Mapping

Assume a kernel processes `hidden_size` elements across `N` rows, applying a shared `weight_vector` to every row.
If a thread block is launched with 128 threads (4 warps) and your goal is to process 4 rows per block:

```cpp
#define ROWS_PER_BLOCK 4
#define WARP_SIZE 32

__global__ void my_multiro_kernel(const float* input, const float* weight_vector, float* output) {
    // blockIdx.x defines the starting row for the current block
    int base_row = blockIdx.x * ROWS_PER_BLOCK;
    
    // Divide work among warps: each warp handles ONE row
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    
    int my_row = base_row + warp_id;
    if (my_row >= TOTAL_ROWS) return;

    // By structuring it this way, when lane_id fetches the weight for its element, 
    // the L1 hardware caches it temporarily, making it immediately available to the 
    // adjacent warps processing the parallel rows, reducing separate L2 fetch cycles.
    int element_offset = lane_id; // (plus loops advancing by 32)
    
    // ... Process input[my_row * row_stride + element_offset]
    // ... Read weight_vector[element_offset]
}
```

This structural transformation directly increases SM register lifespan per L1 fetch by exactly `ROWS_PER_BLOCK` multiples. When combined with fully fused 1-pass kernels, it mitigates hardware bottlenecks for shared-vector memory loads.
