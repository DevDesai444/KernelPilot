# Shape-Specialized Logic and Full Loop Unrolling

## The Advantage over General Libraries
Production libraries (like FlashInfer or PyTorch kernels) must accept variables like `hidden_size`, `num_rows`, or `block_size` as generic function arguments, loading them into dynamic registers at runtime. This causes massive branch latency, kills register budgeting, and prevents loop unrolling.

Since you know the EXACT tensor shapes compiling on this node, you can dramatically exceed production library throughput by burning the dimensions straight into the C++ compiler.

## Implementation Guide

### 1. Hardcode dimensions as template parameters or `constexpr`
Instead of accepting `int hidden_size` dynamically, declare it globally. 

```cpp
template<int HIDDEN_SIZE_CONST, int ROWS_CONST>
__global__ void my_optimized_kernel(...) {
    // ...
}
```

### 2. Fully unroll memory fetch paths
Force the compiler to eliminate the `for` loop logic and instruction branching by using `#pragma unroll`.
*Note: `#pragma unroll` ONLY works if the loop bounds are compile-time constants.*

```cpp
// The compiler unravels this into N sequential fma/ldg instructions
// no jumping, zero branch divergence, maximal ILP.
#pragma unroll
for (int i = 0; i < HIDDEN_SIZE_CONST / 16; ++i) {
    // ...
}
```

### 3. Replace bounds checks with structural assumptions
If `HIDDEN_SIZE_CONST = 2048` and `BLOCK_SIZE = 256`, every thread reliably does exactly 8 loads (`2048 / 256 = 8`). You no longer need an `if (tid < hidden_size)` bounds check in your most critical loop. Delete them all!
