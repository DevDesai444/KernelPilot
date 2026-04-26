# RLM Kernel Optimizer

An autonomous CUDA kernel optimization system that uses LLM agents with beam search and RAG to generate faster GPU kernels. Given a reference CUDA kernel, the system discovers optimizations, generates improved code, and validates correctness — with no manual kernel analysis.


---

## Results

Benchmarked on NVIDIA B200 (sm_100a / Blackwell), compared against FlashInfer production baselines:

| Kernel | Shape | Speedup |
|--------|-------|---------|
| add_rmsnorm_fp4quant | 128×2048 | **1.691x** |
| add_rmsnorm_fp4quant | 128×4096 | **1.650x** |
| add_rmsnorm_fp4quant | 128×8192 | **1.420x** |
| nvfp4_quantize | 128×14336 | **1.845x** |
| silu_mul_fp4quant | 8×256×7168 | **1.450x** |

**Geometric mean: ~1.61x** over FlashInfer production baselines.

---

## How It Works

### Architecture

```
   Planner Agent  ←── RAG (Pinecone)
        │
        ▼
  Beam Search (width=4)
  ┌─────┬─────┬─────┬─────┐
  │ B1  │ B2  │ B3  │ B4  │   ← parallel branches
  └──┬──┴──┬──┴──┬──┴──┬──┘
     │     │     │     │
  Coder Agents (per branch)
     │
  nvcc compile
     │
  Benchmark (CUDA graph timing)
     │
  Correctness check (FlashInfer reference)
     │
  Profiler feedback → next iteration
     │
  Planner Tree (expand best branches)
     │
  Final best kernel
```

