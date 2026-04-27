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

### Key Components

| Module | Description |
|--------|-------------|
| `rlm/engine.py` | Multi-agent orchestration — planner, coder, fixer, reflector |
| `rlm/planner.py` | Generates optimization hypotheses from kernel source + RAG |
| `rlm/planner_spec.py` | Builds structured planner prompts with general analysis instructions |
| `rlm/coder.py` | LLM agent that writes optimized CUDA code |
| `rlm/fixer.py` | Repairs compile/correctness failures iteratively |
| `rlm/rag_retriever.py` | Pinecone-backed semantic search over real CUDA kernels |
| `search/beam_search.py` | Beam search with tree expansion, diversity selection, early stopping |
| `search/combiner.py` | Merges top-K candidate kernels into a combined variant |
| `profiler/kernel_profiler.py` | nvcc compilation + CUDA event benchmarking |
| `profiler/hybrid_profiler.py` | Occupancy, memory bandwidth, roofline analysis |
| `eval/correctness.py` | Numerical correctness validation against FlashInfer |
| `eval/flashinfer_ref.py` | Official FlashInfer baseline timing (CUDA graph methodology) |

### Agent Loop (per branch)

1. **Planner** reads the reference kernel source and RAG context, proposes an optimization hypothesis
2. **Coder** implements the hypothesis as complete CUDA code
3. **nvcc** compiles with `-arch=sm_100a -O3 --use_fast_math`
4. **Benchmark** times the kernel using CUDA graph replay (same methodology as FlashInfer baseline)
5. **Correctness** checks outputs against FlashInfer reference across 3 random seeds
6. **Profiler** reports occupancy, register count, spills, memory bandwidth
7. **Fixer** iterates if the kernel fails or regresses
8. If speedup > 1.0x → **Planner Tree** expands the branch into child variants

---

## Optimizations Discovered Autonomously

The agents discover these by analyzing the reference kernel source — no manual analysis provided:

- **Thread utilization**: Detecting loops where iteration bound < blockDim.x (idle threads)
- **Hardware FP4 intrinsics**: Replacing scalar 7-branch `float_to_nvfp4` with `__nv_cvt_float2_to_fp4x2` (one instruction per pair on sm_100a)
- **Vectorized loads**: Widening scalar bf16 loads to `uint4` (128-bit, 8 elements per transaction)
- **Register caching**: Eliminating Phase-2 global memory re-reads by caching values in registers across `__syncthreads`
- **Launch bounds**: `__launch_bounds__(N, M)` to cap register usage and maximize SM occupancy

---

## Beam Search

```yaml
beam:
  width: 4                      # parallel strategy variants
  refine_rounds: 4              # profiler feedback iterations
  tree_speedup_threshold: 1.0   # expand branches that beat baseline
  tree_branching_factor: 2      # child branches per above-baseline parent
  diversity_mode: family_diverse
  early_stop_min_improvement: 0.03
  population_crossover: true
```

- **Diversity selection** preserves distinct strategy families (vectorization, reduction, FP4, etc.)
- **Early stopping** halts when round-over-round improvement < 3%
- **Crossover** injects family-distinct candidates each round
- **Family retirement** prunes plateaued families that trail the leader

---

