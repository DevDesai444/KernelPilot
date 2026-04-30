# KernelPilot

Author: Dev Desai

KernelPilot is an autonomous CUDA kernel optimization system for WaferBench-style NVFP4 workloads on NVIDIA Blackwell hardware. It takes a reference kernel, studies the kernel and hardware context, proposes optimization directions with LLM agents, compiles candidate kernels, validates numerical correctness, profiles performance, and returns the best verified implementation.

The repository is centered on three reference workloads:

- `add_rmsnorm`: fused add + RMSNorm + NVFP4 quantization
- `nvfp4_quantize`: BF16 to NVFP4 block quantization
- `silu_mul`: fused SiLU and elementwise multiply with NVFP4 quantization

The project combines multi-agent planning, beam-search style exploration, hardware-aware profiling, correctness checks, and retrieval-augmented context from real CUDA codebases.

## What This Repository Does

KernelPilot is designed to answer a specific question: given a baseline CUDA kernel and a concrete target shape, can an automated system generate a faster version without breaking correctness?

At a high level, the system:

1. Loads a kernel task definition and problem shape.
2. Measures an official FlashInfer baseline when available.
3. Builds an optimization environment with hardware, search, and evaluation constraints.
4. Uses planner and coder agents to generate candidate kernels.
5. Compiles and benchmarks those candidates on GPU.
6. Rejects incorrect, slower, or suspicious implementations.
7. Iterates on promising candidates until the search budget is exhausted.
8. Emits the best validated kernel and a formatted submission artifact.

The emphasis is not just code generation. The repository includes the machinery needed to make automated optimization credible:

- benchmark parity against the production baseline
- correctness validation across seeded runs
- profiler-driven feedback
- search branching and pruning
- retrieval of external CUDA patterns through Pinecone-backed RAG

## Current Scope

The executable entry point in [run.py](/Users/DEVDESAI1/Desktop/University_at_Buffalo/Projects/KernelPilot/run.py) defines six task variants:

- `add_rmsnorm_fp4quant_b128xh2048`
- `add_rmsnorm_fp4quant_b128xh4096`
- `add_rmsnorm_fp4quant_b128xh8192`
- `nvfp4_quantize_m128xk14336`
- `silu_mul_fp4quant_b8xm256xk7168`
- `silu_mul_fp4quant_b8xm256xk14336`

Each task maps to a reference kernel source file under `kernels/reference/` and a specific shape tuple passed through the optimization pipeline.

## Architecture

KernelPilot is organized into a few major subsystems:

- `rlm/`: agent prompts, orchestration, planner logic, feedback generation, environment state, and RAG integration
- `search/`: beam search, diversity handling, candidate combination, and branch management
- `profiler/`: compilation and profiling helpers, including hybrid hardware analysis
- `eval/`: benchmarking, correctness validation, FlashInfer baseline measurement, and submission formatting
- `kernels/`: shared CUDA headers plus baseline reference kernels
- `config/`: model, beam, profiler, RAG, and evaluation settings

The control flow is roughly:

```text
run.py
  -> load config + environment
  -> measure official baseline
  -> initialize RLMEnvironment
  -> run BeamSearch
  -> validate best candidate
  -> benchmark final candidate
  -> format and save submission
```

Within search, the pipeline looks like:

```text
Planner -> Coder -> nvcc compile -> benchmark -> correctness -> profiler feedback -> refinement
```

## Core Modules

### Entry and task orchestration

- [run.py](/Users/DEVDESAI1/Desktop/University_at_Buffalo/Projects/KernelPilot/run.py): CLI entry point, task definitions, baseline handling, output generation, and top-level execution flow
- [rlm/environment.py](/Users/DEVDESAI1/Desktop/University_at_Buffalo/Projects/KernelPilot/rlm/environment.py): state container for each optimization session, including search configuration, metrics, counters, and persistent branch information
- [rlm/env_loader.py](/Users/DEVDESAI1/Desktop/University_at_Buffalo/Projects/KernelPilot/rlm/env_loader.py): project-level `.env` loading

### Agent and prompt system

- [rlm/engine.py](/Users/DEVDESAI1/Desktop/University_at_Buffalo/Projects/KernelPilot/rlm/engine.py): main orchestration logic, prompt construction, tool exposure, decomposition, refinement, and multi-turn agent execution
- [rlm/planner.py](/Users/DEVDESAI1/Desktop/University_at_Buffalo/Projects/KernelPilot/rlm/planner.py): branch planning and direction-setting logic
- [rlm/planner_spec.py](/Users/DEVDESAI1/Desktop/University_at_Buffalo/Projects/KernelPilot/rlm/planner_spec.py): structured planner instructions and plan schema support
- [rlm/root_prompts.py](/Users/DEVDESAI1/Desktop/University_at_Buffalo/Projects/KernelPilot/rlm/root_prompts.py): root-level prompt templates
- [rlm/sub_prompts.py](/Users/DEVDESAI1/Desktop/University_at_Buffalo/Projects/KernelPilot/rlm/sub_prompts.py): branch and sub-agent prompt builders
- [rlm/coder.py](/Users/DEVDESAI1/Desktop/University_at_Buffalo/Projects/KernelPilot/rlm/coder.py): code-generation prompt logic
- [rlm/fixer.py](/Users/DEVDESAI1/Desktop/University_at_Buffalo/Projects/KernelPilot/rlm/fixer.py): repair-oriented prompt construction
- [rlm/reflector.py](/Users/DEVDESAI1/Desktop/University_at_Buffalo/Projects/KernelPilot/rlm/reflector.py): refinement feedback shaping around measured outcomes
- [rlm/feedback.py](/Users/DEVDESAI1/Desktop/University_at_Buffalo/Projects/KernelPilot/rlm/feedback.py): targeted runtime-grounded feedback and experimental next-step framing

### Search and candidate management

- [search/beam_search.py](/Users/DEVDESAI1/Desktop/University_at_Buffalo/Projects/KernelPilot/search/beam_search.py): main search loop, branch expansion, candidate timing, pruning, and stopping rules
- [search/diversity_selector.py](/Users/DEVDESAI1/Desktop/University_at_Buffalo/Projects/KernelPilot/search/diversity_selector.py): family-aware diversity preservation
- [search/combiner.py](/Users/DEVDESAI1/Desktop/University_at_Buffalo/Projects/KernelPilot/search/combiner.py): top-candidate combination logic
- [search/strategy_bank.py](/Users/DEVDESAI1/Desktop/University_at_Buffalo/Projects/KernelPilot/search/strategy_bank.py): optimization strategy catalog

### Profiling and evaluation

- [profiler/kernel_profiler.py](/Users/DEVDESAI1/Desktop/University_at_Buffalo/Projects/KernelPilot/profiler/kernel_profiler.py): compilation and profiling orchestration
- [profiler/hybrid_profiler.py](/Users/DEVDESAI1/Desktop/University_at_Buffalo/Projects/KernelPilot/profiler/hybrid_profiler.py): hybrid analysis layer combining measured and inferred bottleneck signals
- [profiler/roofline.py](/Users/DEVDESAI1/Desktop/University_at_Buffalo/Projects/KernelPilot/profiler/roofline.py): B200 roofline support
- [profiler/metrics.py](/Users/DEVDESAI1/Desktop/University_at_Buffalo/Projects/KernelPilot/profiler/metrics.py): structured profiler metric definitions
- [profiler/validate_hybrid.py](/Users/DEVDESAI1/Desktop/University_at_Buffalo/Projects/KernelPilot/profiler/validate_hybrid.py): validation helper for hybrid profiling output
- [eval/benchmark.py](/Users/DEVDESAI1/Desktop/University_at_Buffalo/Projects/KernelPilot/eval/benchmark.py): timing harness and aggregate result helpers
- [eval/correctness.py](/Users/DEVDESAI1/Desktop/University_at_Buffalo/Projects/KernelPilot/eval/correctness.py): correctness checker and seeded numerical validation
- [eval/flashinfer_ref.py](/Users/DEVDESAI1/Desktop/University_at_Buffalo/Projects/KernelPilot/eval/flashinfer_ref.py): official baseline measurement path
- [eval/runtime_checks.py](/Users/DEVDESAI1/Desktop/University_at_Buffalo/Projects/KernelPilot/eval/runtime_checks.py): runtime safety and kernel-specific validation helpers
- [eval/hack_detector.py](/Users/DEVDESAI1/Desktop/University_at_Buffalo/Projects/KernelPilot/eval/hack_detector.py): trivial or invalid optimization detection
- [eval/waferbench_format.py](/Users/DEVDESAI1/Desktop/University_at_Buffalo/Projects/KernelPilot/eval/waferbench_format.py): final result formatting and output persistence

### RAG and external knowledge

- [rlm/rag_retriever.py](/Users/DEVDESAI1/Desktop/University_at_Buffalo/Projects/KernelPilot/rlm/rag_retriever.py): Pinecone retrieval, source filtering, ranking, and formatting
- [rlm/query_embedder.py](/Users/DEVDESAI1/Desktop/University_at_Buffalo/Projects/KernelPilot/rlm/query_embedder.py): embedding support for retrieval
- [rlm/cuda_docs.py](/Users/DEVDESAI1/Desktop/University_at_Buffalo/Projects/KernelPilot/rlm/cuda_docs.py): CUDA doc search support
- `rlm/knowledge_base/`: curated local notes on FP4 intrinsics, cache streaming, multi-row processing, and shape-specialized unroll patterns

## Reference Kernels and CUDA Support Code

The baseline kernels live in:

- [kernels/reference/add_rmsnorm.cu](/Users/DEVDESAI1/Desktop/University_at_Buffalo/Projects/KernelPilot/kernels/reference/add_rmsnorm.cu)
- [kernels/reference/nvfp4_quantize.cu](/Users/DEVDESAI1/Desktop/University_at_Buffalo/Projects/KernelPilot/kernels/reference/nvfp4_quantize.cu)
- [kernels/reference/silu_mul.cu](/Users/DEVDESAI1/Desktop/University_at_Buffalo/Projects/KernelPilot/kernels/reference/silu_mul.cu)

Shared CUDA helpers live in:

- [kernels/common/nvfp4_utils.cuh](/Users/DEVDESAI1/Desktop/University_at_Buffalo/Projects/KernelPilot/kernels/common/nvfp4_utils.cuh)
- [kernels/common/b200_intrinsics.cuh](/Users/DEVDESAI1/Desktop/University_at_Buffalo/Projects/KernelPilot/kernels/common/b200_intrinsics.cuh)

These files provide the primitive building blocks the agents are expected to exploit, including FP4 packing paths, Blackwell-specific intrinsics, and utility routines needed by the reference workloads.

## Search Configuration

The active default configuration in [config/search_config.yaml](/Users/DEVDESAI1/Desktop/University_at_Buffalo/Projects/KernelPilot/config/search_config.yaml) is intentionally conservative:

- beam width: `1`
- refinement rounds: `0`
- maximum concurrent API calls: `4`
- `combine_top_k`: `2`
- diversity mode: `family_diverse`
- tree branching factor: `2`
- tree speedup threshold: `1.0`
- profiler tool: `hybrid`
- profile workers: `2`
- target minimum speedup: `1.5x`

Model configuration is currently centered on `claude-haiku-4-5-20251001` for planner, coder, fixer, root, sub, and combine roles. That means the system is wired for cost-sensitive iteration by default, even though the rest of the codebase still supports a much richer multi-round search flow.

This is important context when reading benchmark numbers or expected behavior: the codebase supports a deeper search tree than the default runtime settings currently enable.

## RAG Pipeline

KernelPilot uses Pinecone-backed retrieval to supply examples and patterns from real CUDA sources to the planner.

Current RAG settings include:

- provider: `pinecone`
- index: `cuda-kernels-v2`
- top-k retrieval: `4`
- rerank pool: `25`
- text field: `source_code`
- source cap: `1`

The retrieval system is built to:

1. embed a query derived from kernel type, shape, and performance signals
2. search the configured Pinecone index
3. rerank matches using metadata and source quality
4. format the most relevant snippets for planner consumption
5. expose targeted documentation lookup during agent execution

Supporting tools and scripts:

- [check_pinecone.py](/Users/DEVDESAI1/Desktop/University_at_Buffalo/Projects/KernelPilot/check_pinecone.py): verifies index connectivity and configuration
- [migrate_pinecone.py](/Users/DEVDESAI1/Desktop/University_at_Buffalo/Projects/KernelPilot/migrate_pinecone.py): migration helper for index evolution
- [branches/add_rmsnorm_combined_fix.json](/Users/DEVDESAI1/Desktop/University_at_Buffalo/Projects/KernelPilot/branches/add_rmsnorm_combined_fix.json): branch data artifact used by one of the search workflows

## Evaluation Methodology

The repository makes a serious attempt to avoid inflated speedup claims.

Baseline measurement:

- tries to use an official FlashInfer implementation first
- falls back to the local reference kernel only when explicitly allowed
- records whether the baseline is official or unofficial

Candidate evaluation:

- compiles generated CUDA code
- benchmarks using the same measurement path used for the baseline
- runs correctness checks against expected outputs
- captures profiler and compiler evidence for the next refinement step

Important thresholds from configuration:

- correctness absolute tolerance: `1e-2`
- correctness relative tolerance: `1e-3`
- correctness seeds: `42`, `123`, `999`
- benchmark warmup iterations: `500`
- benchmark iterations: `100`

The code also includes GPU clock locking support in `run.py` to reduce variance when the environment allows it.

## Benchmark Results Recorded in the Repository

The existing project README history recorded the following headline results on NVIDIA B200:

| Kernel | Shape | Speedup |
| --- | --- | --- |
| add_rmsnorm_fp4quant | 128x2048 | 1.691x |
| add_rmsnorm_fp4quant | 128x4096 | 1.650x |
| add_rmsnorm_fp4quant | 128x8192 | 1.420x |
| nvfp4_quantize | 128x14336 | 1.845x |
| silu_mul_fp4quant | 8x256x7168 | 1.450x |

That yields an approximate geometric mean of `1.61x` over the reported FlashInfer baselines.

These figures are useful as repository context, but they should still be treated as run-dependent until reproduced in your target environment with the current configuration.

## Installation

### Python requirements

From [pyproject.toml](/Users/DEVDESAI1/Desktop/University_at_Buffalo/Projects/KernelPilot/pyproject.toml):

- Python `>=3.10`
- `anthropic`
- `pyyaml`
- `numpy`
- `pinecone`
- `sentence-transformers`
- `torch`
- `flashinfer`

The lightweight [requirements.txt](/Users/DEVDESAI1/Desktop/University_at_Buffalo/Projects/KernelPilot/requirements.txt) currently lists the non-heavy subset:

- `anthropic`
- `pyyaml`
- `numpy`
- `pinecone`
- `sentence-transformers`

Install with:

```bash
pip install -r requirements.txt
pip install torch flashinfer
```

If you want the package entry point:

```bash
pip install -e .
```

## Environment Variables

At minimum, expect to provide:

```bash
ANTHROPIC_API_KEY=...
PINECONE_API_KEY=...
PINECONE_INDEX_NAME=cuda-kernels-v2
PINECONE_INDEX_HOST=...
```

The configuration also supports environment overrides for:

- `PINECONE_NAMESPACE`
- `PINECONE_EMBED_PROVIDER`
- `PINECONE_EMBED_MODEL`
- `PINECONE_RERANK_MODEL`

## Usage

Optimize all tasks:

```bash
python run.py
```

Optimize a single task:

```bash
python run.py --kernel add_rmsnorm_fp4quant_b128xh2048
```

Validate setup without calling the model:

```bash
python run.py --dry-run
```

Override beam settings from the CLI:

```bash
python run.py --beam-width 2 --rounds 2
```

Allow a non-official fallback baseline for local experimentation:

```bash
python run.py --allow-reference-baseline
```

## Outputs

By default, the system writes submission artifacts under `outputs/submissions` and logs to `rlm_optimizer.log`.

The formatted submission metadata includes fields such as:

- chosen strategy
- bottleneck classification
- official-baseline flag
- search rounds used
- cumulative API cost
- elapsed time
- correctness status
- maximum absolute error
- rejection counts and attempt counters

This makes the outputs useful both as benchmark artifacts and as postmortem material for search behavior.

## Testing and Utility Scripts

The repository includes helper scripts and tests that are part of the development workflow:

- [test_llm_codegen.py](/Users/DEVDESAI1/Desktop/University_at_Buffalo/Projects/KernelPilot/test_llm_codegen.py): end-to-end or semi-structured codegen testing
- [test_llm_picks.py](/Users/DEVDESAI1/Desktop/University_at_Buffalo/Projects/KernelPilot/test_llm_picks.py): strategy-selection evaluation
- [scripts/coder_sandbox_check.py](/Users/DEVDESAI1/Desktop/University_at_Buffalo/Projects/KernelPilot/scripts/coder_sandbox_check.py): sandbox-oriented coder validation
- [scripts/agent_checks.sh](/Users/DEVDESAI1/Desktop/University_at_Buffalo/Projects/KernelPilot/scripts/agent_checks.sh): shell-level validation helpers

## Design Philosophy

A few themes show up consistently across the codebase:

- measure against a real baseline instead of trusting intuition
- prefer runtime-grounded feedback over generic optimization advice
- keep the planner informed by retrieval and profiler evidence
- separate planning, coding, fixing, and reflection responsibilities
- preserve diversity in candidate search rather than overcommitting to one idea too early
- treat correctness as a hard gate, not a secondary concern

This makes KernelPilot more than a prompt wrapper around `nvcc`. It is an attempt to build a disciplined, inspectable optimization loop for CUDA kernels under realistic benchmark pressure.

## Limitations

KernelPilot is powerful, but it is not plug-and-play on arbitrary machines.

Key constraints:

- it assumes NVIDIA GPU access for meaningful execution
- many paths are Blackwell and NVFP4 specific
- FlashInfer availability changes whether results are considered official
- the best results depend on external services such as Anthropic and Pinecone
- the default config currently runs a narrower search than the full architecture supports

If you are reproducing results, pay close attention to GPU model, driver/toolchain compatibility, FlashInfer installation, and current search settings.

## Recommended First Read

If you are onboarding to the repository, read in this order:

1. [run.py](/Users/DEVDESAI1/Desktop/University_at_Buffalo/Projects/KernelPilot/run.py)
2. [config/search_config.yaml](/Users/DEVDESAI1/Desktop/University_at_Buffalo/Projects/KernelPilot/config/search_config.yaml)
3. [search/beam_search.py](/Users/DEVDESAI1/Desktop/University_at_Buffalo/Projects/KernelPilot/search/beam_search.py)
4. [rlm/engine.py](/Users/DEVDESAI1/Desktop/University_at_Buffalo/Projects/KernelPilot/rlm/engine.py)
5. [profiler/hybrid_profiler.py](/Users/DEVDESAI1/Desktop/University_at_Buffalo/Projects/KernelPilot/profiler/hybrid_profiler.py)
6. [eval/correctness.py](/Users/DEVDESAI1/Desktop/University_at_Buffalo/Projects/KernelPilot/eval/correctness.py)

That sequence gives the clearest picture of how tasks are defined, how search proceeds, how candidates are judged, and how results are finalized.
