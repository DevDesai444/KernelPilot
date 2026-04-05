"""
engine.py — RLM core engine.
Orchestrates root LLM decomposition, parallel sub-LLM beam generation, and refinement.
"""

from __future__ import annotations
import asyncio
import json
import logging
import re
from pathlib import Path

import anthropic
from anthropic import AsyncAnthropic

from .coder import build_coder_prompt
from .environment import RLMEnvironment, KernelCandidate
from .feedback import build_sandbox_feedback
from .feedback import _kernel_aliases, _kernel_operation_phrase
from .fixer import build_fixer_prompt
from .planner import (
    build_initial_plan_prompt,
    build_tree_plan_prompt,
    fallback_branches,
    parse_plan_response,
)
from .rag_retriever import init_knowledge_base
from .root_prompts import SYSTEM_PROMPT, combine_prompt
from .reflector import (
    _get_launch_signature,
    _format_profile_section,
)

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
ALLOWED_READ_ROOTS = (
    (PROJECT_ROOT / "kernels" / "common").resolve(),
    (PROJECT_ROOT / "kernels" / "reference").resolve(),
)


# ── Refinement: tool-use agent loop ────────────────────────────────────────

MAX_INNER_TURNS = 5

SUBMIT_KERNEL_TOOL = {
    "name": "submit_kernel",
    "description": (
        "Submit optimized CUDA kernel for compilation, correctness checking, "
        "and profiling.\n\n"
        "Returns one of:\n"
        "- COMPILE ERROR: first error with file:line plus surrounding context\n"
        "- CORRECTNESS FAILURE: max error magnitude and which check failed\n"
        "- Result verdict (IMPROVED / REGRESSION / NO CHANGE) with:\n"
        "  timing_us, speedup vs baseline, SM occupancy,\n"
        "  compiler resource usage, register spills,\n"
        "  delta from your previous submission,\n"
        "  remaining optimization suggestions from profiler data"
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "cuda_code": {
                "type": "string",
                "description": (
                    "Complete .cu file content with all #includes, "
                    "kernel functions, and the launch_* wrapper."
                ),
            }
        },
        "required": ["cuda_code"],
    },
}

READ_FILE_TOOL = {
    "name": "read_file",
    "description": (
        "Read a source file from the project. Available files:\n"
        "- kernels/common/nvfp4_utils.cuh — FP4/FP8 quantization helpers, pack/unpack\n"
        "- kernels/common/b200_intrinsics.cuh — Blackwell TMA, TMEM, pipeline wrappers\n"
        "- kernels/reference/add_rmsnorm.cu — Naive Add+RMSNorm+FP4 reference kernel\n"
        "- kernels/reference/silu_mul.cu — Naive SiLU*Mul+FP4 reference kernel\n"
        "- kernels/reference/nvfp4_quantize.cu — Naive BF16→FP4 reference kernel\n"
        "Costs no submit_kernel turn."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Relative path from project root (e.g. 'kernels/common/nvfp4_utils.cuh')",
            }
        },
        "required": ["path"],
    },
}

SEARCH_DOCS_TOOL = {
    "name": "search_docs",
    "description": (
        "Search CUDA intrinsic documentation. Query by keyword to find correct "
        "function signatures, headers, and usage examples. Covers: FP4/FP8 conversion "
        "(cuda_fp4.h, cuda_fp8.h), warp intrinsics (shuffle, reduction), fast math "
        "(SFU), memory intrinsics (ldg, stcg, async copy), bfloat16/half operations.\n"
        "Example queries: 'fp4 convert float', 'fp8 e4m3 to float', 'warp reduction', "
        "'fast reciprocal sqrt', 'bfloat16 pair load'\n"
        "Costs no submit_kernel turn."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search keywords (e.g. 'fp4 quantize float', 'e4m3 convert', 'warp reduce')",
            }
        },
        "required": ["query"],
    },
}

SEARCH_PINECONE_TOOL = {
    "name": "search_pinecone",
    "description": (
        "Search the Pinecone knowledge index for CUDA optimization notes, prior "
        "experiments, compiler pitfalls, and kernel-specific guidance that the user "
        "already stored there. Use this when the local docs are not enough. "
        "Costs no submit_kernel turn."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Semantic search query to run against Pinecone.",
            },
            "top_k": {
                "type": "integer",
                "description": "Optional number of matches to return.",
            },
        },
        "required": ["query"],
    },
}

REFINE_TOOLS = [
    SUBMIT_KERNEL_TOOL,
    READ_FILE_TOOL,
    SEARCH_DOCS_TOOL,
]

REFINE_SYSTEM_PROMPT = f"""\
You are a CUDA kernel optimization agent. You have {{turns}} submit_kernel calls.

Your speedup is measured against FlashInfer, a production GPU library.
You target a single GPU (B200, sm_100a) and a single problem shape — use this to your advantage.

Available tools (only submit_kernel counts toward your turn limit):
- submit_kernel: compile, test correctness, and benchmark your kernel
- read_file: read project header files (nvfp4_utils.cuh, b200_intrinsics.cuh) or reference kernels
- search_docs: look up CUDA intrinsic signatures and usage (fp4, fp8, warp, fast math, memory)

Target hardware — NVIDIA B200 (sm_100a, Blackwell):
- HBM3e: 8 TB/s bandwidth, 192 GB
- L2 cache: 126 MB — benchmark uses L2 cache cycling (data is COLD every iteration)
- 148 SMs, 228 KB shared memory per SM, 255 registers per thread
- 128-bit load/store = uint4 = 8 bf16 values per transaction
- Use read_file to check available hardware intrinsics in the project headers

Before EVERY submit_kernel call, explain in 2-3 sentences:
1. What the latest measured result confirms or leaves uncertain
2. What specific code change you will make and why you expect it to help

Rules:
{{constraint}}
- NEVER put __syncthreads() inside if/else branches (deadlock).
- The launch_* function signature must match exactly.
- Output must match reference within atol=1e-2.
- Keep original #include directives, not expanded headers.
- Do not use torch headers (torch/extension.h, ATen, c10).
"""

def _build_refine_system_prompt(
    speedup: float,
    prev_metrics: dict = None,
    kernel_type: str = "",
    problem_shape: tuple | None = None,
) -> str:
    """Build REFINE_SYSTEM_PROMPT with constraints from measured runtime data."""
    if speedup >= 1.0 and prev_metrics:
        cm = prev_metrics.get("_compiler", {})
        occupancy = float(prev_metrics.get("sm_occupancy", 0) or 0.0)
        hints = []
        if cm.get("spill_stores_bytes", 0) or cm.get("spill_loads_bytes", 0):
            hints.append("spills are present, so reduce live state before adding more work per thread")
        regs = int(cm.get("registers_per_thread", 0) or 0)
        reg_limit = 96
        occ_limit = 75.0
        if kernel_type == "add_rmsnorm" and tuple(problem_shape or ()) == (128, 2048):
            reg_limit = 44   # baseline is 40; flag if candidate exceeds baseline+4
            occ_limit = 74.0  # flag if occupancy drops below current baseline of 75%
        elif kernel_type == "add_rmsnorm":
            reg_limit = 44
            occ_limit = 74.0
        if regs >= reg_limit or occupancy < occ_limit:
            hints.append("register pressure or occupancy is already tight")

        constraint = f"- You are ABOVE baseline ({speedup:.2f}x). Prefer surgical follow-up changes over structural rewrites."
        if hints:
            constraint += "\n- Measured constraints: " + "; ".join(hints) + "."
    else:
        constraint = "- Structural changes, algorithmic rewrites, and surgical optimizations are all allowed."

    return REFINE_SYSTEM_PROMPT.replace("{{turns}}", str(MAX_INNER_TURNS)).replace("{{constraint}}", constraint)


class RLMEngine:
    """
    Main orchestrator for the RLM beam search loop.
    Handles: decomposition → beam generation → profiler-guided refinement → combination.
    """

    def __init__(self, env: RLMEnvironment):
        self.env = env
        cfg = env.search_config
        self.beam_width    = cfg["beam"]["width"]
        self.refine_rounds = cfg["beam"]["refine_rounds"]
        # Sync client for root/combine calls (sequential); async client for parallel beams
        self.client       = anthropic.Anthropic()
        self.async_client = AsyncAnthropic(max_retries=10)
        max_concurrent_api_calls = int(
            cfg["beam"].get("max_concurrent_api_calls", max(2, min(self.beam_width, 4)))
        )
        # Limit concurrent API calls to avoid 429s without artificially serializing the beam.
        self._api_semaphore = asyncio.Semaphore(max(1, max_concurrent_api_calls))
        self._loop = None  # persistent event loop for async calls

        self.root_model    = cfg["models"].get("planner_model", cfg["models"]["root_model"])
        self.sub_model     = cfg["models"].get("coder_model", cfg["models"]["sub_model"])
        self.fixer_model   = cfg["models"].get("fixer_model", cfg["models"]["sub_model"])
        self.combine_model = cfg["models"]["combine_model"]
        self.combine_top_k = cfg["beam"]["combine_top_k"]
        self.tree_speedup_threshold = float(cfg["beam"].get("tree_speedup_threshold", 1.0))
        self.tree_branching_factor = int(cfg["beam"].get("tree_branching_factor", 2))
        self.max_tokens    = cfg["cost_control"]["max_tokens_per_sub_call"]
        self.rag = init_knowledge_base(cfg.get("rag", {}))

    # ── Low-level LLM call ────────────────────────────────────────────────────

    def _call_llm(
        self,
        prompt: str,
        model: str,
        system: str = SYSTEM_PROMPT,
        temperature: float = 0.3,
    ) -> tuple:
        if self.env.over_budget():
            raise RuntimeError(
                f"Budget exhausted: ${self.env.total_api_cost_usd:.4f} spent"
            )

        response = self.client.messages.create(
            model=model,
            max_tokens=self.max_tokens,
            temperature=temperature,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )

        text       = response.content[0].text
        tokens_in  = response.usage.input_tokens
        tokens_out = response.usage.output_tokens
        cost = self.env.record_api_cost(tokens_in, tokens_out, model)
        logger.info(
            "LLM call: model=%s in=%d out=%d cost=$%.4f",
            model, tokens_in, tokens_out, cost,
        )
        return text, tokens_in, tokens_out

    async def _call_llm_async(
        self,
        prompt: str,
        model: str,
        system: str = SYSTEM_PROMPT,
        temperature: float = 0.3,
    ) -> tuple:
        """True async API call — all 4 beam coroutines run concurrently."""
        if self.env.over_budget():
            raise RuntimeError(
                f"Budget exhausted: ${self.env.total_api_cost_usd:.4f} spent"
            )

        async with self._api_semaphore:
            response = await self.async_client.messages.create(
                model=model,
                max_tokens=self.max_tokens,
                temperature=temperature,
                system=system,
                messages=[{"role": "user", "content": prompt}],
            )

        text       = response.content[0].text
        tokens_in  = response.usage.input_tokens
        tokens_out = response.usage.output_tokens
        cost = self.env.record_api_cost(tokens_in, tokens_out, model)
        logger.info(
            "LLM call (async): model=%s in=%d out=%d cost=$%.4f",
            model, tokens_in, tokens_out, cost,
        )
        return text, tokens_in, tokens_out

    async def _call_llm_with_tools_async(
        self,
        messages: list,
        tools: list,
        model: str,
        system: str = SYSTEM_PROMPT,
        temperature: float = 0.4,
    ):
        """Async API call with tool use support. Returns full response object."""
        if self.env.over_budget():
            raise RuntimeError(
                f"Budget exhausted: ${self.env.total_api_cost_usd:.4f} spent"
            )

        async with self._api_semaphore:
            response = await self.async_client.messages.create(
                model=model,
                max_tokens=self.max_tokens,
                temperature=temperature,
                system=system,
                messages=messages,
                tools=tools,
            )

        tokens_in  = response.usage.input_tokens
        tokens_out = response.usage.output_tokens
        cost = self.env.record_api_cost(tokens_in, tokens_out, model)
        logger.info(
            "LLM tool call: model=%s in=%d out=%d cost=$%.4f stop=%s",
            model, tokens_in, tokens_out, cost, response.stop_reason,
        )
        return response

    def _planner_baseline_context(self) -> str:
        env = self.env
        ops = env.count_memory_ops()
        missing = env.detect_missing_optimizations()
        analysis = (
            f"  Source signals: loads={ops['loads']} stores={ops['stores']} "
            f"float4={ops['float4']} tma={ops['tma']} "
            f"syncthreads~={ops['syncthreads']} (src={ops['syncthreads_source']}) "
            f"shfl={ops['shfl']}\n"
            f"  Preferred search surfaces: {', '.join(missing[:5]) if missing else 'none singled out from source analysis'}\n"
        )
        if env.baseline_naive_us and env.baseline_us_reported:
            rows = env.problem_shapes[0][0]
            sm_count = env.hw_spec.get("sm", {}).get("count", 148)
            cm = env.baseline_compiler_metrics
            cm_str = cm.summary_str() if cm else "unavailable"
            baseline_label = "FlashInfer timing" if env.official_baseline else "Reference fallback timing (UNOFFICIAL)"
            return (
                f"BASELINE PROFILER DATA (reference kernel):\n"
                f"  Naive kernel timing: {env.baseline_naive_us:.3f} us\n"
                f"  {baseline_label}:   {env.baseline_us_reported:.3f} us\n"
                f"  Compiler: {cm_str}\n"
                f"  Grid: {rows} blocks launched on {sm_count} SMs"
                f"{' — some SMs get zero work' if rows < sm_count else ''}\n"
                f"{analysis}"
            )
        return (
            "BASELINE PROFILER DATA: unavailable — analyze kernel source and retrieved production patterns.\n"
            f"{analysis}"
        )

    def _search_pinecone_context(self, queries: list[str], top_k: int | None = None) -> str:
        clean_queries = [q.strip() for q in queries if q and q.strip()]
        if not clean_queries:
            return "No Pinecone query provided."
        effective_top_k = int(top_k or getattr(self.rag, "top_k", 4))
        # For add_rmsnorm, exclude flashinfer (circular — it IS the baseline) and sakana
        # (synthetic competition data, not production CUDA). Forces vllm/sglang/apex/cutlass.
        exclude = ["flashinfer", "sakana"] if getattr(self.env, "kernel_type", "") == "add_rmsnorm" else []
        matches = self.rag.search_many(clean_queries[:4], top_k=effective_top_k,
                                       exclude_source_patterns=exclude or None)
        return self.rag.format_matches(matches)

    def _log_planner_block(self, title: str, content: str) -> None:
        text = content.strip() if content else "(empty)"
        logger.info("\n%s\n%s\n%s", "=" * 80, title, "=" * 80)
        logger.info("%s", text)

    def _initial_plan_queries(self) -> list[str]:
        env = self.env
        shape = "x".join(str(dim) for dim in env.problem_shapes[0])
        operation = _kernel_operation_phrase(env.kernel_type)
        aliases = ", ".join(_kernel_aliases(env.kernel_type)[:4])
        queries = [
            (
                f"Operation: {operation}. "
                f"Shape: {shape}. Aliases: {aliases}. Need: production CUDA kernel source_code."
            ),
            f"{operation} FlashInfer production kernel source code",
            f"{operation} best production kernel B200 bf16 fp4 source code",
            f"{operation} vectorized loads stores bf16 fp4 CUDA source code",
        ]
        for hint in env.detect_missing_optimizations()[:2]:
            queries.append(f"{operation} {hint.replace('_', ' ')} CUDA source code")
        return queries

    def _expand_tree_plans(self, parent: KernelCandidate, branch_count: int | None = None) -> list[dict]:
        branch_count = int(branch_count or self.tree_branching_factor)
        planner_queries = (
            parent.plan_branch.get("rag_queries")
            or [f"{self.env.kernel_type} {parent.strategy} next optimization"]
        )
        rag_context = self._search_pinecone_context(planner_queries)
        feedback = build_sandbox_feedback(
            {
                "compile_ok": parent.compile_ok,
                "correct": parent.correct,
                "speedup": parent.speedup,
                "metrics": parent.metrics,
                "error": parent.compile_error,
            },
            parent_speedup=parent.speedup,
            prev_inner_metrics=parent.prev_metrics,
            kernel_type=self.env.kernel_type,
            candidate=parent,
        )
        self._log_planner_block(
            f"PLANNER RAG CONTEXT [tree parent={parent.strategy}] queries={planner_queries}",
            rag_context,
        )
        prompt = build_tree_plan_prompt(
            kernel_type=self.env.kernel_type,
            operation=_kernel_operation_phrase(self.env.kernel_type),
            aliases=_kernel_aliases(self.env.kernel_type),
            problem_shape=self.env.problem_shapes[0],
            parent_strategy=parent.strategy,
            parent_speedup=parent.speedup,
            kernel_src=parent.best_code or parent.code,
            feedback_summary=feedback.planner_summary(),
            rag_context=rag_context,
            branch_count=branch_count,
        )
        response, _, _ = self._call_llm(prompt, model=self.root_model, temperature=0.2)
        self._log_planner_block(
            f"PLANNER RAW OUTPUT [tree parent={parent.strategy}]",
            response,
        )
        branches = parse_plan_response(
            response,
            count=branch_count,
            prefix=f"{parent.strategy}_child",
            parent_strategy=parent.strategy,
        )
        self._log_planner_block(
            f"PLANNER PARSED BRANCHES [tree parent={parent.strategy}]",
            json.dumps(branches, indent=2, sort_keys=True),
        )
        return branches

    # ── Round 0: Decomposition ────────────────────────────────────────────────

    def decompose(self) -> list:
        env = self.env
        num_strategies = self.beam_width * 2
        planner_queries = self._initial_plan_queries()
        rag_context = self._search_pinecone_context(planner_queries)
        self._log_planner_block(
            f"PLANNER RAG CONTEXT [root] queries={planner_queries}",
            rag_context,
        )
        prompt = build_initial_plan_prompt(
            kernel_type=env.kernel_type,
            operation=_kernel_operation_phrase(env.kernel_type),
            aliases=_kernel_aliases(env.kernel_type),
            problem_shape=env.problem_shapes[0],
            kernel_src=env.kernel_src,
            baseline_context=self._planner_baseline_context(),
            rag_context=rag_context,
            branch_count=num_strategies,
        )

        logger.info("Planner: generating %d root branches for %s", num_strategies, env.kernel_type)
        response, _, _ = self._call_llm(prompt, model=self.root_model, temperature=0.2)
        self._log_planner_block("PLANNER RAW OUTPUT [root]", response)
        strategies = parse_plan_response(
            response,
            count=num_strategies,
            prefix="root_plan",
        )
        self._log_planner_block(
            "PLANNER PARSED BRANCHES [root]",
            json.dumps(strategies, indent=2, sort_keys=True),
        )
        if strategies:
            logger.info("Planner produced %d branches", len(strategies))
            return strategies

        logger.warning("Planner returned no usable branches, using fallback plans")
        return fallback_branches(num_strategies, prefix="root_plan")

    # ── Sub-LLM beam generation (parallel) ───────────────────────────────────

