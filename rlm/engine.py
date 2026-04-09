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

    async def _run_agent_loop(
        self,
        initial_prompt: str,
        strategy_name: str,
        round_num: int,
        profile_fn,
        model_id: str,
        comparison_speedup: float = 0.0,
        prev_inner_metrics: dict | None = None,
        strategy_context: str = "",
        plan_branch: dict | None = None,
        parent_candidate: KernelCandidate | None = None,
    ) -> KernelCandidate:
        messages = [{"role": "user", "content": initial_prompt}]
        best = None
        last_error = ""
        feedback_route = ""
        submit_count = 0
        max_api_turns = MAX_INNER_TURNS + 4
        best_speedup = comparison_speedup
        plan_branch = dict(plan_branch or {})

        for turn in range(max_api_turns):
            if submit_count >= MAX_INNER_TURNS:
                break

            system_prompt = _build_refine_system_prompt(
                best_speedup,
                prev_inner_metrics,
                kernel_type=self.env.kernel_type,
                problem_shape=self.env.problem_shapes[0],
            )
            try:
                response = await self._call_llm_with_tools_async(
                    messages=messages,
                    tools=REFINE_TOOLS,
                    model=model_id,
                    system=system_prompt,
                    temperature=0.4,
                )
            except RuntimeError as exc:
                logger.error("Budget exceeded for %s turn %d: %s", strategy_name, turn, exc)
                break

            text_blocks = [
                block.text for block in response.content
                if hasattr(block, "text") and block.text.strip()
            ]
            if text_blocks:
                logger.info(
                    "\nAGENT [%s turn %d]:\n%s\n",
                    strategy_name,
                    turn,
                    "\n".join(text_blocks),
                )

            messages.append({"role": "assistant", "content": response.content})

            submit_code, submit_block_id, aux_results = self._handle_tool_calls(
                response, messages, profile_fn, strategy_name, round_num,
                max(best_speedup, comparison_speedup), prev_inner_metrics,
            )

            if submit_code is None and not submit_block_id:
                has_any_tool = any(block.type == "tool_use" for block in response.content)
                if aux_results:
                    messages.append({"role": "user", "content": aux_results})
                    continue
                if not has_any_tool:
                    break
                continue

            if submit_code is None:
                if aux_results:
                    messages.append({"role": "user", "content": aux_results})
                continue

            submit_count += 1
            if profile_fn:
                result = await asyncio.get_event_loop().run_in_executor(
                    None, profile_fn, submit_code, strategy_name, round_num
                )
            else:
                result = {
                    "compile_ok": False,
                    "correct": False,
                    "speedup": 0.0,
                    "metrics": {},
                    "error": "No profiler available",
                    "branch_family": "unknown",
                }

            feedback = build_sandbox_feedback(
                result=result,
                parent_speedup=max(best_speedup, comparison_speedup),
                prev_inner_metrics=prev_inner_metrics,
                kernel_type=self.env.kernel_type,
                candidate=best or parent_candidate,
            )
            feedback_route = feedback.route

            if result["compile_ok"] and result["correct"] and result.get("metrics"):
                prev_inner_metrics = result["metrics"]

            all_results = list(aux_results) + [
                {
                    "type": "tool_result",
                    "tool_use_id": submit_block_id,
                    "content": feedback.to_tool_result_json(),
                }
            ]
            messages.append({"role": "user", "content": all_results})

            logger.info(
                "SANDBOX [%s submit %d]: route=%s speedup=%.3fx compile=%s correct=%s",
                strategy_name,
                submit_count,
                feedback.route,
                result.get("speedup", 0.0),
                result.get("compile_ok"),
                result.get("correct"),
            )

            if result["compile_ok"] and result["correct"]:
                if best is None or result["speedup"] > best.speedup:
                    best = KernelCandidate(
                        code=submit_code,
                        strategy=strategy_name,
                        round_num=round_num,
                        compile_ok=True,
                        correct=True,
                        speedup=result["speedup"],
                        metrics=result.get("metrics", {}),
                        bottleneck=result.get("branch_family") or result.get("bottleneck", "unknown"),
                        prev_metrics=parent_candidate.metrics if parent_candidate else None,
                        parent_strategy=(
                            parent_candidate.strategy if parent_candidate else plan_branch.get("parent_strategy", "")
                        ),
                        branch_family=(
                            parent_candidate.branch_family
                            if parent_candidate and parent_candidate.branch_family
                            else (plan_branch.get("parent_strategy") or plan_branch.get("name") or strategy_name)
                        ),
                        plan_branch=dict(plan_branch),
                        feedback_route=feedback.route,
                    )
                    best.strategy_context = strategy_context
                    best.best_code = submit_code
                    best.best_speedup = result["speedup"]
                    best_speedup = result["speedup"]
            else:
                last_error = result.get("error", "") or feedback.uncertainty

        if best:
            return best

        failed = KernelCandidate(
            code="",
            strategy=strategy_name,
            round_num=round_num,
            compile_ok=False,
            prev_metrics=parent_candidate.metrics if parent_candidate else None,
            parent_strategy=(
                parent_candidate.strategy if parent_candidate else plan_branch.get("parent_strategy", "")
            ),
            branch_family=(
                parent_candidate.branch_family
                if parent_candidate and parent_candidate.branch_family
                else (plan_branch.get("parent_strategy") or plan_branch.get("name") or strategy_name)
            ),
            plan_branch=dict(plan_branch),
            feedback_route=feedback_route,
        )
        failed.compile_error = last_error or "All inner refinement attempts failed"
        failed.strategy_context = strategy_context
        return failed

    async def _generate_single_beam(
        self,
        strategy,
        kernel_slice: str,
        current_metrics: dict = None,
        round_num: int = 0,
        profile_fn=None,
    ) -> KernelCandidate:
        if isinstance(strategy, dict):
            plan_branch = dict(strategy)
            strat_name = plan_branch.get("name", "unknown")
            strat_desc = plan_branch.get("change_summary") or plan_branch.get("what", "")
        else:
            strat_name = str(strategy)
            strat_desc = ""
            plan_branch = {
                "name": strat_name,
                "goal": strat_name,
                "what": strat_desc,
                "change_summary": strat_desc,
                "expected_signal": "Sandbox output improves.",
                "rag_queries": [],
            }

        launch_sig = _get_launch_signature(self.env.kernel_type)

        if profile_fn and strat_desc:
            rag_context = self._search_pinecone_context(
                plan_branch.get("rag_queries")
                or [f"{self.env.kernel_type} {strat_name} CUDA optimization"]
            )
            current_profile = (
                _format_profile_section(current_metrics, round_num)
                if current_metrics else ""
            )
            _bcm = self.env.baseline_compiler_metrics
            _baseline_regs = int(getattr(_bcm, "registers_per_thread", 0) or 0) if _bcm else 0
            initial_prompt = build_coder_prompt(
                plan_branch=plan_branch,
                kernel_code=kernel_slice,
                launch_signature=launch_sig,
                rag_context=rag_context,
                current_profile=current_profile,
                baseline_regs=_baseline_regs,
            )
            return await self._run_agent_loop(
                initial_prompt=initial_prompt,
                strategy_name=strat_name,
                round_num=round_num,
                profile_fn=profile_fn,
                model_id=self.sub_model,
                comparison_speedup=0.0,
                prev_inner_metrics=current_metrics,
                strategy_context=strat_desc,
                plan_branch=plan_branch,
            )

        # ── One-shot fallback (no profile_fn or no description) ──────────
        if strat_desc:
            shape_str = str(self.env.problem_shapes[0])
            prompt = f"""\
You are an expert CUDA kernel optimizer targeting NVIDIA B200 (sm_100a, Blackwell).

Apply this optimization to the kernel below:

## Optimization: {strat_name}
{strat_desc}

## Context
Speedup is measured against FlashInfer, a production GPU library.
You target ONE GPU (B200, sm_100a) and ONE shape ({shape_str}).

## Naive reference kernel (starting point):
```cuda
{kernel_slice}
```

{launch_sig}

CRITICAL RULES:
1. Return the COMPLETE .cu file in a single ```cuda code block
2. Keep all #includes (use the original #include directives, NOT the expanded content)
3. Do NOT use torch headers (torch/extension.h, ATen, c10) — this is standalone CUDA
4. Keep ALL kernel functions and the launch_* wrapper function
5. The launch_* function signature MUST match the "Required Launch Function" section EXACTLY.
   If you change it, you will get "undefined reference" linker errors.
6. Output must match reference within atol=1e-2
7. NEVER put __syncthreads() inside an if/else branch — all threads in a block MUST hit the same barrier or the kernel will deadlock.
8. No explanations — just the code block
9. You may call any function defined in the expanded headers above. Do NOT invent
   helper functions that aren't defined in the headers.
"""
        else:
            # No strategy description — use minimal prompt
            prompt = f"""\
You are an expert CUDA kernel optimizer targeting NVIDIA B200 (sm_100a, Blackwell).

Apply the "{strat_name}" optimization to this kernel:

```cuda
{kernel_slice}
```

{launch_sig}

Return the COMPLETE .cu file in a single ```cuda code block. No explanations.
"""

        try:
            response, _, _ = await self._call_llm_async(
                prompt, model=self.sub_model, temperature=0.6
            )
        except RuntimeError as e:
            logger.error("Budget exceeded during beam %s: %s", strat_name, e)
            return KernelCandidate(code="", strategy=strat_name, round_num=round_num)

        code = self._extract_cuda_code(response)
        if not code:
            logger.warning("No CUDA code extracted for strategy=%s (response starts: %s)",
                           strat_name, response[:100])
        c = KernelCandidate(
            code=code,
            strategy=strat_name,
            round_num=round_num,
            compile_ok=bool(code),
            branch_family=(plan_branch.get("parent_strategy") or plan_branch.get("name") or strat_name),
            plan_branch=plan_branch,
        )
        c.strategy_context = strat_desc
        return c

    async def generate_beams(
        self,
        strategies: list,
        kernel_slice: str,
        current_metrics: dict = None,
        round_num: int = 0,
        profile_fn=None,
    ) -> list:
        tasks = [
            self._generate_single_beam(
                s, kernel_slice, current_metrics, round_num, profile_fn)
            for s in strategies
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return [
            result if not isinstance(result, Exception)
            else self._beam_exception_candidate(strategy, round_num, result)
            for strategy, result in zip(strategies, results)
        ]

    # ── Refinement: multi-turn tool-use loop ─────────────────────────────────

    async def refine_beams(self, survivors: list, round_num: int,
                           profile_fn=None) -> list:
        tasks = []
        fallbacks = []
        for candidate in survivors:
            if candidate.compile_ok and candidate.correct:
                branch_count = (
                    self.tree_branching_factor
                    if candidate.speedup >= self.tree_speedup_threshold
                    and candidate.feedback_route == "planner_tree"
                    else 1
                )
                followup_plans = self._expand_tree_plans(candidate, branch_count=branch_count)
                for child_plan in followup_plans:
                    fallbacks.append(
                        self._refine_exception_candidate(candidate, round_num, child_plan)
                    )
                    tasks.append(
                        self._refine_single_beam(
                            candidate,
                            round_num,
                            profile_fn=profile_fn,
                            plan_branch=child_plan,
                            fixer_mode=False,
                        )
                    )
            else:
                feedback = build_sandbox_feedback(
                    {
                        "compile_ok": candidate.compile_ok,
                        "correct": candidate.correct,
                        "speedup": candidate.speedup,
                        "metrics": candidate.metrics,
                        "error": candidate.compile_error,
                    },
                    parent_speedup=candidate.speedup,
                    prev_inner_metrics=candidate.prev_metrics,
                    kernel_type=self.env.kernel_type,
                    candidate=candidate,
                )
                tasks.append(
                    self._refine_single_beam(
                        candidate,
                        round_num,
                        profile_fn=profile_fn,
                        plan_branch={
                            "name": f"{candidate.strategy}_repair",
                            "goal": "Repair the failing branch without changing its overall strategy family.",
                            "what": feedback.next_action,
                            "change_summary": feedback.next_action,
                            "bottleneck": "",
                            "expected_signal": "Compilation succeeds, correctness holds, and speed improves.",
                            "rag_queries": feedback.rag_queries,
                            "planner_notes": feedback.planner_summary(),
                            "parent_strategy": candidate.strategy,
                            "tree_ready": False,
                        },
                        fixer_mode=True,
                        feedback=feedback,
                    )
                )
                fallbacks.append(
                    self._refine_exception_candidate(
                        candidate,
                        round_num,
                        {
                            "name": f"{candidate.strategy}_repair",
                            "parent_strategy": candidate.strategy,
                        },
                    )
                )
        results = await asyncio.gather(*tasks, return_exceptions=True)
        refined = []
        for fallback, result in zip(fallbacks, results):
            if isinstance(result, Exception):
                fallback.compile_error = f"Refinement task failed: {result}"
                refined.append(fallback)
            else:
                refined.append(result)
        return refined

    def _beam_exception_candidate(self, strategy, round_num: int, exc: Exception) -> KernelCandidate:
        plan_branch = strategy if isinstance(strategy, dict) else {}
        strategy_name = plan_branch.get("name") if isinstance(plan_branch, dict) else str(strategy)
        family = (
            plan_branch.get("parent_strategy")
            or plan_branch.get("name")
            or strategy_name
        )
        return KernelCandidate(
            code="",
            strategy=strategy_name or "beam_failed",
            round_num=round_num,
            compile_ok=False,
            compile_error=f"Beam generation failed: {exc}",
            branch_family=(family or "").split("__", 1)[0],
            plan_branch=plan_branch,
        )

    def _refine_exception_candidate(
        self,
        parent: KernelCandidate,
        round_num: int,
        plan_branch: dict | None = None,
    ) -> KernelCandidate:
        plan_branch = dict(plan_branch or {})
        branch_name = plan_branch.get("name", "refine_failed")
        return KernelCandidate(
            code=parent.best_code or parent.code,
            strategy=f"{parent.strategy}__{branch_name}_r{round_num}",
            round_num=round_num,
            speedup=0.0,
            compile_ok=False,
            correct=False,
            branch_family=parent.branch_family or parent.strategy.split("__", 1)[0],
            parent_strategy=parent.strategy,
            plan_branch=plan_branch,
        )

    async def _refine_single_beam(
        self,
        parent: 'KernelCandidate',
        round_num: int,
        profile_fn=None,
        plan_branch: dict | None = None,
        fixer_mode: bool = False,
        feedback=None,
    ) -> 'KernelCandidate':
        metrics = parent.metrics or {}
        launch_sig = _get_launch_signature(self.env.kernel_type)
        base_code = parent.best_code or parent.code
        plan_branch = dict(plan_branch or parent.plan_branch or {})
        plan_branch.setdefault("parent_strategy", parent.strategy)
        branch_name = plan_branch.get("name", "repair")
        strategy_name = f"{parent.strategy}__{branch_name}_r{round_num}"

        if feedback is None:
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

        rag_queries = plan_branch.get("rag_queries") or feedback.rag_queries
        rag_context = self._search_pinecone_context(
            rag_queries or [f"{self.env.kernel_type} {branch_name} CUDA optimization"]
        )

        if fixer_mode:
            initial_prompt = build_fixer_prompt(
                plan_branch=plan_branch,
                kernel_code=base_code,
                launch_signature=launch_sig,
                rag_context=rag_context,
                feedback_json=feedback.to_tool_result_json(),
            )
            model_id = self.fixer_model
        else:
            current_profile = _format_profile_section(metrics, round_num) if metrics else ""
            _bcm2 = self.env.baseline_compiler_metrics
            _baseline_regs2 = int(getattr(_bcm2, "registers_per_thread", 0) or 0) if _bcm2 else 0
            initial_prompt = build_coder_prompt(
                plan_branch=plan_branch,
                kernel_code=base_code,
                launch_signature=launch_sig,
                rag_context=rag_context,
                current_profile=current_profile,
                baseline_regs=_baseline_regs2,
            )
            model_id = self.sub_model

        return await self._run_agent_loop(
            initial_prompt=initial_prompt,
            strategy_name=strategy_name,
            round_num=round_num,
            profile_fn=profile_fn,
            model_id=model_id,
            comparison_speedup=parent.speedup,
            prev_inner_metrics=metrics,
            strategy_context=plan_branch.get("change_summary") or plan_branch.get("what", ""),
            plan_branch=plan_branch,
            parent_candidate=parent,
        )

    # ── Auxiliary tool handlers ────────────────────────────────────────────────

    def _handle_read_file(self, path: str) -> str:
        """Read an allowed project file."""
        clean = path.strip()
        if not clean:
            return "File path is required."

        candidate_path = (PROJECT_ROOT / clean.lstrip("/")).resolve()
        if not any(root == candidate_path or root in candidate_path.parents for root in ALLOWED_READ_ROOTS):
            return (f"Access denied: '{path}'. Allowed paths:\n"
                    "- kernels/common/nvfp4_utils.cuh\n"
                    "- kernels/common/b200_intrinsics.cuh\n"
                    "- kernels/reference/add_rmsnorm.cu\n"
                    "- kernels/reference/silu_mul.cu\n"
                    "- kernels/reference/nvfp4_quantize.cu")
        if not candidate_path.exists():
            return f"File not found: {clean}"
        try:
            content = candidate_path.read_text()
            # Truncate very large files
            if len(content) > 12000:
                content = content[:12000] + "\n... (truncated)"
            return content
        except Exception as e:
            return f"Error reading {candidate_path.relative_to(PROJECT_ROOT)}: {e}"

