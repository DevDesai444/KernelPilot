"""
reflector.py — Reflection + reward system for the RLM kernel-optimization pipeline.

After each profiling round, computes a numerical reward score and generates a
reflection prompt with real profiler data for the next LLM attempt.

3 templates only:
  1. Compile failure  → show error, fix it
  2. Correctness failure → show code, fix the math
  3. Performance → show reward score + profiler data, make it faster

Reward scoring:
  +20  if kernel compiles
  +100 if correctness passes
  +(baseline_us / optimized_us) × 100  for speedup (e.g. 1.5x = 150 pts)
"""

from __future__ import annotations

import logging
from textwrap import dedent

logger = logging.getLogger(__name__)


# ── Reward computation ────────────────────────────────────────────────────────

def compute_reward(compile_ok: bool, correct: bool, speedup: float) -> tuple[float, str]:
    """Compute numerical reward score.
    Returns (total_score, breakdown_string)."""
    score = 0.0
    parts = []

    if compile_ok:
        score += 20
        parts.append("compile: +20")
    else:
        parts.append("compile: +0 (FAILED)")
        return score, " | ".join(parts)

    if correct:
        score += 100
        parts.append("correctness: +100")
    else:
        parts.append("correctness: +0 (FAILED)")
        return score, " | ".join(parts)

    perf_score = speedup * 100
    score += perf_score
    parts.append(f"speedup: +{perf_score:.0f} ({speedup:.3f}x)")

    return score, " | ".join(parts)


# ── Reflection templates ─────────────────────────────────────────────────────

COMPILE_REFLECTION = dedent("""\
    ## Iteration {iteration}

    **Reward: {reward:.0f}** ({reward_breakdown})

    ### Compiler error
    ```
    {error}
    ```

    ### Your previous solution
    ```cuda
    {solution}
    ```

    Maximize reward. Return the COMPLETE .cu file in a single ```cuda code block.
""")


CORRECTNESS_REFLECTION = dedent("""\
    ## Iteration {iteration}

    **Reward: {reward:.0f}** ({reward_breakdown})

    ### Your previous solution
    ```cuda
    {solution}
    ```

    Maximize reward. Return the COMPLETE .cu file in a single ```cuda code block.
""")


PERFORMANCE_REFLECTION = dedent("""\
    ## Iteration {iteration}

    **Reward: {reward:.0f}** ({reward_breakdown})
    {profile_section}
    {suggestions_section}
    {delta_section}
    {stagnation_section}
    {last_error_section}
    {history_section}

    ### Your previous solution (achieves {speedup:.3f}x)
    ```cuda
    {solution}
    ```

    Do NOT rewrite from scratch. Keep all working optimizations intact.
    Apply ONE targeted change based on the optimization targets above.
    Maximize reward. Return the COMPLETE .cu file in a single ```cuda code block.
""")


# ── Hardware context builder ─────────────────────────────────────────────────

def _build_hw_context(hw_spec: dict) -> str:
    mem = hw_spec.get("memory", {})
    sm = hw_spec.get("sm", {})
    name = hw_spec.get("hardware", {}).get("name", "GPU")
    bw_tbs = mem.get("hbm_bandwidth_tbs", 8.0)
    sm_count = sm.get("count", 142)
    smem_kb = mem.get("shared_memory_per_sm_kb", 228)
    max_threads = sm.get("max_threads_per_sm", 2048)
    warp_size = sm.get("warp_size", 32)

    return dedent(f"""\
        ### {name} Hardware
        - HBM bandwidth: {bw_tbs:.1f} TB/s
        - {sm_count} SMs, {smem_kb} KB shared memory per SM, 255 registers per thread
        - Warp size: {warp_size}, max {max_threads} threads per SM
        - 128-bit load/store transactions (uint4 = 8 bf16 values)
        - Fast math SFU: __expf ~4 cycles vs expf ~20 cycles
        - Warp shuffle: __shfl_xor_sync ~2 cycles vs shared mem reduction ~10+ cycles
    """)


# ── Profile data formatter ───────────────────────────────────────────────────

def _format_profile_section(metrics: dict, iteration: int) -> str:
    """Format real runtime and compiler resource data only."""
    if not metrics:
        return ""

    lines = [f"\n### Profiler Data (Iteration {iteration})"]
    lines.append("```")

    # Real runtime metrics
    occupancy = metrics.get("sm_occupancy", 0)
    duration = metrics.get("duration_us", 0)
    speedup = metrics.get("speedup", 1.0)

    lines.append(f"Kernel timing:                 {duration:.3f} us")
    lines.append(f"Speedup vs FlashInfer:         {speedup:.3f}x")
    lines.append(f"SM occupancy:                  {occupancy:.1f}%")

    stall_mem = metrics.get("stall_memory", 0)
    dram_bw = metrics.get("dram_read_bw_gbps", 0)
    l2_hit = metrics.get("l2_hit_rate", 0)

    if stall_mem > 0 or dram_bw > 0 or l2_hit > 0:
        lines.append("")
        if stall_mem > 0:
            lines.append(f"Warp stalls (memory):          {stall_mem:.1f}%")
        if dram_bw > 0:
            lines.append(f"DRAM read bandwidth:           {dram_bw:.0f} GB/s")
        if l2_hit > 0:
            lines.append(f"L2 hit rate:                   {l2_hit:.1f}%")

    # Compiler metrics (from nvcc -Xptxas -v)
    cm = metrics.get("_compiler", {})
    if cm:
        regs = cm.get("registers_per_thread", 0)
        spill_total = cm.get("spill_stores_bytes", 0) + cm.get("spill_loads_bytes", 0)
        smem = cm.get("static_smem_bytes", 0)
        stack = cm.get("stack_frame_bytes", 0)

        lines.append("")
        lines.append(f"Registers per thread:          {regs}")
        lines.append(f"Register spills:               {spill_total} bytes{' *** SPILLING ***' if spill_total > 0 else ''}")
        lines.append(f"Shared memory:                 {smem} bytes")
        if stack > 0:
            lines.append(f"Stack frame:                   {stack} bytes")

    lines.append("```")
    return "\n".join(lines)


# ── Last failed refinement error ──────────────────────────────────────────────

def _format_last_error_section(candidate) -> str:
    """If the model's last refinement attempt failed, show the error so it
    doesn't repeat the same mistake."""
    error = getattr(candidate, 'last_refine_error', '')
    if not error:
        return ""
    # Truncate long errors to keep the prompt focused
    if len(error) > 600:
        error = error[:600] + "\n... (truncated)"
    return dedent(f"""\

        ### Your Last Refinement Attempt FAILED
        ```
        {error}
        ```
        Do NOT repeat this mistake. Fix the error while improving performance.
    """)


# ── Proven-ineffective detection ──────────────────────────────────────────────

def _compute_proven_ineffective(latest: dict, best: dict) -> tuple:
    """Stubbed: We no longer track hardcoded 'ineffective' metrics.
    We pass pure universal metrics and let the LLM deduce failure."""
    return set(), []


# ── Data-driven optimization suggestions ─────────────────────────────────────

_REDUCTION_KERNEL_TYPES = frozenset({
    "add_rmsnorm", "rmsnorm", "layernorm", "softmax", "logsoftmax",
    "cross_entropy", "rms_norm", "layer_norm",
})


def _format_suggestions_section(metrics: dict, ineffective: set = None,
                                 kernel_type: str = "") -> str:
    """Generate data-driven suggestions from runtime and compiler-resource data."""
    cm = metrics.get("_compiler", {})

    hints: list[str] = []
    occupancy = metrics.get("sm_occupancy", 0)

    spill_total = cm.get("spill_stores_bytes", 0) + cm.get("spill_loads_bytes", 0)
    if spill_total > 0:
        hints.append(
            f"Register spills ({spill_total} bytes) — reduce live state or simplify per-thread work"
        )
    regs = cm.get("registers_per_thread", 0)
    if regs >= 96:
        hints.append(
            f"High register count ({regs} per thread) — trim live ranges before adding more fused work"
        )
    if occupancy > 0 and occupancy < 75.0:
        hints.append(
            f"Low SM occupancy ({occupancy:.1f}%) — prefer smaller localized changes that preserve occupancy"
        )
    if not hints:
        return ""

    lines = ["\n### Optimization Signals"]
    for h in hints:
        lines.append(f"- {h}")
    return "\n".join(lines)


# ── Refinement history ────────────────────────────────────────────────────────

def _format_history_section(candidate) -> str:
    """Show the model what optimizations were already tried across rounds."""
    history = getattr(candidate, 'refinement_history', [])
    if not history:
        return ""
    lines = ["\n### Refinement History (do NOT repeat failed/stagnant approaches)"]
    for entry in history:
        outcome = entry.get("outcome", "?")
        speedup = entry.get("speedup", 0)
        strategy = entry.get("strategy", "?")
        rnd = entry.get("round", "?")
        desc = entry.get("strategy_desc", "")
        lines.append(f"- Round {rnd}: {strategy} → {outcome} ({speedup:.3f}x)")
        # Show what was tried for failed attempts so the LLM avoids repeating them
        if desc and outcome in ("regression", "stagnant", "compile_fail", "correctness_fail"):
            lines.append(f"  Attempted: {desc[:200]}")
    return "\n".join(lines)


def _format_react_trace(candidate) -> str:
    """Format refinement history as a ReAct trace: Action → Result pairs.

    Shows the model its own prior reasoning chain so it can learn from
    what worked and what didn't — proper ReAct interleaving.
    """
    history = getattr(candidate, 'refinement_history', [])
    if not history:
        return ""

    lines = []
    for i, entry in enumerate(history, 1):
        outcome = entry.get("outcome", "?")
        speedup = entry.get("speedup", 0)
        desc = entry.get("strategy_desc", "")

        # Map outcomes to clear result labels
        result_map = {
            "improved": "IMPROVED",
            "regression": "SLOWER",
            "stagnant": "NO CHANGE",
            "compile_fail": "COMPILE ERROR",
            "correctness_fail": "WRONG OUTPUT",
        }
        result_label = result_map.get(outcome, outcome.upper())

        action_text = desc[:250] if desc else "(no description)"
        lines.append(f"Action {i}: {action_text}")
        lines.append(f"Result {i}: {result_label} ({speedup:.3f}x)")
    return "\n".join(lines)


# ── Stagnation detection ─────────────────────────────────────────────────────

def _format_stagnation_section(metrics: dict, prev_metrics: dict, iteration: int,
                               candidate=None) -> str:
    """Detect reward stagnation and tell the model to change approach.
    Only triggers after at least one refinement attempt has been made,
    to avoid falsely firing on carried-over survivors whose metrics==prev_metrics."""
    if not metrics or not prev_metrics or iteration < 1:
        return ""

    # Don't trigger stagnation if this beam hasn't had a failed refinement yet —
    # metrics and prev_metrics are identical for fresh/newly-promoted survivors.
    # Use refine_attempts (tracks actual failures on THIS candidate) rather than
    # refinement_history (inherited from parent lineage).
    if candidate is not None:
        if getattr(candidate, 'refine_attempts', 0) == 0:
            return ""

    cur_speedup = metrics.get("speedup", 1.0)
    prev_speedup = prev_metrics.get("speedup", 1.0)
    delta = abs(cur_speedup - prev_speedup)

    if delta < 0.02:  # Less than 2% improvement = stagnant
        return dedent(f"""\

            ### Stagnation Detected
            Speedup has NOT improved: {prev_speedup:.3f}x -> {cur_speedup:.3f}x
            Your previous refinement attempts are not working.
            Try a different optimization technique than what's listed in the Refinement History.
        """)
    return ""


# ── Round-over-round delta ───────────────────────────────────────────────────

def _format_delta_section(current: dict, previous: dict,
                          title: str = "Changes vs Previous Round") -> str:
    """Format round-over-round deltas using only real measured data."""
    if not previous or not current:
        return ""

    lines = [f"\n### {title}"]
    lines.append("```")

    # Timing delta
    cur_timing = current.get("duration_us", 0)
    prev_timing = previous.get("duration_us", 0)
    if cur_timing > 0 and prev_timing > 0:
        delta_us = cur_timing - prev_timing
        arrow = "v" if delta_us < 0 else "^" if delta_us > 0 else "="
        better = "  (faster)" if delta_us < 0 else "  (slower)" if delta_us > 0 else ""
        lines.append(f"{'Kernel timing (us)':30s}  {prev_timing:9.3f} -> {cur_timing:9.3f}  ({arrow} {abs(delta_us):.3f}){better}")

    # Speedup delta
    cur_spd = current.get("speedup", 1.0)
    prev_spd = previous.get("speedup", 1.0)
    if cur_spd != prev_spd:
        lines.append(f"{'Speedup':30s}  {prev_spd:9.3f}x -> {cur_spd:9.3f}x")

    # Occupancy delta
    cur_occ = current.get("sm_occupancy", 0)
    prev_occ = previous.get("sm_occupancy", 0)
    if cur_occ != prev_occ and (cur_occ > 0 or prev_occ > 0):
        delta = cur_occ - prev_occ
        arrow = "^" if delta > 0 else "v" if delta < 0 else "="
        lines.append(f"{'SM occupancy':30s}  {prev_occ:6.1f}% -> {cur_occ:6.1f}%  ({arrow} {abs(delta):.1f})")

    # Compiler metric deltas
    cur_cm = current.get("_compiler", {})
    prev_cm = previous.get("_compiler", {})
    if cur_cm and prev_cm:
        cur_regs = cur_cm.get("registers_per_thread", 0)
        prev_regs = prev_cm.get("registers_per_thread", 0)
        if cur_regs != prev_regs:
            lines.append(f"{'Registers/thread':30s}  {prev_regs:6d}  -> {cur_regs:6d}")

        cur_spills = cur_cm.get("spill_stores_bytes", 0) + cur_cm.get("spill_loads_bytes", 0)
        prev_spills = prev_cm.get("spill_stores_bytes", 0) + prev_cm.get("spill_loads_bytes", 0)
        if cur_spills != prev_spills:
            warn = "  *** REGRESSED ***" if cur_spills > prev_spills else "  (improved)" if cur_spills < prev_spills else ""
            lines.append(f"{'Register spill bytes':30s}  {prev_spills:6d}  -> {cur_spills:6d}{warn}")

        for label, key in [
            ("Shared memory bytes", "static_smem_bytes"),
            ("Stack frame bytes", "stack_frame_bytes"),
            ("Constant memory bytes", "cmem_bytes"),
        ]:
            cur_v = cur_cm.get(key, 0)
            prev_v = prev_cm.get(key, 0)
            if cur_v != prev_v and (cur_v > 0 or prev_v > 0):
                lines.append(f"{label:30s}  {prev_v:6d}  -> {cur_v:6d}")

    lines.append("```")

    # Only return if we actually have deltas (more than header + backticks)
    if len(lines) <= 3:
        return ""
    return "\n".join(lines)


# ── Launch function signatures (must match harness exactly) ──────────────────

def _get_launch_signature(kernel_type: str) -> str:
    """Return the EXACT launch function signature the harness expects.
    If the kernel changes this, it gets 'undefined reference' linker errors."""
    sigs = {
        "add_rmsnorm": dedent("""\
            ### Required Launch Function (DO NOT CHANGE)
            The benchmark harness calls this exact function. Your code MUST define it with this exact signature:
            ```c
            void launch_fused_add_rmsnorm_nvfp4(
                const __nv_bfloat16* input,
                const __nv_bfloat16* residual,
                const __nv_bfloat16* weight,
                __nv_bfloat16* residual_out,
                unsigned char* quant_out,
                __nv_fp8_storage_t* scale_out,
                int rows, int hidden,
                cudaStream_t stream);
            ```
            - `unsigned char*` for quant_out, NOT `uint8_t*`
            - `__nv_fp8_storage_t*` for scale_out
            - Exactly 9 parameters in this order
        """),
        "silu_mul": dedent("""\
            ### Required Launch Function (DO NOT CHANGE)
            The benchmark harness calls this exact function. Your code MUST define it with this exact signature:
            ```c
            void launch_silu_mul_fp4quant(
                const __nv_bfloat16* gate,
                const __nv_bfloat16* up,
                uint8_t* quant_out,
                __nv_fp8_storage_t* scale_out,
                int N,
                cudaStream_t stream);
            ```
            - Exactly 6 parameters in this order
        """),
        "nvfp4_quantize": dedent("""\
            ### Required Launch Function (DO NOT CHANGE)
            The benchmark harness calls this exact function. Your code MUST define it with this exact signature:
            ```c
            void launch_nvfp4_quantize_bf16(
                const __nv_bfloat16* input,
                uint8_t* packed_out,
                __nv_fp8_storage_t* scale_out,
                int N,
                cudaStream_t stream);
            ```
            - Exactly 5 parameters in this order
        """),
    }
    return sigs.get(kernel_type, "")


# ── Critical rules footer ────────────────────────────────────────────────────

CRITICAL_RULES = dedent("""\
    ### Critical Rules
    1. Return the COMPLETE .cu file in a single ```cuda code block.
    2. Keep all #includes (use the original #include directives, NOT expanded content).
    3. Do NOT use torch headers (torch/extension.h, ATen, c10) -- this is standalone CUDA.
    4. Keep ALL kernel functions and the launch_* wrapper function.
    5. The launch_* function signature MUST match the "Required Launch Function" section EXACTLY.
       If you change it, the code will fail with "undefined reference" linker errors.
    6. Output must match reference within atol=1e-2.
    7. NEVER put __syncthreads() inside an if/else branch -- all threads in a block MUST hit the same barrier or the kernel will deadlock.
    8. No explanations -- just the code block.
    9. Only call functions defined in the included headers. Do NOT invent helper functions.
""")


# ── Main reflection dispatcher ───────────────────────────────────────────────

def reflect(
    candidate,
    iteration: int,
    hw_spec: dict,
    kernel_type: str = "",
    baseline_us: float = 0.0,
    prev_metrics: dict = None,
    atol: float = 1e-2,
    original_kernel_src: str = "",
    **kwargs,  # absorb unused args (target_speedup, min_speedup) from callers
) -> str:
    """Generate a reflection prompt based on the candidate's result.

    3 paths:
      1. compile_ok=False → COMPILE_REFLECTION + compiler error
      2. correct=False    → CORRECTNESS_REFLECTION
      3. otherwise        → PERFORMANCE_REFLECTION + reward score + profiler data
    """
    solution = candidate.code or "# (no code generated)"
    metrics = candidate.metrics or {}
    speedup = candidate.speedup
    hw_context = _build_hw_context(hw_spec)
    launch_sig = _get_launch_signature(kernel_type)
    profile_section = _format_profile_section(metrics, iteration)
    delta_section = _format_delta_section(metrics, prev_metrics)
    stagnation_section = _format_stagnation_section(metrics, prev_metrics, iteration, candidate=candidate)
    last_error_section = _format_last_error_section(candidate)
    history_section = _format_history_section(candidate)
    suggestions_section = _format_suggestions_section(metrics, kernel_type=kernel_type)

    reward, reward_breakdown = compute_reward(
        candidate.compile_ok, candidate.correct, speedup
    )

    # Reference source section — gives the model the original headers/types
    ref_section = ""
    if original_kernel_src:
        ref_section = dedent(f"""\

            ### Reference kernel (headers expanded — use these types and functions):
            ```cuda
            {original_kernel_src}
            ```
        """)

    # Common footer: reference source + launch signature + hw context + rules
    footer = "\n" + ref_section + "\n" + launch_sig + "\n" + hw_context + "\n" + CRITICAL_RULES

    if not candidate.compile_ok:
        error = getattr(candidate, 'compile_error', '') or "Unknown compilation error"
        return COMPILE_REFLECTION.format(
            iteration=iteration,
            reward=reward,
            reward_breakdown=reward_breakdown,
            error=error,
            solution=solution,
        ) + footer

    if not candidate.correct:
        return CORRECTNESS_REFLECTION.format(
            iteration=iteration,
            reward=reward,
            reward_breakdown=reward_breakdown,
            solution=solution,
        ) + footer

    return PERFORMANCE_REFLECTION.format(
        iteration=iteration,
        reward=reward,
        reward_breakdown=reward_breakdown,
        speedup=speedup,
        solution=solution,
        profile_section=profile_section,
        suggestions_section=suggestions_section,
        delta_section=delta_section,
        stagnation_section=stagnation_section,
        last_error_section=last_error_section,
        history_section=history_section,
    ) + footer
