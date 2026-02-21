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


