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

