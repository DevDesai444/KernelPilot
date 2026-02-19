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

