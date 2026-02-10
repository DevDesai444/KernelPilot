"""
environment.py — RLM environment state management.
Loads kernel source, hardware spec, manages optimization state across rounds.
"""

from __future__ import annotations
import logging
import re
import yaml
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

PROJECT_ROOT = Path(__file__).parent.parent
logger = logging.getLogger(__name__)


@dataclass
class KernelCandidate:
    """Represents a single kernel variant at any point in the search."""
    code: str
    strategy: str
    round_num: int = 0
    metrics: dict = field(default_factory=dict)
    speedup: float = 1.0
    bottleneck: str = "unknown"
    compile_ok: bool = False
    correct: bool = False
    profile_report_path: Optional[str] = None
    prev_metrics: Optional[dict] = None  # parent's metrics for delta comparison
    compile_error: str = ""  # compiler error message for reflection
    last_refine_error: str = ""  # error from last failed refinement attempt
    refinement_history: list = field(default_factory=list)  # [{round, strategy, outcome, speedup}]
    refine_attempts: int = 0  # times this candidate was refined without improvement
    best_code: str = ""       # code that achieved best_speedup (for refinement base)
    best_speedup: float = 0.0 # best speedup seen for this beam lineage
    strategy_context: str = "" # original strategy description — anchors refinement direction
    parent_strategy: str = ""
    branch_family: str = ""
    plan_branch: dict = field(default_factory=dict)
    feedback_route: str = ""

    def is_viable(self) -> bool:
        return self.compile_ok and self.correct

    def summary(self) -> str:
        return (
            f"[{self.strategy}] round={self.round_num} "
            f"speedup={self.speedup:.3f}x "
            f"family={self.bottleneck} "
            f"compile={'ok' if self.compile_ok else 'FAIL'} "
            f"correct={'yes' if self.correct else 'NO'}"
        )


@dataclass
class OptimizationHistory:
    """Tracks everything tried across all rounds and strategies."""
    entries: list = field(default_factory=list)

    def record(self, candidate: KernelCandidate, notes: str = "") -> None:
        self.entries.append({
            "timestamp": time.time(),
            "strategy": candidate.strategy,
            "round": candidate.round_num,
            "speedup": candidate.speedup,
            "family": candidate.bottleneck,
            "compile_ok": candidate.compile_ok,
            "correct": candidate.correct,
            "notes": notes,
        })

    def best_speedup(self) -> float:
        viable = [e["speedup"] for e in self.entries
                  if e["compile_ok"] and e["correct"]]
        return max(viable) if viable else 1.0

    def strategies_tried(self) -> list:
        return list({e["strategy"] for e in self.entries})

    def to_summary_str(self) -> str:
        lines = []
        for e in self.entries:
            lines.append(
                f"  round={e['round']} strategy={e['strategy']} "
                f"speedup={e['speedup']:.3f}x "
                f"ok={e['compile_ok']} correct={e['correct']}"
            )
        return "\n".join(lines) if lines else "  (none)"


class RLMEnvironment:
    """
    Shared state object for the RLM REPL.
    Root LLM reads/writes this; sub-LLMs get slices of it.
    """

