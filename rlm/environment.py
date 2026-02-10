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
