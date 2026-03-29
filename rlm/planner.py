from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from .planner_spec import (
    PlannerSpec,
    build_root_planner_spec,
    build_tree_planner_spec,
)


@dataclass
class PlanBranch:
    name: str
    goal: str
    change_summary: str
    expected_signal: str
    bottleneck: str = ""
    rag_queries: list[str] = field(default_factory=list)
    planner_notes: str = ""
    rationale: str = ""
    risk: str = ""
    evidence: list[str] = field(default_factory=list)
    parent_strategy: str = ""
    tree_ready: bool = False

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "goal": self.goal,
            "bottleneck": self.bottleneck,
            "what": self.change_summary,
            "change_summary": self.change_summary,
            "expected_signal": self.expected_signal,
            "rag_queries": list(self.rag_queries),
            "planner_notes": self.planner_notes,
            "rationale": self.rationale,
            "risk": self.risk,
            "evidence": list(self.evidence),
            "parent_strategy": self.parent_strategy,
            "tree_ready": self.tree_ready,
        }


def _coerce_string_list(value) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def fallback_branches(
    count: int,
    prefix: str,
    parent_strategy: str = "",
) -> list[dict]:
    branches = []
    for idx in range(count):
        branches.append(
            PlanBranch(
                name=f"{prefix}_{idx + 1}",
                goal="Make one measurable CUDA optimization change.",
                bottleneck="",
                change_summary="Implement one targeted optimization and preserve correctness.",
                expected_signal="Compiler succeeds and sandbox metrics improve.",
                rag_queries=[],
                planner_notes="Fallback plan because the planner output could not be parsed.",
                rationale="Fallback branch used because the planner response was invalid.",
                risk="Low confidence: planner output was missing or malformed.",
                evidence=[],
                parent_strategy=parent_strategy,
                tree_ready=bool(parent_strategy),
            ).to_dict()
        )
    return branches


