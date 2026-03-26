from __future__ import annotations

import json
from dataclasses import dataclass, field


@dataclass
class PlannerSpec:
    mode: str
    kernel_type: str
    operation: str
    aliases: list[str]
    problem_shape: tuple
    branch_count: int
    kernel_src: str
    rag_context: str
    objective: str
    baseline_context: str = ""
    feedback_summary: str = ""
    parent_strategy: str = ""
    parent_speedup: float | None = None
    success_criteria: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)

    def to_prompt_dict(self) -> dict:
        payload = {
            "mode": self.mode,
            "kernel_type": self.kernel_type,
            "operation": self.operation,
            "aliases": list(self.aliases),
            "problem_shape": list(self.problem_shape),
            "branch_count": self.branch_count,
            "objective": self.objective,
            "success_criteria": list(self.success_criteria),
            "constraints": list(self.constraints),
        }
        if self.baseline_context:
            payload["baseline_context"] = self.baseline_context
        if self.feedback_summary:
            payload["feedback_summary"] = self.feedback_summary
        if self.parent_strategy:
            payload["parent_strategy"] = self.parent_strategy
        if self.parent_speedup is not None:
            payload["parent_speedup"] = round(float(self.parent_speedup), 6)
        return payload

    def to_prompt_json(self) -> str:
        return json.dumps(self.to_prompt_dict(), indent=2, sort_keys=True)


