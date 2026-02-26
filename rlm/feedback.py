from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

KERNEL_QUERY_CONTEXT = {
    "add_rmsnorm": {
        "operation": "fused add rmsnorm fp4 quantization",
        "aliases": ["rmsnorm", "residual add", "fp4 quantization", "layernorm"],
    },
    "silu_mul": {
        "operation": "fused silu mul fp4 quantization",
        "aliases": ["silu", "swiglu", "gated silu", "fp4 quantization"],
    },
    "nvfp4_quantize": {
        "operation": "nvfp4 block quantization",
        "aliases": ["fp4 quantization", "nvfp4", "bf16 to fp4", "packing"],
    },
}

OBSERVATION_COMPILER_KEYS = (
    "registers_per_thread",
    "spill_stores_bytes",
    "spill_loads_bytes",
    "static_smem_bytes",
    "cmem_bytes",
    "stack_frame_bytes",
)


@dataclass
class SandboxFeedback:
    status: str
    stage: str
    route: str
    confidence: float
    speedup: float
    parent_speedup: float
    uncertainty: str
    next_action: str
    action_type: str
    evidence: list[dict] = field(default_factory=list)
    observations: dict = field(default_factory=dict)
    hypothesis_test: dict = field(default_factory=dict)
    memory: dict = field(default_factory=dict)
    rag_queries: list[str] = field(default_factory=list)
    rag_filters: dict = field(default_factory=dict)
    preserve: list[str] = field(default_factory=list)
    revert: list[str] = field(default_factory=list)
    avoid: list[str] = field(default_factory=list)
    focus: list[str] = field(default_factory=list)
    success_criteria: list[str] = field(default_factory=list)
    abort_if: list[str] = field(default_factory=list)
    error: str = ""

    def to_payload(self) -> dict:
        preserve = _unique_queries(self.preserve)
        revert = _unique_queries(self.revert)
        avoid = _unique_queries(self.avoid)
        focus = _unique_queries(self.focus)
        success_criteria = _unique_queries(self.success_criteria)
        abort_if = _unique_queries(self.abort_if)
        payload = {
            "verdict": self.status,
            "stage": self.stage,
            "route": self.route,
            "confidence": self.confidence,
            "speedup": round(self.speedup, 6),
            "parent_speedup": round(self.parent_speedup, 6),
            "uncertainty": self.uncertainty,
            "observations": self.observations,
            "hypothesis_test": self.hypothesis_test,
            "evidence": self.evidence,
            "next_action": {
                "type": self.action_type,
                "instruction": self.next_action,
                "preserve": preserve,
                "revert": revert,
                "avoid": avoid,
                "focus": focus,
                "success_criteria": success_criteria,
                "abort_if": abort_if,
            },
            "memory": self.memory,
            "rag": {
                "provider": "pinecone",
                "queries": self.rag_queries,
                "filters": self.rag_filters,
            },
        }
        if self.error:
            payload["error"] = self.error[:600]
        return payload

    def to_tool_result_json(self) -> str:
        return json.dumps(self.to_payload(), indent=2, sort_keys=True)

    def planner_summary(self) -> str:
        summary = {
            "verdict": self.status,
            "route": self.route,
            "speedup": round(self.speedup, 6),
            "parent_speedup": round(self.parent_speedup, 6),
            "uncertainty": self.uncertainty,
            "hypothesis_test": {
                "previous_hypothesis": self.hypothesis_test.get("previous_hypothesis", ""),
                "status": self.hypothesis_test.get("status", ""),
            },
            "next_action": {
                "type": self.action_type,
                "instruction": self.next_action,
                "focus": self.focus[:6],
            },
            "memory": {
                "branch_family": self.memory.get("branch_family", ""),
                "plateau_count": self.memory.get("plateau_count", 0),
                "tried_and_failed": self.memory.get("tried_and_failed", [])[:6],
                "tried_and_helped": self.memory.get("tried_and_helped", [])[:6],
            },
            "rag": {
                "queries": self.rag_queries[:6],
                "filters": self.rag_filters,
            },
        }
        if self.observations:
            summary["observations"] = {
                key: self.observations[key]
                for key in ("timing_us", "speedup", "delta_vs_parent")
                if key in self.observations
            }
        if self.evidence:
            summary["evidence"] = self.evidence[:10]
        return json.dumps(summary, indent=2, sort_keys=True)


def _first_actionable_error(error: str) -> str:
    if not error:
        return "Unknown compiler error."
    lines = [line.strip() for line in error.splitlines() if line.strip()]
    actionable = [
        line for line in lines
        if "error" in line.lower() and ("(" in line or ":" in line)
    ]
    return actionable[0] if actionable else lines[0]


def _metric_evidence(name: str, value, kind: str = "metric", unit: str | None = None) -> dict:
    item = {"kind": kind, "name": name, "value": value}
    if unit:
        item["unit"] = unit
    return item


def _delta_evidence(name: str, before, after, unit: str | None = None) -> dict:
    item = {"kind": "delta", "name": name, "before": before, "after": after}
    if unit:
        item["unit"] = unit
    return item


def _kernel_operation_phrase(kernel_type: str) -> str:
    context = KERNEL_QUERY_CONTEXT.get(kernel_type, {})
    return context.get("operation", kernel_type.replace("_", " "))


def _kernel_aliases(kernel_type: str) -> list[str]:
    context = KERNEL_QUERY_CONTEXT.get(kernel_type, {})
    aliases = list(context.get("aliases", []))
    aliases.append(kernel_type.replace("_", " "))
    return [alias for alias in aliases if alias]


def _unique_queries(queries: list[str]) -> list[str]:
    seen = set()
    ordered = []
    for query in queries:
        cleaned = " ".join(str(query).split())
        if not cleaned:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        ordered.append(cleaned)
    return ordered


def _get_candidate_attr(candidate: Any, name: str, default):
    if candidate is None:
        return default
    return getattr(candidate, name, default)


def _candidate_plan(candidate: Any) -> dict:
    plan = _get_candidate_attr(candidate, "plan_branch", {}) or {}
    return plan if isinstance(plan, dict) else {}


def _branch_family(candidate: Any) -> str:
    family = _get_candidate_attr(candidate, "branch_family", "") or ""
    if family:
        return family
    parent_strategy = _get_candidate_attr(candidate, "parent_strategy", "") or ""
    if parent_strategy:
        return parent_strategy.split("__", 1)[0]
    strategy = _get_candidate_attr(candidate, "strategy", "") or ""
    return strategy.split("__", 1)[0]


def _latest_experiment_label(candidate: Any) -> str:
    plan = _candidate_plan(candidate)
    for key in ("change_summary", "what", "goal", "bottleneck", "name"):
        value = plan.get(key)
        if value:
            return str(value)
    strategy = _get_candidate_attr(candidate, "strategy", "")
    return strategy or "latest optimization attempt"


