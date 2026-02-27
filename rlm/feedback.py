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


def _candidate_memory(candidate: Any, uncertainty: str) -> dict:
    history = list(_get_candidate_attr(candidate, "refinement_history", []) or [])
    failed = []
    helped = []
    plateau_count = 0

    for entry in history[-8:]:
        label = (
            entry.get("branch")
            or entry.get("strategy_desc")
            or entry.get("strategy")
            or "unnamed_change"
        )
        outcome = entry.get("outcome", "")
        if outcome == "improved":
            helped.append(label)
        elif outcome in {"compile_fail", "correctness_fail", "regression", "stagnant"}:
            failed.append(label)
        if outcome == "stagnant":
            plateau_count += 1

    return {
        "branch_family": _branch_family(candidate),
        "best_branch_family": _branch_family(candidate),
        "latest_experiment": _latest_experiment_label(candidate),
        "plateau_count": plateau_count,
        "refine_attempts": int(_get_candidate_attr(candidate, "refine_attempts", 0) or 0),
        "tried_and_failed": failed[-4:],
        "tried_and_helped": helped[-4:],
        "uncertainty": uncertainty,
    }


def _experiment_focus_terms(metrics: dict, memory: dict) -> list[str]:
    compiler = metrics.get("_compiler", {}) if metrics else {}
    focus = []

    if compiler.get("spill_stores_bytes", 0) or compiler.get("spill_loads_bytes", 0):
        focus.append("spill elimination")
    if compiler.get("registers_per_thread", 0) > 40:
        focus.extend(["register pressure", "occupancy tuning", "launch bounds"])
    occupancy = metrics.get("sm_occupancy", 0)
    if occupancy and occupancy < 90.0:
        focus.extend(["latency hiding", "occupancy tuning", "independent work per thread"])

    family = memory.get("branch_family", "")
    if family:
        focus.append(family.replace("_", " "))
    if not focus:
        focus.extend(["minimal adaptation", "preserve working structure", "localized experiment"])

    return _unique_queries(focus)


def _build_targeted_query(kernel_type: str, focus_terms: list[str], memory: dict) -> str:
    operation = _kernel_operation_phrase(kernel_type)
    aliases = ", ".join(_kernel_aliases(kernel_type)[:4])
    optimizations = ", ".join(focus_terms[:4]) or "instruction mix"
    latest = memory.get("latest_experiment", "")
    return (
        f"Operation: {operation}. "
        f"Aliases: {aliases}. "
        f"Current experiment: {latest}. "
        f"Optimizations: {optimizations}. "
        f"Need: production CUDA kernel source_code."
    )


def _performance_queries(kernel_type: str, metrics: dict, memory: dict) -> list[str]:
    focus_terms = _experiment_focus_terms(metrics, memory)
    operation = _kernel_operation_phrase(kernel_type)
    queries = [_build_targeted_query(kernel_type, focus_terms, memory)]

    latest = memory.get("latest_experiment", "")
    family = memory.get("branch_family", "")
    if latest:
        queries.append(f"{operation} {latest} CUDA source code")
    if family:
        queries.append(f"{operation} {family.replace('_', ' ')} production CUDA source code")
    if focus_terms:
        queries.append(f"{operation} {' '.join(focus_terms[:2])} CUDA source code")

    compiler = metrics.get("_compiler", {}) if metrics else {}
    spill_total = compiler.get("spill_stores_bytes", 0) + compiler.get("spill_loads_bytes", 0)
    regs = compiler.get("registers_per_thread", 0)
    occupancy = float(metrics.get("sm_occupancy", 0) or 0.0)
    reg_limit = 96
    occ_limit = 75.0
    if kernel_type == "add_rmsnorm":
        reg_limit = 40
        occ_limit = 90.0
    if spill_total > 0:
        queries.append(f"{operation} reduce register spills live range CUDA")
    elif regs > reg_limit or (regs > 0 and occupancy < occ_limit):
        queries.append(f"{operation} lower register pressure occupancy CUDA")
    else:
        queries.append(f"{operation} minimal adaptation best production kernel CUDA")

    return _unique_queries(queries)


def _collect_performance_evidence(
    speedup: float,
    parent_speedup: float,
    metrics: dict,
    prev_inner_metrics: dict | None,
) -> list[dict]:
    evidence = [
        _metric_evidence("speedup", round(speedup, 6), unit="x"),
        _delta_evidence("speedup", round(parent_speedup, 6), round(speedup, 6), unit="x"),
    ]

    for key, unit in (
        ("duration_us", "us"),
        ("sm_occupancy", "%"),
    ):
        value = metrics.get(key)
        if value is not None and value != 0:
            evidence.append(_metric_evidence(key, round(float(value), 3), unit=unit))

    compiler = metrics.get("_compiler", {}) if metrics else {}
    prev_compiler = (prev_inner_metrics or {}).get("_compiler", {})
    for key in OBSERVATION_COMPILER_KEYS:
        value = compiler.get(key)
        if value is None:
            continue
        evidence.append(_metric_evidence(key, value, kind="compiler"))
        if key in prev_compiler and prev_compiler.get(key) != value:
            evidence.append(_delta_evidence(key, prev_compiler.get(key), value))

    return evidence[:16]


def _build_observations(
    compile_ok: bool,
    correct: bool,
    speedup: float,
    parent_speedup: float,
    metrics: dict,
    prev_inner_metrics: dict | None,
    error: str = "",
) -> dict:
    observations = {
        "compile_ok": bool(compile_ok),
        "correct": bool(correct),
        "speedup": round(speedup, 6),
        "parent_speedup": round(parent_speedup, 6),
    }

    duration_us = metrics.get("duration_us")
    if duration_us:
        observations["timing_us"] = round(float(duration_us), 3)
    for key in ("sm_occupancy",):
        value = metrics.get(key)
        if value:
            observations[key] = round(float(value), 3)

    delta_vs_parent = {}
    if parent_speedup:
        delta_vs_parent["speedup"] = round(speedup - parent_speedup, 6)

    prev_metrics = prev_inner_metrics or {}
    for key in ("duration_us", "sm_occupancy"):
        before = prev_metrics.get(key)
        after = metrics.get(key)
        if before is not None and after is not None and before != after:
            delta_vs_parent[key] = round(float(after) - float(before), 3)

    compiler = metrics.get("_compiler", {}) if metrics else {}
    prev_compiler = prev_metrics.get("_compiler", {}) if prev_metrics else {}
    for key in OBSERVATION_COMPILER_KEYS:
        before = prev_compiler.get(key)
        after = compiler.get(key)
        if before is not None and after is not None and before != after:
            delta_vs_parent[key] = after - before

    if delta_vs_parent:
        observations["delta_vs_parent"] = delta_vs_parent
    if error:
        observations["error"] = error[:240]
    return observations


