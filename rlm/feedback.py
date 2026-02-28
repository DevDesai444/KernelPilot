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


def _metric_support(metrics: dict, prev_inner_metrics: dict | None) -> tuple[list[str], list[str]]:
    compiler = metrics.get("_compiler", {}) if metrics else {}
    prev_compiler = (prev_inner_metrics or {}).get("_compiler", {})
    evidence_for = []
    evidence_against = []

    before_occ = (prev_inner_metrics or {}).get("sm_occupancy")
    after_occ = metrics.get("sm_occupancy")

    before_spills = prev_compiler.get("spill_stores_bytes", 0) + prev_compiler.get("spill_loads_bytes", 0)
    after_spills = compiler.get("spill_stores_bytes", 0) + compiler.get("spill_loads_bytes", 0)
    if before_spills != after_spills:
        msg = f"spill_bytes {before_spills} -> {after_spills}"
        if after_spills < before_spills:
            evidence_for.append(msg)
        else:
            evidence_against.append(msg)

    before_regs = prev_compiler.get("registers_per_thread")
    after_regs = compiler.get("registers_per_thread")
    if before_regs is not None and after_regs is not None and before_regs != after_regs:
        msg = f"registers_per_thread {before_regs} -> {after_regs}"
        if after_regs < before_regs:
            evidence_for.append(msg)
        else:
            evidence_against.append(msg)

    if before_occ is not None and after_occ is not None and before_occ != after_occ:
        msg = f"sm_occupancy {before_occ} -> {after_occ}"
        if after_occ > before_occ:
            evidence_for.append(msg)
        else:
            evidence_against.append(msg)

    return evidence_for, evidence_against


def _hypothesis_test(
    *,
    speedup: float,
    parent_speedup: float,
    metrics: dict,
    prev_inner_metrics: dict | None,
    candidate: Any,
    compile_ok: bool,
    correct: bool,
    error: str = "",
) -> dict:
    previous_hypothesis = _latest_experiment_label(candidate)
    evidence_for, evidence_against = _metric_support(metrics, prev_inner_metrics)

    if not compile_ok:
        status = "inconclusive"
        evidence_against.append(_first_actionable_error(error))
    elif not correct:
        status = "falsified"
        evidence_against.append(error[:160] or "Correctness failed.")
    elif speedup > parent_speedup + 0.02:
        status = "confirmed"
        evidence_for.append(f"speedup improved {parent_speedup:.3f}x -> {speedup:.3f}x")
    elif speedup < max(parent_speedup - 0.001, 1.0):
        status = "falsified"
        evidence_against.append(f"speedup regressed {parent_speedup:.3f}x -> {speedup:.3f}x")
    elif evidence_for:
        status = "inconclusive"
        evidence_against.append("Compiler or occupancy signals moved, but runtime did not materially improve.")
    else:
        status = "inconclusive"
        evidence_against.append("Runtime held flat, so the previous hypothesis is not yet validated.")

    return {
        "previous_hypothesis": previous_hypothesis,
        "status": status,
        "evidence_for": evidence_for[:4],
        "evidence_against": evidence_against[:4],
    }


def _next_experiment_fields(
    *,
    status: str,
    metrics: dict,
    candidate: Any,
    speedup: float,
    parent_speedup: float,
) -> tuple[str, list[str], list[str], list[str], list[str], list[str], list[str]]:
    compiler = metrics.get("_compiler", {}) if metrics else {}
    spill_total = compiler.get("spill_stores_bytes", 0) + compiler.get("spill_loads_bytes", 0)
    regs = compiler.get("registers_per_thread", 0)
    occupancy = float(metrics.get("sm_occupancy", 0) or 0.0)
    preserve = ["launch_signature", "correctness", "working_kernel_structure"]
    revert = []
    avoid = ["full_rewrite"]
    focus = []
    success_criteria = ["Correctness must hold."]
    abort_if = []

    history = list(_get_candidate_attr(candidate, "refinement_history", []) or [])
    if status == "falsified":
        latest = history[-1] if history else {}
        label = latest.get("branch") or latest.get("strategy_desc") or latest.get("strategy")
        if label:
            revert.append(str(label))

    if spill_total > 0:
        instruction = "Keep the algorithm unchanged. Remove spills or trim live state around the current working path."
        focus = ["spill elimination", "live-range trimming", "smaller per-thread state"]
        success_criteria.extend([
            "spill bytes go down or disappear.",
            "timing_us improves against the parent.",
        ])
        abort_if = [
            "register count rises while spills remain.",
            "the fix requires a launch-contract change.",
        ]
    elif regs > 40 or (regs > 0 and occupancy < 90.0):
        instruction = "Keep the fastest path intact. Reduce register pressure or recover occupancy with one local change."
        focus = ["register pressure reduction", "occupancy preservation", "launch bounds"]
        success_criteria.extend([
            "registers_per_thread drops or occupancy rises.",
            "runtime does not regress.",
        ])
        abort_if = [
            "spills appear.",
            "occupancy falls without a meaningful runtime win.",
        ]
    else:
        instruction = "Preserve the working structure and make one localized adaptation of the current best path."
        focus = ["minimal adaptation", "one localized change", "preserve working structure"]
        success_criteria.append("timing_us improves against the parent.")
        abort_if = ["multiple unrelated changes are required to explain the result."]

    if speedup < 1.0:
        avoid.append("stacking_multiple_changes")
    elif speedup <= parent_speedup + 0.02:
        avoid.append("new_branch_family")
    else:
        avoid.append("breaking_working_structure")

    return instruction, preserve, revert, avoid, focus, success_criteria[:4], abort_if[:4]


def _compiler_queries(kernel_type: str, error: str) -> list[str]:
    first = _first_actionable_error(error)
    operation = _kernel_operation_phrase(kernel_type)
    queries = [
        f"Operation: {operation}. Problem: CUDA compile error. Signature: {first}. Need: production CUDA source_code.",
        f"{operation} launch signature compile error CUDA",
    ]
    if "__syncthreads" in error:
        queries.append(f"{operation} __syncthreads divergent branch fix CUDA")
    if "undefined reference" in error.lower():
        queries.append(f"{operation} launch wrapper signature linker error CUDA")
    return _unique_queries(queries)


def build_sandbox_feedback(
    result: dict,
    parent_speedup: float,
    prev_inner_metrics: dict | None,
    kernel_type: str,
    candidate: Any = None,
) -> SandboxFeedback:
    compile_ok = result.get("compile_ok", False)
    correct = result.get("correct", False)
    speedup = float(result.get("speedup", 0.0) or 0.0)
    _binary = result.get("binary_speedup")
    routing_speedup = max(speedup, float(_binary)) if _binary is not None else speedup
    metrics = result.get("metrics", {}) or {}
    error = result.get("error", "") or ""

    if not compile_ok:
        uncertainty = "Compiler output is authoritative for this failure."
        memory = _candidate_memory(candidate, uncertainty)
        observations = _build_observations(
            compile_ok=False,
            correct=False,
            speedup=0.0,
            parent_speedup=parent_speedup,
            metrics=metrics,
            prev_inner_metrics=prev_inner_metrics,
            error=error,
        )
        hypothesis_test = _hypothesis_test(
            speedup=0.0,
            parent_speedup=parent_speedup,
            metrics=metrics,
            prev_inner_metrics=prev_inner_metrics,
            candidate=candidate,
            compile_ok=False,
            correct=False,
            error=error,
        )
        instruction = "Fix the compiler error and keep the launch contract unchanged."
        return SandboxFeedback(
            status="compile_error",
            stage="compile",
            route="fixer_with_rag",
            confidence=0.98,
            speedup=0.0,
            parent_speedup=parent_speedup,
            uncertainty=uncertainty,
            next_action=instruction,
            action_type="repair_compile",
            evidence=[
                _metric_evidence("speedup", 0.0, unit="x"),
                {"kind": "compiler_error", "name": "first_error", "value": _first_actionable_error(error)},
            ],
            observations=observations,
            hypothesis_test=hypothesis_test,
            memory=memory,
            rag_queries=_compiler_queries(kernel_type, error),
            rag_filters={"kernel_type": kernel_type, "failure_mode": "compile_error"},
            preserve=["launch_signature", "kernel_interface"],
            revert=[],
            avoid=["signature_changes", "unrelated_optimizations"],
            focus=["compiler_error", "include_paths", "wrapper_signature"],
            success_criteria=["Compilation succeeds without changing the launch contract."],
            abort_if=["The fix requires changing the kernel interface or wrapper signature."],
            error=error,
        )

    if not correct:
        uncertainty = "Correctness failures must be fixed before any performance diagnosis matters."
        memory = _candidate_memory(candidate, uncertainty)
        observations = _build_observations(
            compile_ok=True,
            correct=False,
            speedup=speedup,
            parent_speedup=parent_speedup,
            metrics=metrics,
            prev_inner_metrics=prev_inner_metrics,
            error=error,
        )
        hypothesis_test = _hypothesis_test(
            speedup=speedup,
            parent_speedup=parent_speedup,
            metrics=metrics,
            prev_inner_metrics=prev_inner_metrics,
            candidate=candidate,
            compile_ok=True,
            correct=False,
            error=error,
        )
        return SandboxFeedback(
            status="correctness_failure",
            stage="correctness",
            route="fixer_with_rag",
            confidence=0.95,
            speedup=speedup,
            parent_speedup=parent_speedup,
            uncertainty=uncertainty,
            next_action="Repair correctness before making any new optimization change.",
            action_type="repair_correctness",
            evidence=_collect_performance_evidence(speedup, parent_speedup, metrics, prev_inner_metrics),
            observations=observations,
            hypothesis_test=hypothesis_test,
            memory=memory,
            rag_queries=[
                f"{kernel_type} CUDA correctness fix",
                f"{kernel_type} kernel numerics debugging",
            ],
            rag_filters={"kernel_type": kernel_type, "failure_mode": "correctness_failure"},
            preserve=["launch_signature"],
            revert=[],
            avoid=["new_optimizations", "algorithm_changes_before_fix"],
            focus=["numerics", "bounds", "synchronization"],
            success_criteria=["Correctness passes before any new optimization is attempted."],
            abort_if=["A new performance optimization is introduced before the mismatch is fixed."],
            error=error,
        )

    uncertainty = (
        "Trust runtime delta, correctness, registers, spills, and occupancy. "
        "Treat estimated throughput figures as context only."
    )
    memory = _candidate_memory(candidate, uncertainty)
    observations = _build_observations(
        compile_ok=True,
        correct=True,
        speedup=speedup,
        parent_speedup=parent_speedup,
        metrics=metrics,
        prev_inner_metrics=prev_inner_metrics,
    )
    hypothesis_test = _hypothesis_test(
        speedup=speedup,
        parent_speedup=parent_speedup,
        metrics=metrics,
        prev_inner_metrics=prev_inner_metrics,
        candidate=candidate,
        compile_ok=True,
        correct=True,
    )
    evidence = _collect_performance_evidence(speedup, parent_speedup, metrics, prev_inner_metrics)
    queries = _performance_queries(kernel_type, metrics, memory)
    instruction, preserve, revert, avoid, focus, success_criteria, abort_if = _next_experiment_fields(
        status=hypothesis_test["status"],
        metrics=metrics,
        candidate=candidate,
        speedup=speedup,
        parent_speedup=parent_speedup,
    )

    if routing_speedup < 1.0:
        return SandboxFeedback(
            status="below_baseline",
            stage="benchmark",
            route="fixer_with_rag",
            confidence=0.88,
            speedup=speedup,
            parent_speedup=parent_speedup,
            uncertainty=uncertainty,
            next_action=instruction,
            action_type="revise_experiment",
            evidence=evidence,
            observations=observations,
            hypothesis_test=hypothesis_test,
            memory=memory,
            rag_queries=queries,
            rag_filters={"kernel_type": kernel_type, "failure_mode": "below_baseline"},
            preserve=preserve,
            revert=revert,
            avoid=avoid + ["new_branch_family"],
            focus=focus,
            success_criteria=success_criteria,
            abort_if=abort_if,
        )

    if routing_speedup > parent_speedup + 0.02:
        return SandboxFeedback(
            status="improved",
            stage="benchmark",
            route="planner_tree",
            confidence=0.86,
            speedup=speedup,
            parent_speedup=parent_speedup,
            uncertainty=uncertainty,
            next_action="Preserve the working structure and branch into one surgical follow-up experiment.",
            action_type="branch_tree",
            evidence=evidence,
            observations=observations,
            hypothesis_test=hypothesis_test,
            memory=memory,
            rag_queries=queries,
            rag_filters={"kernel_type": kernel_type, "failure_mode": "improved"},
            preserve=preserve,
            revert=revert,
            avoid=avoid + ["full_rewrite"],
            focus=focus,
            success_criteria=success_criteria,
            abort_if=abort_if,
        )

    return SandboxFeedback(
        status="plateaued_above_baseline",
        stage="benchmark",
        route="fixer_with_rag",
        confidence=0.82,
        speedup=speedup,
        parent_speedup=parent_speedup,
        uncertainty=uncertainty,
        next_action=instruction,
        action_type="targeted_refine",
        evidence=evidence,
        observations=observations,
        hypothesis_test=hypothesis_test,
        memory=memory,
        rag_queries=queries,
        rag_filters={"kernel_type": kernel_type, "failure_mode": "plateaued_above_baseline"},
        preserve=preserve,
        revert=revert,
        avoid=avoid + ["new_branch_family"],
        focus=focus,
        success_criteria=success_criteria,
        abort_if=abort_if,
    )
