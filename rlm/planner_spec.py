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


def build_root_planner_spec(
    *,
    kernel_type: str,
    operation: str,
    aliases: list[str],
    problem_shape: tuple,
    kernel_src: str,
    baseline_context: str,
    rag_context: str,
    branch_count: int,
) -> PlannerSpec:
    constraints = [
        "Use Pinecone RAG context when naming the concrete technique to test.",
        "Prefer production-style CUDA implementation patterns over vague advice.",
        "Prefer exact operation matches over kernels that only share datatype or hardware family.",
        "Do not propose full rewrites in every branch; vary the plan surface.",
        "Analyze kernel_src for thread utilization: check every loop where the iteration bound may be less than blockDim.x — threads beyond the bound are completely idle and waste SM resources.",
        "Analyze kernel_src for compute bottlenecks: look for scalar if/else chains or manual encode loops that could be replaced by hardware intrinsics available on sm_100a.",
        "Analyze kernel_src for redundant global memory passes: identify data written to global memory in one phase and re-read in a later phase that could be cached in registers instead.",
        "Analyze kernel_src for memory access patterns: identify loads and stores that could be widened to float4/uint4 (128-bit) for better memory throughput.",
    ]
    if kernel_type == "add_rmsnorm":
        constraints.extend([
            "For fused add+rmsnorm+fp4, prioritize eliminating the Phase-2 residual_out reread before minor local tweaks.",
            "Do not spend multiple root branches on reduction-only ideas; treat warp-reduction-only branches as secondary unless paired with a larger memory-path improvement.",
            "The reference kernel baseline_context shows the actual register count and occupancy — use those numbers, not assumed values.",
        ])
    return PlannerSpec(
        mode="root",
        kernel_type=kernel_type,
        operation=operation,
        aliases=aliases,
        problem_shape=problem_shape,
        branch_count=branch_count,
        kernel_src=kernel_src,
        rag_context=rag_context,
        objective="Generate root branches that each test one distinct optimization hypothesis.",
        baseline_context=baseline_context,
        success_criteria=[
            "Each branch should make one focused adaptation only.",
            "Each branch is testable in one sandbox iteration.",
            "The set of branches is diverse, not repeated variations.",
        ],
        constraints=constraints,
    )


def build_tree_planner_spec(
    *,
    kernel_type: str,
    operation: str,
    aliases: list[str],
    problem_shape: tuple,
    kernel_src: str,
    feedback_summary: str,
    rag_context: str,
    branch_count: int,
    parent_strategy: str,
    parent_speedup: float,
) -> PlannerSpec:
    constraints = [
        "Use the sandbox feedback and Pinecone RAG context directly.",
        "Prefer child branches that preserve the closest retrieved source pattern from the working family.",
        "Do not repeat the parent plan with different wording.",
        "Avoid full rewrites and avoid stacking multiple risky changes into one child.",
    ]
    if kernel_type == "add_rmsnorm":
        constraints.extend([
            "For add+rmsnorm+fp4, prefer child branches that remove the second global-memory pass or improve FP4 packing over reduction-only tweaks.",
            "Treat register growth above the current working regime as a risk; preserve occupancy unless the runtime gain is clearly worth it.",
        ])
    return PlannerSpec(
        mode="tree",
        kernel_type=kernel_type,
        operation=operation,
        aliases=aliases,
        problem_shape=problem_shape,
        branch_count=branch_count,
        kernel_src=kernel_src,
        rag_context=rag_context,
        objective="Expand the working parent into surgical child branches without breaking the current structure.",
        feedback_summary=feedback_summary,
        parent_strategy=parent_strategy,
        parent_speedup=parent_speedup,
        success_criteria=[
            "Each child branch should try a different minimal follow-up adaptation.",
            "Children preserve the parent's working structure.",
            "Children are small enough to validate in one sandbox turn.",
        ],
        constraints=constraints,
    )
