from __future__ import annotations


def build_fixer_prompt(
    plan_branch: dict,
    kernel_code: str,
    launch_signature: str,
    rag_context: str,
    feedback_json: str,
) -> str:
    name = plan_branch.get("name", "repair_branch")
    planner_notes = plan_branch.get("planner_notes", "")

    parts = [
        f"You are the fixer agent for branch \"{name}\".",
        "Your job is to repair compile failures, correctness failures, or below-baseline kernels.",
        "The sandbox evaluator returns strict JSON. Read the fields directly.",
        f"Evaluator JSON:\n{feedback_json}",
    ]

    if planner_notes:
        parts.append(f"Planner notes: {planner_notes}")

    parts.append(f"Pinecone RAG context:\n{rag_context or 'No Pinecone context returned.'}")
    parts.append(f"Current kernel:\n```cuda\n{kernel_code}\n```")
    parts.append(launch_signature)
    parts.append(
        "Rules:\n"
        "- Fix the concrete failure mode first.\n"
        "- Do not change strategy family or overall optimization direction while repairing.\n"
        "- If speedup is below 1.0x, question the current adaptation before making another large change.\n"
        "- Use observations as ground truth and hypothesis_test.status to decide whether the last idea was confirmed or falsified.\n"
        "- Treat uncertainty as real. Do not invent a stronger diagnosis than the measurements support.\n"
        "- Follow next_action.preserve, next_action.revert, next_action.avoid, and next_action.focus.\n"
        "- Make one local repair or follow-up experiment at a time.\n"
        "- Use next_action.success_criteria and next_action.abort_if to bound the experiment.\n"
        "- Use memory.tried_and_failed to avoid repeating dead ends.\n"
        "- Use the attached Pinecone RAG context only; do not start a new Pinecone search.\n"
        "- Before submit_kernel, explain the observed failure mode or measured result and the exact repair you are making.\n"
        "- Then call submit_kernel with the complete .cu file."
    )

    return "\n\n".join(parts)
