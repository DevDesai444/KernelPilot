from __future__ import annotations


def _infer_kernel_type(launch_signature: str, kernel_code: str) -> str:
    haystack = f"{launch_signature}\n{kernel_code}"
    if "launch_fused_add_rmsnorm_nvfp4" in haystack:
        return "add_rmsnorm"
    if "launch_silu_mul_fp4quant" in haystack:
        return "silu_mul"
    if "launch_nvfp4_quantize_bf16" in haystack:
        return "nvfp4_quantize"
    return "unknown"


def _branch_mentions(text: str, *terms: str) -> bool:
    norm = text.lower()
    return any(term in norm for term in terms)


def build_coder_prompt(
    plan_branch: dict,
    kernel_code: str,
    launch_signature: str,
    rag_context: str,
    current_profile: str = "",
    baseline_regs: int = 0,
) -> str:
    name = plan_branch.get("name", "unnamed_branch")
    goal = plan_branch.get("goal") or plan_branch.get("change_summary") or plan_branch.get("what", "")
    change_summary = plan_branch.get("change_summary") or plan_branch.get("what") or goal
    expected_signal = plan_branch.get("expected_signal", "")
    planner_notes = plan_branch.get("planner_notes", "")
    bottleneck = plan_branch.get("bottleneck", "")
    rationale = plan_branch.get("rationale", "")
    risk = plan_branch.get("risk", "")
    evidence = plan_branch.get("evidence", []) or []
    kernel_type = _infer_kernel_type(launch_signature, kernel_code)
    branch_text = " ".join(
        str(item)
        for item in (
            name,
            goal,
            change_summary,
            planner_notes,
            rationale,
            risk,
            " ".join(str(item) for item in evidence),
        )
        if item
    ).lower()

    parts = [
        f"You are the coder agent for branch \"{name}\".",
        f"Planner goal: {goal}",
        f"Required change: {change_summary}",
    ]

    if bottleneck:
        parts.append(f"Observed concern: {bottleneck}")
    if expected_signal:
        parts.append(f"Expected sandbox signal: {expected_signal}")
    if planner_notes:
        parts.append(f"Planner notes: {planner_notes}")
    if rationale:
        parts.append(f"Planner rationale: {rationale}")
    if risk:
        parts.append(f"Primary risk: {risk}")
    if evidence:
        parts.append("Planner evidence:\n- " + "\n- ".join(str(item) for item in evidence[:4]))
    if current_profile:
        parts.append(f"Current sandbox snapshot:\n{current_profile}")

    parts.append(f"Pinecone RAG context:\n{rag_context or 'No Pinecone context returned.'}")
    parts.append(f"Base kernel:\n```cuda\n{kernel_code}\n```")
    parts.append(launch_signature)

    kernel_specific_rules = []
    if kernel_type == "add_rmsnorm":
        _baseline_regs = baseline_regs or 40  # actual measured baseline register count
        _abort_regs = _baseline_regs + 8      # allow 8 regs above baseline before aborting
        _baseline_occ = 75 if _baseline_regs > 32 else 100
        kernel_specific_rules.extend([
            f"BASELINE: reference kernel compiles to {_baseline_regs} registers/thread, {_baseline_occ}% occupancy on SM100 (256-thread blocks).",
            f"On SM100, <=32 registers/thread gives 100% occupancy (8 blocks/SM). At {_baseline_regs} regs the baseline is already below 100%.",
            "ALWAYS add `__launch_bounds__(256, 8)` to this kernel. It caps registers at 32 on SM100 regardless of which branch you implement.",
            "CRITICAL THREAD WASTE BUG: Phase-2 loop is `for (int qb = tid; qb < 128; qb += 256)`. "
            "With 256 threads and 128 quant blocks, threads 128-255 execute ZERO iterations — 50% of threads are completely idle in Phase 2. "
            "This is the primary performance bottleneck. Fix it by either: "
            "(a) changing BLOCK_THREADS to 128 so every thread handles exactly one quant block, OR "
            "(b) using warp-cooperative packing (cvt_warp_fp16_to_fp4 with CVT_FP4_NUM_THREADS_PER_SF=2) so all 256 threads participate.",
            "REGISTER SOURCE: `float block_vals[NVFP4_BLOCK_SIZE]` = float[16] costs 16 registers in Phase 2. "
            "With __launch_bounds__(256, 8) the compiler spills these to stay at 32 regs total.",
            "For add+rmsnorm+fp4 on shape 128x2048, treat Phase-2 thread waste and residual_out re-read as the two primary cost centers.",
            f"Hard guard: if your first submit shows registers above {_abort_regs} and speedup below 1.05x, revert to baseline in the next turn.",
        ])
        if not _branch_mentions(branch_text, "warp", "shuffle", "reduction", "shfl", "syncthreads"):
            kernel_specific_rules.append(
                "This branch is NOT a reduction branch. Preserve the existing reduction structure instead of sneaking in warp-reduction changes."
            )
        if _branch_mentions(branch_text, "fuse", "single pass", "single-pass", "reread", "re-read", "smem cache"):
            kernel_specific_rules.extend([
                "This branch should eliminate the second global-memory read of residual_out.",
                f"Each thread owns exactly 8 elements (2048/256). If you cache them, use a float reg[8] — that adds ~8 registers above the {_baseline_regs}-reg baseline.",
                "If you use reg[8], pair with `__launch_bounds__(256, 8)` to keep total regs at 32 and maintain 100% occupancy.",
                f"Abort condition: if registers exceed {_abort_regs} and speedup stays below 1.05x, revert to baseline immediately.",
            ])
        if _branch_mentions(branch_text, "fp4", "intrinsic", "pack", "quant"):
            kernel_specific_rules.extend([
                "Do not leave the scalar float_to_nvfp4 if/else chain as the hot-path encoder if a project helper or hardware intrinsic path can replace it.",
                "HARDWARE FP4 PATH: `kernels/common/nvfp4_utils.cuh` provides `quantize_block_nvfp4(float* x, uint8_t* packed, fp8* scale)` "
                "which internally calls `__nv_cvt_float2_to_fp4x2` on sm_100a — ONE hardware instruction per pair instead of 7 scalar comparisons. "
                "Call `quantize_block_nvfp4` instead of the manual amax/scale/float_to_nvfp4 loop. "
                "#include \"nvfp4_utils.cuh\" is already available via the -I kernels/common compile flag.",
            ])

    if kernel_specific_rules:
        parts.append("Kernel-specific rules:\n- " + "\n- ".join(kernel_specific_rules))

    parts.append(
        "Rules:\n"
        "- Implement this branch only.\n"
        "- Planner owns strategy selection. Do not change branch family, optimization surface, or overall direction.\n"
        "- Do not silently add a second optimization surface. If the branch is about vectorized loads, do not also change reduction or quantization strategy unless the branch explicitly says so.\n"
        "- Preserve correctness and the launch signature.\n"
        "- submit_kernel returns evaluator JSON, not prose. Use observations, hypothesis_test, next_action, memory, uncertainty, and rag.\n"
        "- Use the attached Pinecone RAG context as the planner-approved reference set. Do not start a new Pinecone search or invent a new strategy.\n"
        "- Treat observations as ground truth. Treat hypothesis_test.status as the verdict on your last idea.\n"
        "- If the attached RAG context shows a strong production pattern, adapt that pattern as faithfully as possible within this branch.\n"
        "- Make one local experiment at a time. Follow next_action.success_criteria and next_action.abort_if.\n"
        "- If the branch seems weak, keep the adaptation minimal and let the sandbox result send control back to the planner.\n"
        "- Before submit_kernel, explain what the latest result confirmed or left uncertain and the exact code change.\n"
        "- Then call submit_kernel with the complete .cu file."
    )

    return "\n\n".join(parts)
