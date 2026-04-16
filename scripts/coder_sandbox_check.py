#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from eval import flashinfer_ref
from eval.benchmark import Benchmarker
from rlm.env_loader import load_project_env
from rlm.environment import RLMEnvironment
from run import PROJECT_ROOT, WAFERBENCH_KERNELS, _lock_gpu_clocks
from search.beam_search import BeamSearch


logger = logging.getLogger(__name__)


def _warmup_gpu() -> None:
    try:
        import torch
    except ImportError:
        logger.warning("torch not installed; skipping GPU warmup")
        return

    if not torch.cuda.is_available():
        logger.warning("CUDA unavailable; skipping GPU warmup")
        return

    logger.info("Warming up GPU before coder sandbox check")
    try:
        dummy_a = torch.randn(8192, 8192, device="cuda", dtype=torch.float16)
        dummy_b = torch.randn(8192, 8192, device="cuda", dtype=torch.float16)
        for _ in range(100):
            _ = torch.matmul(dummy_a, dummy_b)
        torch.cuda.synchronize()
        del dummy_a, dummy_b
    except Exception as exc:
        logger.warning("GPU warmup failed: %s", exc)


def _find_kernel(kernel_name: str) -> dict:
    for item in WAFERBENCH_KERNELS:
        if item["name"] == kernel_name:
            return item
    available = ", ".join(item["name"] for item in WAFERBENCH_KERNELS)
    raise SystemExit(f"Unknown kernel: {kernel_name}\nAvailable: {available}")


def _measure_baseline(kernel_def: dict, config: dict, allow_reference_baseline: bool) -> tuple[float, str, bool]:
    kernel_type = kernel_def["kernel_type"]
    shape = tuple(kernel_def["shape"])
    src_path = PROJECT_ROOT / kernel_def["src"]

    baseline, baseline_source = flashinfer_ref.measure_baseline_with_source(kernel_type, shape)
    if baseline is not None:
        logger.info("FlashInfer baseline for %s: %.2f us", kernel_def["name"], baseline)
        return baseline, baseline_source, True

    if not allow_reference_baseline:
        raise RuntimeError(
            "FlashInfer baseline unavailable. Re-run with --allow-reference-baseline only for unofficial debugging."
        )

    logger.warning("FlashInfer unavailable; measuring reference kernel baseline (UNOFFICIAL)")
    benchmarker = Benchmarker(config, kernel_type=kernel_type)
    baseline = benchmarker._compile_and_time(src_path.read_text(), shape)
    if baseline is None:
        raise RuntimeError(f"Could not measure baseline for {kernel_def['name']}")
    logger.info("Reference baseline for %s: %.2f us", kernel_def["name"], baseline)
    return baseline, "reference_fallback", False


def _pick_branch(plans: list[dict], branch_index: int, branch_name: str | None) -> tuple[int, dict]:
    if branch_name:
        for idx, plan in enumerate(plans):
            if plan.get("name") == branch_name:
                return idx, plan
        names = ", ".join(plan.get("name", f"branch_{idx + 1}") for idx, plan in enumerate(plans))
        raise SystemExit(f"Unknown branch name: {branch_name}\nAvailable: {names}")

    if branch_index < 1 or branch_index > len(plans):
        raise SystemExit(f"branch-index must be in [1, {len(plans)}], got {branch_index}")
    idx = branch_index - 1
    return idx, plans[idx]


def _write_json(path: Path, payload: dict | list) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run coder agent through real sandbox profiling.")
    parser.add_argument("--kernel", default="add_rmsnorm_fp4quant_b128xh2048")
    parser.add_argument("--branch-index", type=int, default=1)
    parser.add_argument("--branch-name", type=str, default=None)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="outputs/coder_sandbox")
    parser.add_argument("--allow-reference-baseline", action="store_true",
                        help="Allow unofficial fallback to the local reference-kernel baseline.")
    parser.add_argument("--skip-gpu-warmup", action="store_true")
    parser.add_argument("--skip-planner", action="store_true", help="Use a synthetic branch instead of planner output.")
    parser.add_argument("--branch-json", type=str, default=None, help="Path to a branch JSON file to run directly.")
    args = parser.parse_args()

    load_project_env(ROOT)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if not args.skip_gpu_warmup:
        _lock_gpu_clocks()
        _warmup_gpu()

    kernel_def = _find_kernel(args.kernel)
    config_path = Path(args.config) if args.config else PROJECT_ROOT / "config" / "search_config.yaml"
    config = yaml.safe_load(config_path.read_text())
    config.setdefault("_overrides", {})

    baseline, baseline_source, official_baseline = _measure_baseline(
        kernel_def, config, allow_reference_baseline=args.allow_reference_baseline
    )
    src_path = PROJECT_ROOT / kernel_def["src"]
    env = RLMEnvironment(
        kernel_name=kernel_def["name"],
        kernel_src_path=str(src_path),
        kernel_type=kernel_def["kernel_type"],
        problem_shape=tuple(kernel_def["shape"]),
        config_path=str(config_path),
    )
    env.baseline_us_reported = baseline
    env.baseline_source = baseline_source
    env.official_baseline = official_baseline

    output_root = Path(args.output_dir)
    run_dir = output_root / f"{kernel_def['name']}_{time.strftime('%Y%m%d-%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    beam = BeamSearch(env)
    # Measure reference kernel to populate baseline_compiler_metrics (register count etc.)
    # so coder and planner receive accurate register/occupancy baseline instead of hardcoded guesses.
    problem_shape = tuple(kernel_def["shape"])
    logger.info("Measuring reference kernel compiler metrics (register count, occupancy baseline)...")
    naive_timing, baseline_cm = beam.measure_search_baseline(problem_shape)
    if baseline_cm:
        logger.info("Reference kernel: regs=%d smem=%s", baseline_cm.registers_per_thread, baseline_cm.static_smem_bytes)
    try:
        if args.branch_json:
            branch = json.loads(Path(args.branch_json).read_text())
            selected_idx = None
            plans = [branch]
        elif args.skip_planner:
            plans = [{
                "name": "manual_branch",
                "goal": "Run a direct coder sandbox check.",
                "bottleneck": "",
                "change_summary": "Implement one targeted optimization and preserve correctness.",
                "expected_signal": "Compiler succeeds and sandbox metrics improve.",
                "rag_queries": [],
                "planner_notes": "Synthetic branch for coder sandbox test.",
                "rationale": "Planner was intentionally skipped.",
                "risk": "Low confidence because no planner branch was selected.",
                "evidence": [],
                "tree_ready": False,
            }]
            branch = plans[0]
            selected_idx = 0
        else:
            plans = beam.engine.run_decompose()
            selected_idx, branch = _pick_branch(plans, args.branch_index, args.branch_name)

        _write_json(run_dir / "planner_branches.json", plans)
        _write_json(run_dir / "selected_branch.json", branch)

        problem_shape = env.problem_shapes[0]
        profile_fn = beam._make_inner_profile_fn(problem_shape, baseline)
        candidates = beam.engine.run_generate_beams(
            strategies=[branch],
            kernel_slice=env.kernel_src,
            round_num=0,
            profile_fn=profile_fn,
        )
        candidate = candidates[0]

        code_path = run_dir / "candidate.cu"
        if candidate.code:
            code_path.write_text(candidate.code)

        summary = {
            "kernel": kernel_def["name"],
            "branch_index": None if selected_idx is None else selected_idx + 1,
            "branch_name": branch.get("name"),
            "baseline_us": baseline,
            "baseline_source": baseline_source,
            "official_baseline": official_baseline,
            "compile_ok": candidate.compile_ok,
            "correct": candidate.correct,
            "speedup": candidate.speedup,
            "selected_timing_path": candidate.metrics.get("selected_timing_path", ""),
            "selected_timing_us": candidate.metrics.get("selected_timing_us"),
            "graph_timing_us": candidate.metrics.get("graph_timing_us"),
            "binary_timing_us": candidate.metrics.get("binary_timing_us"),
            "timing_delta_pct": candidate.metrics.get("timing_delta_pct"),
            "graph_speedup": candidate.metrics.get("graph_speedup"),
            "binary_speedup": candidate.metrics.get("binary_speedup"),
            "branch_family": candidate.branch_family or candidate.bottleneck,
            "feedback_route": candidate.feedback_route,
            "compile_error": candidate.compile_error,
            "metrics": candidate.metrics,
            "code_path": str(code_path) if candidate.code else None,
            "run_dir": str(run_dir),
        }
        _write_json(run_dir / "summary.json", summary)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0 if candidate.compile_ok and candidate.correct else 1
    finally:
        beam.engine.close()


if __name__ == "__main__":
    raise SystemExit(main())
