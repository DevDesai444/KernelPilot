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


