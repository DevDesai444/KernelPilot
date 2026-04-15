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


