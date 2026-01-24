"""
kernel_profiler.py — CUDA kernel compilation, timing, and profiling.

Handles: write .cu → nvcc compile → benchmark timing → hybrid profiling.
Compiler metrics (registers, spills) extracted via -Xptxas,-v.
"""

from __future__ import annotations
import logging
import re
import subprocess
from pathlib import Path
from typing import Optional

from .metrics import KernelMetrics, CompilerMetrics
from .hybrid_profiler import HybridProfiler

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).parent.parent


class KernelProfiler:
    """
    Manages CUDA kernel compilation, timing, and profiling.
    Workflow: write .cu → nvcc compile → CUDA event timing → hybrid profiling.

    Real data sources:
      - nvcc -Xptxas -v  → registers, spills, shared memory (exact)
      - CUDA events       → kernel timing in microseconds (measured)
      - Occupancy API     → SM occupancy (computed from register/smem usage)
    """

    def __init__(self, config: dict, hw_spec: dict = None):
        self.output_dir = Path(config.get("output", {}).get("output_dir", "outputs"))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.nvcc = "nvcc"
        self.cuda_arch = "sm_100a"
        self.nvcc_flags = [
            "-O3", f"-arch={self.cuda_arch}", "--use_fast_math", "-std=c++17",
            f"-I{PROJECT_ROOT / 'kernels' / 'common'}",
            f"-I{PROJECT_ROOT}",
        ]

        if hw_spec is None:
            import yaml
            hw_spec_path = PROJECT_ROOT / "config" / "b200_spec.yaml"
            if hw_spec_path.exists():
                with open(hw_spec_path) as f:
                    hw_spec = yaml.safe_load(f)
            else:
                hw_spec = {}
        self.hybrid = HybridProfiler(config, hw_spec)

    # ── Compilation ───────────────────────────────────────────────────────────

