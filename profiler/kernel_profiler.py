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

    def compile_kernel(
        self,
        kernel_src: str,
        harness_src: str,
        output_name: str = "kernel_bench",
    ) -> tuple:
        """Compile kernel + harness. Returns (success, error_msg, binary_path, CompilerMetrics).
        Extracts compiler metrics (registers, spills) via -Xptxas,-v."""
        build_dir = self.output_dir / "build"
        build_dir.mkdir(parents=True, exist_ok=True)

        # OS file name limit is 255 bytes; truncate long combined-branch names
        safe_name = output_name[:80] if len(output_name) > 80 else output_name
        kernel_file = build_dir / f"{safe_name}.cu"
        binary_file = build_dir / safe_name
        kernel_file.write_text(kernel_src + "\n\n" + harness_src)

        cmd = [self.nvcc] + self.nvcc_flags + ["-Xptxas", "-v", str(kernel_file), "-o", str(binary_file)]
        logger.info("Compiling: %s", " ".join(cmd[:4]) + " ...")

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        except subprocess.TimeoutExpired:
            logger.warning("Compilation timed out after 120s for %s", output_name)
            return False, "Compilation timed out after 120s", binary_file, None
        if result.returncode != 0:
            error_lines = [l for l in result.stderr.splitlines()
                           if not l.strip().startswith("ptxas info")
                           and "bytes stack frame" not in l
                           and "bytes spill stores" not in l
                           and "bytes spill loads" not in l
                           and "bytes cmem" not in l
                           and "bytes smem" not in l
                           and "Remark: The warnings can be suppressed" not in l]
            error_msg = "\n".join(error_lines).strip() or result.stderr.strip()
            logger.warning("Compilation failed:\n%s", error_msg[:800])
            return False, error_msg, binary_file, None

        # Parse compiler metrics from ptxas verbose output
        compiler_metrics = self._parse_ptxas_verbose(result.stderr)

        logger.info("Compiler metrics: %s", compiler_metrics.summary_str())
        return True, "", binary_file, compiler_metrics

    def _parse_ptxas_verbose(self, stderr: str) -> CompilerMetrics:
        """Parse nvcc -Xptxas,-v output for register count, spills, shared memory."""
        cm = CompilerMetrics()

        reg_match = re.search(r'Used\s+(\d+)\s+registers', stderr)
        if reg_match:
            cm.registers_per_thread = int(reg_match.group(1))

        smem_match = re.search(r'(\d+)\s+bytes\s+smem', stderr)
        if smem_match:
            cm.static_smem_bytes = int(smem_match.group(1))

        cmem_match = re.search(r'(\d+)\s+bytes\s+cmem\[0\]', stderr)
        if cmem_match:
            cm.cmem_bytes = int(cmem_match.group(1))

        stack_match = re.search(r'(\d+)\s+bytes\s+stack\s+frame', stderr)
        if stack_match:
            cm.stack_frame_bytes = int(stack_match.group(1))

        spill_st = re.search(r'(\d+)\s+bytes\s+spill\s+stores', stderr)
        if spill_st:
            cm.spill_stores_bytes = int(spill_st.group(1))

        spill_ld = re.search(r'(\d+)\s+bytes\s+spill\s+loads', stderr)
        if spill_ld:
            cm.spill_loads_bytes = int(spill_ld.group(1))

        return cm

    # ── Profiling ─────────────────────────────────────────────────────────────

