"""
hybrid_profiler.py — Lightweight profiler using CUDA Events + Occupancy API.

Real data sources:
  1. CUDA Event timing   -> duration_us, speedup  (measured by benchmark harness)
  2. CUDA Occupancy API  -> sm_occupancy (compiled query program)
  3. Theoretical occupancy fallback (from register count + block size + shared mem)
  4. Compiler metrics     -> registers, spills, smem (from nvcc -Xptxas -v)
  5. Estimated roofline math -> mem_throughput_pct (from timing + transfer bytes; diagnostic only)
"""

from __future__ import annotations
import logging
import math
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

from .metrics import KernelMetrics, CompilerMetrics

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).parent.parent


# ── Helper ────────────────────────────────────────────────────────────────

def _estimate_shared_memory(kernel_src: str) -> int:
    """Estimate shared memory usage from __shared__ declarations."""
    total = 0
    type_sizes = {
        "float": 4, "double": 8, "int": 4, "unsigned": 4,
        "half": 2, "__half": 2, "__nv_bfloat16": 2,
        "float4": 16, "float2": 8, "int4": 16,
        "char": 1, "uint8_t": 1, "int8_t": 1,
        "uint32_t": 4, "int32_t": 4, "uint16_t": 2,
    }
    for m in re.finditer(r'__shared__\s+(\w+)\s+\w+\[([^\]]+)\]', kernel_src):
        dtype, size_expr = m.group(1), m.group(2).strip()
        elem_size = type_sizes.get(dtype, 4)
        try:
            total += int(size_expr) * elem_size
        except ValueError:
            total += 1024 * elem_size
    if "extern __shared__" in kernel_src:
        total = max(total, 4096)
    return total


# ── Main Profiler Class ───────────────────────────────────────────────────

class HybridProfiler:
    """
    Computes kernel metrics from real data sources.

    Returns: timing, speedup, SM occupancy, an estimated memory-throughput percentage, and compiler metrics.
    """

    def __init__(self, config: dict, hw_spec: dict):
        self.config = config
        self.hw_spec = hw_spec

        sm = hw_spec.get("sm", {})
        mem = hw_spec.get("memory", {})

        self.sm_count = sm.get("count", 148)
        self.max_warps_per_sm = sm.get("max_warps_per_sm", 64)
        self.max_blocks_per_sm = sm.get("max_blocks_per_sm", 32)
        self.warp_size = sm.get("warp_size", 32)
        self.max_threads_per_sm = sm.get("max_threads_per_sm", 2048)
        self.shared_mem_per_sm = mem.get("shared_memory_per_sm_kb", 228) * 1024

        self.nvcc = "nvcc"
        self.cuda_arch = "sm_100a"
        self.nvcc_flags = [
            "-O3", f"-arch={self.cuda_arch}", "--use_fast_math", "-std=c++17",
            f"-I{PROJECT_ROOT / 'kernels' / 'common'}",
            f"-I{PROJECT_ROOT}",
        ]

    # ── Main Entry Point ───────────────────────────────────────────────────

    def profile(
        self,
        kernel_src: str,
        timing_us: float,
        kernel_type: str,
        problem_shape: tuple,
        baseline_us: float = 0.0,
        compiler_metrics: Optional[CompilerMetrics] = None,
    ) -> Optional[KernelMetrics]:
        if timing_us <= 0:
            return None

        cm = compiler_metrics or CompilerMetrics()

        # ── 1. Occupancy (real from CUDA API, or theoretical fallback) ────
        block_size, shared_mem = self._parse_launch_config(kernel_src)
        if cm.static_smem_bytes > 0:
            shared_mem = cm.static_smem_bytes

        gpu_occupancy = self._query_occupancy_from_binary(kernel_src, block_size, shared_mem)
        if gpu_occupancy is None:
            gpu_occupancy = self._compute_theoretical_occupancy(
                block_size, shared_mem, cm.registers_per_thread
            )

        # ── 2. Speedup ───────────────────────────────────────────────────
        speedup = baseline_us / timing_us if timing_us > 0 and baseline_us > 0 else 1.0

        # ── 3. Compute estimated mem_throughput_pct from roofline math ───
        mem_throughput_pct = 0.0
        peak_bw_tbs = self.hw_spec.get("memory", {}).get("hbm_bandwidth_tbs", 8.0)
        total_bytes = self._estimate_transfer_bytes(kernel_type, problem_shape)
        if total_bytes > 0 and timing_us > 0:
            achieved_bw = total_bytes / timing_us / 1e6  # TB/s
            mem_throughput_pct = achieved_bw / peak_bw_tbs * 100

        # ── 4. Build KernelMetrics ────────────────────────────────────────
        metrics = KernelMetrics(
            sm_occupancy=round(gpu_occupancy, 2),
            achieved_occupancy=round(gpu_occupancy * self.max_warps_per_sm / 100.0, 2),
            duration_us=timing_us,
            speedup=speedup,
            mem_throughput_pct=round(mem_throughput_pct, 2),
        )

        # Attach compiler metrics for the reflection prompt
        metrics._compiler_metrics = cm

        # ── 5. Log real data ──────────────────────────────────────────────
        log_parts = [
            f"occ={gpu_occupancy:.1f}%",
            f"timing={timing_us:.1f}us",
            f"speedup={speedup:.3f}x",
            f"est_mem_bw={mem_throughput_pct:.1f}%",
        ]
        if cm.registers_per_thread > 0:
            log_parts.append(f"regs={cm.registers_per_thread}")
        if cm.has_spills:
            log_parts.append(f"SPILLS={cm.spill_stores_bytes}B")

        logger.info("Hybrid profiler: %s", " ".join(log_parts))
        return metrics

    # ── Transfer Bytes Estimation ─────────────────────────────────────────

    @staticmethod
