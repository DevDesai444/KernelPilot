"""
roofline.py — Roofline model analysis for B200.
"""

from __future__ import annotations
from .metrics import KernelMetrics


def operational_intensity(metrics: KernelMetrics) -> float:
    """Compute OI = FLOPs / DRAM bytes."""
    total_bytes = metrics.dram_read_bytes + metrics.dram_write_bytes
    flops = metrics.inst_fp32 * 2 + metrics.inst_fp16
    if total_bytes <= 0:
        return 0.0
    return flops / total_bytes


def roofline_bound(oi: float, hw_spec: dict) -> str:
    """Determine if kernel is memory or compute bound at given OI."""
    peak_flops = hw_spec["compute"]["fp32_tflops"] * 1e12
    peak_bw    = hw_spec["memory"]["hbm_bandwidth_tbs"] * 1e12
    ridge_oi   = peak_flops / peak_bw
    return "memory_bound" if oi < ridge_oi else "compute_bound"


def peak_performance(oi: float, hw_spec: dict, dtype: str = "fp32") -> float:
    """Roofline peak performance (GFLOP/s) for given OI."""
    dtype_map = {
        "fp32":  "fp32_tflops",
        "fp16":  "fp16_tflops",
        "bf16":  "bf16_tflops",
        "nvfp4": "nvfp4_tflops",
    }
    peak_tflops = hw_spec["compute"].get(dtype_map.get(dtype, "fp32_tflops"), 67.0)
    peak_bw_bs  = hw_spec["memory"]["hbm_bandwidth_tbs"] * 1e12
    return min(oi * peak_bw_bs, peak_tflops * 1e12) / 1e9


def efficiency_report(metrics: KernelMetrics, hw_spec: dict) -> str:
    oi    = operational_intensity(metrics)
    bound = roofline_bound(oi, hw_spec)
    peak  = peak_performance(oi, hw_spec)
    return (
        f"\n=== Roofline Analysis ===\n"
        f"Operational Intensity: {oi:.2f} FLOP/byte\n"
        f"Bound: {bound}\n"
        f"Peak achievable (fp32): {peak:.1f} GFLOP/s\n"
        f"HW limits: {hw_spec['memory']['hbm_bandwidth_tbs']} TB/s memory | "
        f"{hw_spec['compute']['fp32_tflops']} TFLOP/s FP32 | "
        f"{hw_spec['compute']['nvfp4_tflops']} TFLOP/s NVFP4\n"
        f"Memory BW:    {metrics.mem_throughput_pct:.1f}% of peak\n"
        f"Compute:      {metrics.compute_throughput_pct:.1f}% of peak\n"
        f"Speedup:      {metrics.speedup:.3f}x\n"
    )
