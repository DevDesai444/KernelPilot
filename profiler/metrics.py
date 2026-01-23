"""
metrics.py — Kernel metric definitions and parsing utilities.
"""

from __future__ import annotations
from dataclasses import dataclass


@dataclass
class KernelMetrics:
    """Parsed profiler metrics for one kernel execution."""
    mem_throughput_pct:     float = 0.0
    l1_throughput_pct:      float = 0.0
    l2_throughput_pct:      float = 0.0
    compute_throughput_pct: float = 0.0
    fp32_throughput_pct:    float = 0.0
    sm_occupancy:           float = 0.0
    achieved_occupancy:     float = 0.0
    stall_memory:           float = 0.0
    stall_barrier:          float = 0.0
    stall_no_instruction:   float = 0.0
    stall_mio_throttle:     float = 0.0
    l2_hit_rate:            float = 0.0
    l2_read_sectors:        float = 0.0
    l2_write_sectors:       float = 0.0
    dram_read_bytes:        float = 0.0
    dram_write_bytes:       float = 0.0
    dram_read_bw_gbps:      float = 0.0
    inst_executed:          float = 0.0
    inst_fp32:              float = 0.0
    inst_fp16:              float = 0.0
    inst_load:              float = 0.0
    inst_store:             float = 0.0
    elapsed_cycles:         float = 0.0
    active_cycles:          float = 0.0
    duration_us:            float = 0.0
    graph_timing_us:        float = 0.0
    binary_timing_us:       float = 0.0
    timing_delta_pct:       float = 0.0
    graph_speedup:          float = 0.0
    binary_speedup:         float = 0.0
    speedup:                float = 1.0

    def to_dict(self) -> dict:
        import dataclasses
        d = dataclasses.asdict(self)
        # Include compiler metrics if available
        if hasattr(self, '_compiler_metrics') and self._compiler_metrics:
            d['_compiler'] = dataclasses.asdict(self._compiler_metrics)
        return d

    def summary_str(self) -> str:
        return (
            f"timing={self.duration_us:.1f}us "
            f"occupancy={self.sm_occupancy:.1f}% "
            f"speedup={self.speedup:.3f}x"
        )


@dataclass
class CompilerMetrics:
    """Metrics extracted directly from compiler output."""
    # From nvcc -Xptxas -v
    registers_per_thread: int = 0
    spill_stores_bytes: int = 0
    spill_loads_bytes: int = 0
    static_smem_bytes: int = 0
    cmem_bytes: int = 0
    stack_frame_bytes: int = 0

    @property
    def has_spills(self) -> bool:
        return self.spill_stores_bytes > 0 or self.spill_loads_bytes > 0

    def summary_str(self) -> str:
        parts = [f"regs={self.registers_per_thread}"]
        if self.has_spills:
            parts.append(f"spill_ld={self.spill_loads_bytes}B spill_st={self.spill_stores_bytes}B")
        parts.append(f"smem={self.static_smem_bytes}B")
        return " ".join(parts)


def metrics_from_dict(d: dict) -> KernelMetrics:
    m = KernelMetrics()
    for field_name, value in d.items():
        if field_name == '_compiler':
            continue  # handled below
        if hasattr(m, field_name):
            setattr(m, field_name, float(value))
    # Reconstruct compiler metrics so to_dict() preserves _compiler
    compiler_data = d.get('_compiler')
    if compiler_data:
        cm = CompilerMetrics()
        for k, v in compiler_data.items():
            if hasattr(cm, k):
                setattr(cm, k, v)
        m._compiler_metrics = cm
    return m
