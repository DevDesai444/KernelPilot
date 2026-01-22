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

