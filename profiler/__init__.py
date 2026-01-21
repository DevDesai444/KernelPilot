"""profiler — Hybrid kernel profiler for RLM beam search."""
from .kernel_profiler import KernelProfiler
from .hybrid_profiler import HybridProfiler
from .metrics import KernelMetrics, metrics_from_dict
from .roofline import operational_intensity, roofline_bound, efficiency_report

__all__ = [
    "KernelProfiler", "HybridProfiler",
    "KernelMetrics", "metrics_from_dict",
    "operational_intensity", "roofline_bound", "efficiency_report",
]
