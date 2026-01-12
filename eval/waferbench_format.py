"""
waferbench_format.py — Format optimized kernels for WaferBench / KernelArena submission.
"""

from __future__ import annotations
import json
import time
from pathlib import Path


def format_submission(
    kernel_name: str,
    kernel_src: str,
    benchmark_results: dict,
    metadata: dict = None,
) -> dict:
    shapes_results = {
        "x".join(str(d) for d in k): v
        for k, v in benchmark_results.items()
        if isinstance(k, tuple)
    }
    return {
        "submission": {
            "kernel_name":  kernel_name,
            "timestamp":    time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "architecture": "blackwell_b200",
            "cuda_arch":    "sm_100a",
            "cuda_version": "12.6+",
        },
        "performance": {
            "geomean_speedup":   benchmark_results.get("geomean_speedup", 0.0),
            "per_shape_results": shapes_results,
        },
        "kernel_source": kernel_src,
        "metadata": metadata or {},
    }


def save_submission(submission: dict, output_dir: Path, name: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{name}_submission.json"
    cu_path   = output_dir / f"{name}_kernel.cu"
    with open(json_path, "w") as f:
        json.dump(submission, f, indent=2)
    cu_path.write_text(submission["kernel_source"])
    return json_path


def print_submission_summary(submission: dict) -> None:
    perf = submission["performance"]
    sub  = submission["submission"]
    print(f"\n{'='*60}")
    print(f"WaferBench Submission: {sub['kernel_name']}")
    print(f"Architecture: {sub['architecture']}")
    print(f"Geometric mean speedup: {perf['geomean_speedup']:.3f}x")
    print("\nPer-shape results:")
    for shape, res in perf["per_shape_results"].items():
        if isinstance(res, dict) and res.get("speedup"):
            print(f"  {shape:12s}: {res['timing_us']:.2f} us  ({res['speedup']:.3f}x)")
    print(f"{'='*60}\n")
