"""
validate_hybrid.py — Validate hybrid profiler against known-good benchmarks.

Run on the GPU server to verify hybrid metrics are real, not fabricated.
Uses kernels with KNOWN data movement so we can verify the math.

Usage:
    python -m profiler.validate_hybrid
"""

import subprocess
import tempfile
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
NVCC = "nvcc"
ARCH = "sm_100a"


def compile_and_run(cuda_src: str, name: str = "validate") -> tuple:
    """Compile CUDA source, run it, return (stdout, binary_path)."""
    with tempfile.NamedTemporaryFile(suffix=".cu", mode="w", delete=False, dir="/tmp") as f:
        f.write(cuda_src)
        src_path = f.name

    bin_path = src_path.replace(".cu", f"_{name}")
    cmd = [NVCC, "-O3", f"-arch={ARCH}", "--use_fast_math", "-std=c++17",
           src_path, "-o", bin_path]
    try:
        comp = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired:
        print("COMPILE ERROR: timed out after 120s")
        return None, None
    if comp.returncode != 0:
        print(f"COMPILE ERROR: {comp.stderr[:500]}")
        return None, None

    try:
        run = subprocess.run([bin_path], capture_output=True, text=True, timeout=30)
    except subprocess.TimeoutExpired:
        print("RUN ERROR: timed out after 30s")
        return None, bin_path
    if run.returncode != 0:
        print(f"RUN ERROR: {run.stderr[:500]}")
        return None, bin_path

    return run.stdout, bin_path


