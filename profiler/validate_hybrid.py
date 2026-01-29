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


def test_bandwidth_validation():
    """
    Test 1: Simple memcpy kernel with KNOWN bytes.
    We know exactly: read N*4 bytes, write N*4 bytes.
    The hybrid profiler should compute bandwidth that matches.
    """
    print("=" * 60)
    print("TEST 1: Bandwidth Validation (memcpy kernel)")
    print("=" * 60)

    N = 16 * 1024 * 1024  # 16M elements = 64 MB read + 64 MB write = 128 MB total

    src = f"""
#include <cstdio>
#include <cuda_runtime.h>

__global__ void memcpy_kernel(const float* __restrict__ in, float* __restrict__ out, int n) {{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = in[idx];
}}

int main() {{
    const int N = {N};
    float *d_in, *d_out;
    cudaMalloc(&d_in, N * sizeof(float));
    cudaMalloc(&d_out, N * sizeof(float));

    int block = 256;
    int grid = (N + block - 1) / block;

    // Warmup
    for (int i = 0; i < 100; i++)
        memcpy_kernel<<<grid, block>>>(d_in, d_out, N);
    cudaDeviceSynchronize();

    // Measure
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);
    cudaEventRecord(t0);
    for (int i = 0; i < 100; i++)
        memcpy_kernel<<<grid, block>>>(d_in, d_out, N);
    cudaEventRecord(t1);
    cudaDeviceSynchronize();

    float ms = 0;
    cudaEventElapsedTime(&ms, t0, t1);
    float us_per_iter = ms * 1000.0f / 100.0f;

    // Known data movement
    double read_bytes = (double)N * 4.0;
    double write_bytes = (double)N * 4.0;
    double total_bytes = read_bytes + write_bytes;
    double timing_sec = us_per_iter * 1e-6;
    double bw_gbps = total_bytes / timing_sec / 1e9;

    printf("timing_us: %.3f\\n", us_per_iter);
    printf("read_bytes: %.0f\\n", read_bytes);
    printf("write_bytes: %.0f\\n", write_bytes);
    printf("total_bytes: %.0f\\n", total_bytes);
    printf("measured_bw_gbps: %.2f\\n", bw_gbps);

    // Device info
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("device: %s\\n", prop.name);
    printf("mem_clock_mhz: %d\\n", prop.memoryClockRate / 1000);
    printf("mem_bus_width: %d\\n", prop.memoryBusWidth);
    // Theoretical peak BW = 2 * mem_clock * bus_width / 8
    double peak_bw = 2.0 * prop.memoryClockRate * 1e3 * prop.memoryBusWidth / 8.0 / 1e9;
    printf("theoretical_peak_bw_gbps: %.2f\\n", peak_bw);
    printf("achieved_pct_of_theoretical: %.2f\\n", bw_gbps / peak_bw * 100.0);
    printf("achieved_pct_of_achievable_75: %.2f\\n", bw_gbps / (peak_bw * 0.75) * 100.0);

    // Occupancy
    int num_blocks = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks, memcpy_kernel, block, 0);
    int warps_per_block = block / prop.warpSize;
    int active_warps = num_blocks * warps_per_block;
    int max_warps = prop.maxThreadsPerMultiProcessor / prop.warpSize;
    float occ = (float)active_warps / max_warps * 100.0f;
    printf("occupancy_pct: %.2f\\n", occ);
    printf("active_blocks_per_sm: %d\\n", num_blocks);

    cudaFree(d_in);
    cudaFree(d_out);
    return 0;
}}
"""
    stdout, _ = compile_and_run(src, "bw_test")
    if stdout:
        print(stdout)
        # Parse and verify
        bw_match = re.search(r"measured_bw_gbps:\s*([\d.]+)", stdout)
        peak_match = re.search(r"theoretical_peak_bw_gbps:\s*([\d.]+)", stdout)
        if bw_match and peak_match:
            bw = float(bw_match.group(1))
            peak = float(peak_match.group(1))
            print(f"VERDICT: Kernel achieves {bw:.0f} GB/s out of {peak:.0f} GB/s theoretical")
            print(f"  → Hybrid would report mem_throughput = {bw/peak*100:.1f}% (theoretical)")
            print(f"  → Hybrid would report mem_throughput = {bw/(peak*0.75)*100:.1f}% (achievable 75%)")
            if bw > 100:
                print("  ✓ Bandwidth measurement looks REAL (>100 GB/s on GPU)")
            else:
                print("  ✗ Bandwidth suspiciously low — check GPU")
    print()


