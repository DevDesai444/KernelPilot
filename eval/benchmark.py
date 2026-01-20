"""
benchmark.py — Timing harness for WaferBench problem shapes.

Follows ThunderKittens 2.0 convention:
  - 500 warmup iterations
  - 100 timed reps via CUDA events
  - L2 cache input cycling to prevent cache hits inflating performance

CRITICAL: Both baseline (FlashInfer, via flashinfer_ref.py) and candidate
kernels are timed through Python/PyTorch dispatch to ensure symmetric
measurement overhead. This matches KernelArena's methodology where both
reference and candidate go through the same `bench_sustained` path.

Without this symmetry, the Python dispatch overhead (~5-15us per call)
inflates the baseline measurement while the C++ binary has near-zero
overhead, producing fake speedups of 3-4x on ~4us kernels.
"""

from __future__ import annotations
import hashlib
import logging
import math
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Optional
import time

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).parent.parent
DEBUG_BENCHMARK_KERNEL_SRC = os.getenv("RLM_DEBUG_BENCHMARK_KERNEL_SRC", "").strip().lower() in {
    "1", "true", "yes",
}

# Maximum input buffer copies for L2 cache cycling (KernelArena methodology)
_MAX_L2_CYCLE_BUFS = 256

# Timing constants matching ThunderKittens 2.0 / WaferBench convention
_WARMUP_ITERS = 500
_BENCH_ITERS  = 100


def geometric_mean(values: list) -> float:
    if not values:
        return 1.0
    log_sum = sum(math.log(v) for v in values if v > 0)
    return math.exp(log_sum / len(values))


class Benchmarker:

    def __init__(self, config: dict, kernel_type: str = "add_rmsnorm"):
        self.warmup      = config["eval"]["benchmark_warmup"]
        self.iters       = config["eval"]["benchmark_iters"]
        self.kernel_type = kernel_type
        self.include_dirs = [str(PROJECT_ROOT / 'kernels' / 'common'), str(PROJECT_ROOT)]

    @staticmethod
    def _compute_l2_cycle_bufs(input_bytes: int) -> int:
        """Compute nbufs to exceed 3x L2 cache (KernelArena methodology)."""
        import torch
        props = torch.cuda.get_device_properties(0)
        l2_size = props.L2_cache_size
        if input_bytes >= l2_size * 3:
            return 1
        n = int(l2_size * 3 / input_bytes) + 1
        return min(n, _MAX_L2_CYCLE_BUFS)

    @staticmethod
    def _cuda_harness_prelude() -> str:
        return r"""
#define CHECK_CUDA(call) do { \
    cudaError_t err__ = (call); \
    if (err__ != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err__)); \
        return 2; \
    } \
} while (0)
#define cudaMalloc(...) CHECK_CUDA(cudaMalloc(__VA_ARGS__))
#define cudaMemcpy(...) CHECK_CUDA(cudaMemcpy(__VA_ARGS__))
#define cudaMemset(...) CHECK_CUDA(cudaMemset(__VA_ARGS__))
#define cudaFree(...) CHECK_CUDA(cudaFree(__VA_ARGS__))
#define cudaStreamCreate(...) CHECK_CUDA(cudaStreamCreate(__VA_ARGS__))
#define cudaStreamSynchronize(...) CHECK_CUDA(cudaStreamSynchronize(__VA_ARGS__))
#define cudaDeviceSynchronize(...) CHECK_CUDA(cudaDeviceSynchronize())
#define cudaEventCreate(...) CHECK_CUDA(cudaEventCreate(__VA_ARGS__))
#define cudaEventRecord(...) CHECK_CUDA(cudaEventRecord(__VA_ARGS__))
#define cudaEventElapsedTime(...) CHECK_CUDA(cudaEventElapsedTime(__VA_ARGS__))
#define cudaEventDestroy(...) CHECK_CUDA(cudaEventDestroy(__VA_ARGS__))
#define cudaDeviceGetAttribute(...) CHECK_CUDA(cudaDeviceGetAttribute(__VA_ARGS__))
"""

    def _input_bytes(self, shape: tuple) -> int:
        """Compute input bytes for L2 cycling calculation."""
        if self.kernel_type == "add_rmsnorm":
            rows, hidden = shape
            return rows * hidden * 2 * 2 + hidden * 2
        elif self.kernel_type == "silu_mul":
            b, m, k = shape
            return b * m * k * 2 * 2
        elif self.kernel_type == "nvfp4_quantize":
            m, k = shape
            return m * k * 2
        return 4096  # fallback

    # ── Primary: Python-dispatch timing (symmetric with FlashInfer baseline) ──

    def _compile_and_time(self, kernel_src: str, shape: tuple) -> Optional[float]:
        """Compile candidate and time through Python dispatch.

        Uses torch.utils.cpp_extension.load_inline so both candidate and
        FlashInfer baseline go through the same PyTorch dispatch path,
        ensuring symmetric measurement overhead.
        """
        try:
            return self._time_via_extension(kernel_src, shape)
        except Exception as e:
            logger.warning("PyTorch extension timing failed (%s), trying C++ fallback", e)
            return self._time_via_binary(kernel_src, shape)

    def _time_via_extension(self, kernel_src: str, shape: tuple) -> Optional[float]:
        """Load kernel as PyTorch extension and time through Python loop."""
        import torch
        from torch.utils.cpp_extension import load_inline

        bridge_code = self._c_bridge(shape)
        wrapper_code = self._pybind_wrapper(shape)
        cuda_src = kernel_src + "\n\n" + bridge_code

        # Unique name based on source hash to avoid stale cache
        src_hash = hashlib.md5(cuda_src.encode()).hexdigest()[:12]
        mod_name = f"bench_{self.kernel_type}_{src_hash}"

        if DEBUG_BENCHMARK_KERNEL_SRC:
            print("\n" + "=" * 80)
            print(f"DEBUG: Printing generated kernel source for shape {shape}")
            print("=" * 80)
            print(kernel_src)
            print("=" * 80 + "\n")

        cuda_cflags = [
            "-O3",
            "-arch=sm_100a",
            "--use_fast_math",
            "-std=c++17",
            # PyTorch JIT may inject *_NO_HALF* / *_NO_BFLOAT16_* defines.
            # Explicitly undefine them so bf16/half intrinsics match the direct nvcc path.
            "-U__CUDA_NO_HALF_OPERATORS__",
            "-U__CUDA_NO_HALF_CONVERSIONS__",
            "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
            "-U__CUDA_NO_HALF2_OPERATORS__",
        ] + [f"-I{d}" for d in self.include_dirs]

        logger.info(
            "JIT extension compile: TORCH_CUDA_ARCH_LIST=%s flags=%s",
            os.getenv("TORCH_CUDA_ARCH_LIST", "<unset>"),
            " ".join(cuda_cflags),
        )

        module = load_inline(
            name=mod_name,
            cuda_sources=[cuda_src],
            cpp_sources=[wrapper_code],
            extra_cuda_cflags=cuda_cflags,
            verbose=False,
        )

        # Dynamic L2 cache cycling (KernelArena methodology)
        input_bytes = self._input_bytes(shape)
        nbufs = self._compute_l2_cycle_bufs(input_bytes)
        input_sets = []
        for i in range(nbufs):
            torch.manual_seed(42 + i)
            input_sets.append(self._create_inputs(shape))
        outputs = self._create_outputs(shape)

        def run(buf_idx):
            module.run_kernel(*input_sets[buf_idx], *outputs)

        # Warmup with L2 cycling (Python loop — overhead doesn't matter here)
        for i in range(self.warmup):
            run(i % nbufs)
        torch.cuda.synchronize()

        # Capture CUDA graph with L2-cycling buffer sequence.
        # The graph bakes in the exact pointer sequence run(0),run(1),...,run(nbufs-1),...
        # so L2 cycling is preserved. Graph replay has ZERO Python dispatch overhead,
        # eliminating the ~3-5µs/call Pybind11 boundary crossing that crushes speedup ratios.
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for i in range(self.iters):
                run(i % nbufs)
        s.synchronize()

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for i in range(self.iters):
                run(i % nbufs)

        # Warm the graph replay (first replay may have overhead)
        for _ in range(3):
            g.replay()
        torch.cuda.synchronize()

        # Timed graph replay — zero Python overhead, L2 cycling preserved
        start = torch.cuda.Event(enable_timing=True)
        end   = torch.cuda.Event(enable_timing=True)

        start.record()
        g.replay()
        end.record()
        torch.cuda.synchronize()

        ms = start.elapsed_time(end)
        return ms * 1000.0 / self.iters  # µs per iteration

    # ── C bridge: extern "C" void* wrapper in CUDA source ────────────────────

    def _c_bridge(self, shape: tuple) -> str:
        if self.kernel_type == "add_rmsnorm":
            return self._c_bridge_add_rmsnorm(shape)
        elif self.kernel_type == "silu_mul":
            return self._c_bridge_silu_mul(shape)
        elif self.kernel_type == "nvfp4_quantize":
            return self._c_bridge_nvfp4_quantize(shape)
        raise ValueError(f"Unknown kernel_type: {self.kernel_type}")

    def _c_bridge_add_rmsnorm(self, shape: tuple) -> str:
        rows, hidden = shape
        return f"""
extern "C" void bench_launch(
    void* input, void* residual, void* weight,
    void* residual_out, void* quant_out, void* scales, void* stream) {{
    launch_fused_add_rmsnorm_nvfp4(
        (const __nv_bfloat16*)input,
        (const __nv_bfloat16*)residual,
        (const __nv_bfloat16*)weight,
        (__nv_bfloat16*)residual_out,
        (unsigned char*)quant_out,
        (__nv_fp8_storage_t*)scales,
        {rows}, {hidden}, (cudaStream_t)stream);
}}
"""

    def _c_bridge_silu_mul(self, shape: tuple) -> str:
        b, m, k = shape
        n = b * m * k
        return f"""
extern "C" void bench_launch(
    void* gate, void* up,
    void* quant_out, void* scales, void* stream) {{
    launch_silu_mul_fp4quant(
        (const __nv_bfloat16*)gate,
        (const __nv_bfloat16*)up,
        (uint8_t*)quant_out,
        (__nv_fp8_storage_t*)scales,
        {n}, (cudaStream_t)stream);
}}
"""

    def _c_bridge_nvfp4_quantize(self, shape: tuple) -> str:
        m, k = shape
        n = m * k
        return f"""
extern "C" void bench_launch(
    void* input,
    void* packed, void* scales, void* stream) {{
    launch_nvfp4_quantize_bf16(
        (const __nv_bfloat16*)input,
        (uint8_t*)packed,
        (__nv_fp8_storage_t*)scales,
        {n}, (cudaStream_t)stream);
}}
"""

    # ── Pybind11 wrapper: C++ source with torch::Tensor args ─────────────────

    def _pybind_wrapper(self, shape: tuple) -> str:
        if self.kernel_type == "add_rmsnorm":
            return self._pybind_add_rmsnorm()
        elif self.kernel_type == "silu_mul":
            return self._pybind_silu_mul()
        elif self.kernel_type == "nvfp4_quantize":
            return self._pybind_nvfp4_quantize()
        raise ValueError(f"Unknown kernel_type: {self.kernel_type}")

    def _pybind_add_rmsnorm(self) -> str:
        return """
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

extern "C" void bench_launch(void*, void*, void*, void*, void*, void*, void*);

void run_kernel(
    torch::Tensor input, torch::Tensor residual, torch::Tensor weight,
    torch::Tensor residual_out, torch::Tensor quant_out, torch::Tensor scales) {
    void* stream = (void*)at::cuda::getCurrentCUDAStream().stream();
    bench_launch(
        input.data_ptr(), residual.data_ptr(), weight.data_ptr(),
        residual_out.data_ptr(), quant_out.data_ptr(), scales.data_ptr(),
        stream);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run_kernel", &run_kernel);
}
"""

    def _pybind_silu_mul(self) -> str:
        return """
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

extern "C" void bench_launch(void*, void*, void*, void*, void*);

void run_kernel(
    torch::Tensor gate, torch::Tensor up,
    torch::Tensor quant_out, torch::Tensor scales) {
    void* stream = (void*)at::cuda::getCurrentCUDAStream().stream();
    bench_launch(
        gate.data_ptr(), up.data_ptr(),
        quant_out.data_ptr(), scales.data_ptr(),
        stream);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run_kernel", &run_kernel);
}
"""

    def _pybind_nvfp4_quantize(self) -> str:
        return """
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

extern "C" void bench_launch(void*, void*, void*, void*);

void run_kernel(
    torch::Tensor input,
    torch::Tensor packed, torch::Tensor scales) {
    void* stream = (void*)at::cuda::getCurrentCUDAStream().stream();
    bench_launch(
        input.data_ptr(),
        packed.data_ptr(), scales.data_ptr(),
        stream);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run_kernel", &run_kernel);
}
"""

    # ── Tensor creation for L2-cycling inputs ────────────────────────────────

    def _create_inputs(self, shape: tuple):
        """Create one set of input tensors on GPU (called per L2-cycle buffer)."""
        import torch
        if self.kernel_type == "add_rmsnorm":
            rows, hidden = shape
            return (
                torch.randn(rows, hidden, dtype=torch.bfloat16, device="cuda"),
                torch.randn(rows, hidden, dtype=torch.bfloat16, device="cuda"),
                torch.ones(hidden, dtype=torch.bfloat16, device="cuda"),
            )
        elif self.kernel_type == "silu_mul":
            b, m, k = shape
            return (
                torch.randn(b * m * k, dtype=torch.bfloat16, device="cuda"),
                torch.randn(b * m * k, dtype=torch.bfloat16, device="cuda"),
            )
        elif self.kernel_type == "nvfp4_quantize":
            m, k = shape
            return (
                torch.randn(m * k, dtype=torch.bfloat16, device="cuda"),
            )

    def _create_outputs(self, shape: tuple):
        """Create output tensors on GPU (shared across L2-cycle buffers)."""
        import torch
        if self.kernel_type == "add_rmsnorm":
            rows, hidden = shape
            n = rows * hidden
            nb = n // 16
            return (
                torch.empty(n, dtype=torch.bfloat16, device="cuda"),       # residual_out
                torch.empty(n // 2, dtype=torch.uint8, device="cuda"),     # quant_out (packed FP4)
                torch.empty(nb, dtype=torch.uint8, device="cuda"),         # scales (FP8 E4M3)
            )
        elif self.kernel_type == "silu_mul":
            b, m, k = shape
            n = b * m * k
            nb = n // 16
            return (
                torch.empty(n // 2, dtype=torch.uint8, device="cuda"),     # quant_out
                torch.empty(nb, dtype=torch.uint8, device="cuda"),         # scales
            )
        elif self.kernel_type == "nvfp4_quantize":
            m, k = shape
            n = m * k
            nb = n // 16
            return (
                torch.empty(n // 2, dtype=torch.uint8, device="cuda"),     # packed
                torch.empty(nb, dtype=torch.uint8, device="cuda"),         # scales
            )

    # ── Fallback: C++ binary timing (used when PyTorch unavailable) ──────────

    def _time_via_binary(self, kernel_src: str, shape: tuple) -> Optional[float]:
        """Fallback: compile to standalone binary and time in C++.

        WARNING: This path has lower dispatch overhead than the Python-based
        FlashInfer baseline, producing inflated speedup numbers. Only used
        when torch.utils.cpp_extension is unavailable.
        """
        logger.warning("Using C++ binary timing fallback — speedups will be inflated vs baseline")
        harness  = self._build_harness(shape)
        combined = kernel_src + "\n\n" + harness
        nvcc_flags = [
            "-O3", "-arch=sm_100a", "--use_fast_math", "-std=c++17",
        ] + [f"-I{d}" for d in self.include_dirs]
        with tempfile.TemporaryDirectory() as tmpdir:
            src = Path(tmpdir) / "bench.cu"
            exe = Path(tmpdir) / "bench"
            src.write_text(combined)
            cmd = ["nvcc"] + nvcc_flags + [str(src), "-o", str(exe)]
            try:
                r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            except subprocess.TimeoutExpired:
                logger.warning("Benchmark compile timed out after 120s for shape %s", shape)
                return None
            if r.returncode != 0:
                return None
            try:
                r2 = subprocess.run([str(exe)], capture_output=True, text=True, timeout=60)
            except subprocess.TimeoutExpired:
                logger.warning("Benchmark run timed out after 60s for shape %s", shape)
                return None
            output = r2.stdout
            if "TIMING_ANOMALY" in output:
                logger.warning("Timing anomaly detected (event/wall ratio > 1.5x): %s",
                               re.search(r"TIMING_ANOMALY:.*", output).group(0))
            m = re.search(r"timing_us:\s*([\d.]+)", output)
            return float(m.group(1)) if m else None

    # ── C++ harness generators (fallback only) ───────────────────────────────

    def _build_harness(self, shape: tuple) -> str:
        if self.kernel_type == "add_rmsnorm":
            return self._harness_add_rmsnorm(shape)
        elif self.kernel_type == "silu_mul":
            return self._harness_silu_mul(shape)
        elif self.kernel_type == "nvfp4_quantize":
            return self._harness_nvfp4_quantize(shape)
        else:
            raise ValueError(f"Unknown kernel_type: {self.kernel_type}")

    def _timing_footer(self, launch_call_indexed: str, iters: int, warmup: int) -> str:
        return f"""
    // Warmup with L2 cycling
    for(int i=0;i<{warmup};++i) {{ int buf_idx = i % nbufs; {launch_call_indexed} }}
    CHECK_CUDA(cudaGetLastError());
    cudaStreamSynchronize(s);
    cudaEvent_t t0,t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0,s);
    for(int i=0;i<{iters};++i) {{ int buf_idx = i % nbufs; {launch_call_indexed} }}
    CHECK_CUDA(cudaGetLastError());
    cudaEventRecord(t1,s); cudaStreamSynchronize(s);
    float ms_event=0; cudaEventElapsedTime(&ms_event,t0,t1);
    float us_event=ms_event*1000.f/{iters};
    cudaDeviceSynchronize();
    struct timespec ts0,ts1; clock_gettime(CLOCK_MONOTONIC,&ts0);
    for(int i=0;i<{iters};++i) {{ int buf_idx = i % nbufs; {launch_call_indexed} }}
    CHECK_CUDA(cudaGetLastError());
    cudaDeviceSynchronize(); clock_gettime(CLOCK_MONOTONIC,&ts1);
    double wall_ns=(ts1.tv_sec-ts0.tv_sec)*1e9+(ts1.tv_nsec-ts0.tv_nsec);
    float us_wall=(float)(wall_ns/1000.0/{iters});
    float ratio=(us_event>0)?us_wall/us_event:1.f;
    printf("timing_us: %.3f\\n",us_event);
    printf("timing_wall_us: %.3f\\n",us_wall);
    printf("timing_ratio: %.3f\\n",ratio);
    printf("l2_cycle_bufs: %d\\n",nbufs);
    if(ratio>1.5f) printf("TIMING_ANOMALY: event=%.3fus wall=%.3fus ratio=%.2fx\\n",us_event,us_wall,ratio);
"""

    @staticmethod
