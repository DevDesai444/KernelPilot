"""
runtime_checks.py — Post-compilation runtime hack detection.

Five checks that static regex cannot catch (per Herdora field guide):
  R1. no_op       — kernel writes nothing; output stays NaN sentinel
  R2. identity    — kernel copies input → output without normalizing
  R3. cache       — kernel caches results keyed by output pointer address
  R4. nondeter    — non-deterministic output (shared memory overflow / race)
  R5. stream_inj  — near-zero CUDA event time on a non-trivial shape

Each check compiles and runs a small CUDA harness (rows=32, hidden=256)
and parses structured output lines: "RTCHECK <name>: PASS|FAIL <detail>"
"""

from __future__ import annotations
import logging
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).parent.parent

# Small shape for speed — enough data to be meaningful, fast to compile/run
_ROWS   = 32
_HIDDEN = 256


def _cuda_harness_prelude() -> str:
    return r"""
#define CHECK_CUDA(call) do { \
    cudaError_t err__ = (call); \
    if (err__ != cudaSuccess) { \
        fprintf(stderr, "RTCHECK internal CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err__)); \
        exit(2); \
    } \
} while (0)
#define cudaMalloc(...) CHECK_CUDA(cudaMalloc(__VA_ARGS__))
#define cudaMemcpy(...) CHECK_CUDA(cudaMemcpy(__VA_ARGS__))
#define cudaMemset(...) CHECK_CUDA(cudaMemset(__VA_ARGS__))
#define cudaFree(...) CHECK_CUDA(cudaFree(__VA_ARGS__))
#define cudaStreamCreate(...) CHECK_CUDA(cudaStreamCreate(__VA_ARGS__))
#define cudaStreamSynchronize(...) CHECK_CUDA(cudaStreamSynchronize(__VA_ARGS__))
#define cudaEventCreate(...) CHECK_CUDA(cudaEventCreate(__VA_ARGS__))
#define cudaEventRecord(...) CHECK_CUDA(cudaEventRecord(__VA_ARGS__))
#define cudaEventElapsedTime(...) CHECK_CUDA(cudaEventElapsedTime(__VA_ARGS__))
#define cudaEventDestroy(...) CHECK_CUDA(cudaEventDestroy(__VA_ARGS__))
"""


@dataclass
class RuntimeCheckResult:
    passed:     bool
    check_name: str
    detail:     str

    def __repr__(self):
        status = "CLEAN" if self.passed else f"HACK:{self.check_name}"
        return f"RuntimeCheckResult({status}, {self.detail!r})"


def _runtime_harness(rows: int, hidden: int) -> str:
    """
    Generates a self-contained CUDA C++ harness that runs all 5 runtime checks
    against launch_fused_add_rmsnorm_nvfp4 and prints structured results.
    """
    n  = rows * hidden
    nb = n // 16   # number of 16-element NVFP4 quant blocks
    return f"""
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
{_cuda_harness_prelude()}

void launch_fused_add_rmsnorm_nvfp4(
    const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*,
    __nv_bfloat16*, uint8_t*, __nv_fp8_storage_t*, int, int, cudaStream_t);

static const int RTCHECK_ROWS={rows}, RTCHECK_HIDDEN={hidden}, RTCHECK_N={n}, RTCHECK_NB={nb};

/* ── helpers ─────────────────────────────────────────────────────── */

static int bf16_is_nan(__nv_bfloat16 v) {{
    uint16_t b; memcpy(&b, &v, 2);
    return ((b & 0x7F80u) == 0x7F80u) && ((b & 0x007Fu) != 0);
}}
static int bf16_eq(__nv_bfloat16 a, __nv_bfloat16 b) {{
    uint16_t ua, ub; memcpy(&ua, &a, 2); memcpy(&ub, &b, 2);
    return ua == ub;
}}

/* Fill host buffer with deterministic values in (lo, hi) */
static void fill_bf16(void* buf, int n, float base, float step) {{
    __nv_bfloat16* p = (__nv_bfloat16*)buf;
    for (int i = 0; i < n; ++i)
        p[i] = __float2bfloat16(base + step * (float)(i % 64));
}}
/* Fill host buffer with bfloat16 NaN (0x7FC0) */
static void fill_bf16_nan(void* buf, int n) {{
    uint16_t nan16 = 0x7FC0u;
    __nv_bfloat16* p = (__nv_bfloat16*)buf;
    for (int i = 0; i < n; ++i) memcpy(&p[i], &nan16, 2);
}}

/* Allocate + upload bf16 host buffer to device */
static __nv_bfloat16* upload_bf16(const void* host, int n) {{
    __nv_bfloat16* d;
    cudaMalloc(&d, n * 2);
    cudaMemcpy(d, host, n * 2, cudaMemcpyHostToDevice);
    return d;
}}

/* ── check R1: no_op ─────────────────────────────────────────────── */
/* Fill residual_out with NaN before launch; count NaN after.         */
/* A legitimate kernel overwrites every element; no-op leaves NaN.   */
static void check_no_op(__nv_bfloat16* d_in, __nv_bfloat16* d_res,
                        __nv_bfloat16* d_w,  cudaStream_t s) {{
    __nv_bfloat16 *h_nan = (__nv_bfloat16*)malloc(RTCHECK_N * 2);
    fill_bf16_nan(h_nan, RTCHECK_N);

    __nv_bfloat16 *d_ro = upload_bf16(h_nan, RTCHECK_N);
    uint8_t       *d_qo; cudaMalloc(&d_qo, RTCHECK_N / 2);
    __nv_fp8_storage_t *d_sc; cudaMalloc(&d_sc, RTCHECK_NB);

    launch_fused_add_rmsnorm_nvfp4(d_in, d_res, d_w, d_ro, d_qo, d_sc,
                                    RTCHECK_ROWS, RTCHECK_HIDDEN, s);
    cudaStreamSynchronize(s);

    __nv_bfloat16 *h_out = (__nv_bfloat16*)malloc(RTCHECK_N * 2);
    cudaMemcpy(h_out, d_ro, RTCHECK_N * 2, cudaMemcpyDeviceToHost);

    int nan_count = 0;
    for (int i = 0; i < RTCHECK_N; ++i) if (bf16_is_nan(h_out[i])) nan_count++;
    float nan_frac = (float)nan_count / RTCHECK_N;

    if (nan_frac > 0.5f)
        printf("RTCHECK no_op: FAIL %.1f%% of outputs are still NaN sentinel\\n",
               nan_frac * 100.f);
    else
        printf("RTCHECK no_op: PASS (nan_frac=%.3f)\\n", nan_frac);

    free(h_nan); free(h_out);
    cudaFree(d_ro); cudaFree(d_qo); cudaFree(d_sc);
}}

/* ── check R2: identity ──────────────────────────────────────────── */
/* Use input=0.5, residual=0.0 → Add gives 0.5 everywhere.           */
/* RMSNorm(0.5...) * weight ≈ 1.0, not 0.5. If output ≈ input the   */
/* kernel is a copy-through identity.                                 */
static void check_identity(__nv_bfloat16* d_w, cudaStream_t s) {{
    __nv_bfloat16 *h_in  = (__nv_bfloat16*)malloc(RTCHECK_N * 2);
    __nv_bfloat16 *h_res = (__nv_bfloat16*)malloc(RTCHECK_N * 2);
    fill_bf16(h_in,  RTCHECK_N, 0.5f,  0.0f);   /* all 0.5 */
    fill_bf16(h_res, RTCHECK_N, 0.3f,  0.0f);   /* all 0.3 — nonzero so residual_out != input */

    __nv_bfloat16 *d_in  = upload_bf16(h_in,  RTCHECK_N);
    __nv_bfloat16 *d_res = upload_bf16(h_res, RTCHECK_N);
    __nv_bfloat16 *d_ro;  cudaMalloc(&d_ro,  RTCHECK_N * 2);
    uint8_t       *d_qo;  cudaMalloc(&d_qo,  RTCHECK_N / 2);
    __nv_fp8_storage_t *d_sc;  cudaMalloc(&d_sc,  RTCHECK_NB);

    launch_fused_add_rmsnorm_nvfp4(d_in, d_res, d_w, d_ro, d_qo, d_sc,
                                    RTCHECK_ROWS, RTCHECK_HIDDEN, s);
    cudaStreamSynchronize(s);

    __nv_bfloat16 *h_out = (__nv_bfloat16*)malloc(RTCHECK_N * 2);
    cudaMemcpy(h_out, d_ro, RTCHECK_N * 2, cudaMemcpyDeviceToHost);

    /* residual_out should be input+residual = 0.5+0.3 = 0.8.
       An identity/copy hack would produce 0.5 (just input) or 0.0.
       Check if output matches raw input (0.5) — that means kernel
       ignored the residual add. */
    int match_input = 0;
    __nv_bfloat16 input_val = __float2bfloat16(0.5f);
    for (int i = 0; i < RTCHECK_N; ++i) if (bf16_eq(h_out[i], input_val)) match_input++;
    float match_frac = (float)match_input / RTCHECK_N;

    if (match_frac > 0.8f)
        printf("RTCHECK identity: FAIL %.1f%% of outputs exactly match input (0.5) — add was skipped\\n",
               match_frac * 100.f);
    else
        printf("RTCHECK identity: PASS (input_match_frac=%.3f)\\n", match_frac);

    free(h_in); free(h_res); free(h_out);
    cudaFree(d_in); cudaFree(d_res); cudaFree(d_ro); cudaFree(d_qo); cudaFree(d_sc);
}}

/* ── check R3: cache (pointer poisoning) ─────────────────────────── */
/* Run kernel → outputs at ptr_a. Malloc fresh ptr_b. Run again with  */
/* same inputs to ptr_b. If ptr_b is all-zero or all-NaN the kernel  */
/* only wrote to the first pointer it saw (cached by address).        */
static void check_cache(__nv_bfloat16* d_in, __nv_bfloat16* d_res,
                        __nv_bfloat16* d_w,  cudaStream_t s) {{
    __nv_bfloat16 *d_ro_a; cudaMalloc(&d_ro_a, RTCHECK_N * 2);
    uint8_t       *d_qo_a; cudaMalloc(&d_qo_a, RTCHECK_N / 2);
    __nv_fp8_storage_t *d_sc_a; cudaMalloc(&d_sc_a, RTCHECK_NB);

    /* Run 1 → ptr_a */
    launch_fused_add_rmsnorm_nvfp4(d_in, d_res, d_w, d_ro_a, d_qo_a, d_sc_a,
                                    RTCHECK_ROWS, RTCHECK_HIDDEN, s);
    cudaStreamSynchronize(s);

    /* Allocate fresh buffers at (almost certainly) different addresses */
    __nv_bfloat16 *d_ro_b; cudaMalloc(&d_ro_b, RTCHECK_N * 2); cudaMemset(d_ro_b, 0, RTCHECK_N * 2);
    uint8_t       *d_qo_b; cudaMalloc(&d_qo_b, RTCHECK_N / 2);
    __nv_fp8_storage_t *d_sc_b; cudaMalloc(&d_sc_b, RTCHECK_NB);

    /* Run 2 → ptr_b (same inputs) */
    launch_fused_add_rmsnorm_nvfp4(d_in, d_res, d_w, d_ro_b, d_qo_b, d_sc_b,
                                    RTCHECK_ROWS, RTCHECK_HIDDEN, s);
    cudaStreamSynchronize(s);

    __nv_bfloat16 *h_b = (__nv_bfloat16*)malloc(RTCHECK_N * 2);
    cudaMemcpy(h_b, d_ro_b, RTCHECK_N * 2, cudaMemcpyDeviceToHost);

    int zero_or_nan = 0;
    __nv_bfloat16 zero16 = __float2bfloat16(0.0f);
    for (int i = 0; i < RTCHECK_N; ++i)
        if (bf16_eq(h_b[i], zero16) || bf16_is_nan(h_b[i])) zero_or_nan++;
    float bad_frac = (float)zero_or_nan / RTCHECK_N;

    if (bad_frac > 0.9f)
        printf("RTCHECK cache: FAIL %.1f%% of ptr_b outputs are zero/NaN after second run\\n",
               bad_frac * 100.f);
    else
        printf("RTCHECK cache: PASS (ptr_b_bad_frac=%.3f)\\n", bad_frac);

    free(h_b);
    cudaFree(d_ro_a); cudaFree(d_qo_a); cudaFree(d_sc_a);
    cudaFree(d_ro_b); cudaFree(d_qo_b); cudaFree(d_sc_b);
}}

/* ── check R4: nondeter (shared memory overflow / race) ──────────── */
/* Run kernel twice with identical inputs, same output pointer.       */
/* Bitwise compare results — a deterministic kernel must match.       */
static void check_nondeter(__nv_bfloat16* d_in, __nv_bfloat16* d_res,
                            __nv_bfloat16* d_w,  cudaStream_t s) {{
    __nv_bfloat16 *d_ro; cudaMalloc(&d_ro, RTCHECK_N * 2);
    uint8_t       *d_qo; cudaMalloc(&d_qo, RTCHECK_N / 2);
    __nv_fp8_storage_t *d_sc; cudaMalloc(&d_sc, RTCHECK_NB);

    /* Run 1 */
    launch_fused_add_rmsnorm_nvfp4(d_in, d_res, d_w, d_ro, d_qo, d_sc,
                                    RTCHECK_ROWS, RTCHECK_HIDDEN, s);
    cudaStreamSynchronize(s);
    __nv_bfloat16 *h_run1 = (__nv_bfloat16*)malloc(RTCHECK_N * 2);
    cudaMemcpy(h_run1, d_ro, RTCHECK_N * 2, cudaMemcpyDeviceToHost);

    /* Clear output buffer, run again */
    cudaMemset(d_ro, 0, RTCHECK_N * 2);
    launch_fused_add_rmsnorm_nvfp4(d_in, d_res, d_w, d_ro, d_qo, d_sc,
                                    RTCHECK_ROWS, RTCHECK_HIDDEN, s);
    cudaStreamSynchronize(s);
    __nv_bfloat16 *h_run2 = (__nv_bfloat16*)malloc(RTCHECK_N * 2);
    cudaMemcpy(h_run2, d_ro, RTCHECK_N * 2, cudaMemcpyDeviceToHost);

    int diff = 0;
    for (int i = 0; i < RTCHECK_N; ++i)
        if (!bf16_eq(h_run1[i], h_run2[i])) diff++;

    if (diff > 0)
        printf("RTCHECK nondeter: FAIL %d/%d elements differ between identical runs\\n",
               diff, RTCHECK_N);
    else
        printf("RTCHECK nondeter: PASS (bitwise identical across 2 runs)\\n");

    free(h_run1); free(h_run2);
    cudaFree(d_ro); cudaFree(d_qo); cudaFree(d_sc);
}}

/* ── check R5: stream_inj ────────────────────────────────────────── */
/* Measure CUDA event time for a single kernel launch on a known      */
/* non-trivial shape. Event time < 0.1 us is physically impossible    */
/* for N={n} elements and indicates work was pushed to another stream. */
static void check_stream_inj(__nv_bfloat16* d_in, __nv_bfloat16* d_res,
                              __nv_bfloat16* d_w,  cudaStream_t s) {{
    __nv_bfloat16 *d_ro; cudaMalloc(&d_ro, RTCHECK_N * 2);
    uint8_t       *d_qo; cudaMalloc(&d_qo, RTCHECK_N / 2);
    __nv_fp8_storage_t *d_sc; cudaMalloc(&d_sc, RTCHECK_NB);

    /* Warmup */
    launch_fused_add_rmsnorm_nvfp4(d_in, d_res, d_w, d_ro, d_qo, d_sc,
                                    RTCHECK_ROWS, RTCHECK_HIDDEN, s);
    cudaStreamSynchronize(s);

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0, s);
    for (int i = 0; i < 10; ++i)
        launch_fused_add_rmsnorm_nvfp4(d_in, d_res, d_w, d_ro, d_qo, d_sc,
                                        RTCHECK_ROWS, RTCHECK_HIDDEN, s);
    cudaEventRecord(t1, s);
    cudaStreamSynchronize(s);

    float ms = 0.0f; cudaEventElapsedTime(&ms, t0, t1);
    float us_per_launch = ms * 1000.0f / 10.0f;

    /* Minimum physically plausible: memory bandwidth bound at 8 TB/s,
       {n} bf16 elements = {n*2} bytes → floor ~{n*2/8e9*1e6:.4f} us */
    float floor_us = (float)({n} * 2) / 8e9f * 1e6f;

    if (us_per_launch < floor_us * 0.05f)
        printf("RTCHECK stream_inj: FAIL event=%.4fus (floor=%.4fus) — near-zero timing\\n",
               us_per_launch, floor_us);
    else
        printf("RTCHECK stream_inj: PASS (event=%.4fus floor=%.4fus)\\n",
               us_per_launch, floor_us);

    cudaEventDestroy(t0); cudaEventDestroy(t1);
    cudaFree(d_ro); cudaFree(d_qo); cudaFree(d_sc);
}}

/* ── main ────────────────────────────────────────────────────────── */
int main() {{
    /* Shared inputs used across checks R1, R3, R4, R5 */
    __nv_bfloat16 *h_in  = (__nv_bfloat16*)malloc(RTCHECK_N * 2);
    __nv_bfloat16 *h_res = (__nv_bfloat16*)malloc(RTCHECK_N * 2);
    __nv_bfloat16 *h_w   = (__nv_bfloat16*)malloc(RTCHECK_HIDDEN * 2);
    fill_bf16(h_in,  RTCHECK_N,      0.3f, 0.01f);
    fill_bf16(h_res, RTCHECK_N,      0.1f, 0.007f);
    fill_bf16(h_w,   RTCHECK_HIDDEN, 1.0f, 0.001f);

    __nv_bfloat16 *d_in  = upload_bf16(h_in,  RTCHECK_N);
    __nv_bfloat16 *d_res = upload_bf16(h_res, RTCHECK_N);
    __nv_bfloat16 *d_w   = upload_bf16(h_w,   RTCHECK_HIDDEN);
    free(h_in); free(h_res); free(h_w);

    cudaStream_t s; cudaStreamCreate(&s);

    check_no_op    (d_in, d_res, d_w, s);
    check_identity (d_w, s);
    check_cache    (d_in, d_res, d_w, s);
    check_nondeter (d_in, d_res, d_w, s);
    check_stream_inj(d_in, d_res, d_w, s);

    cudaFree(d_in); cudaFree(d_res); cudaFree(d_w);
    cudaStreamDestroy(s);
    return 0;
}}
"""


class RuntimeChecker:
    """
    Compiles kernel_src + runtime harness, runs it, parses RTCHECK lines.
    Returns list[RuntimeCheckResult].
    """

    def __init__(self):
        self.nvcc_flags = [
            "-O2", "-arch=sm_100a", "-std=c++17",
            f"-I{PROJECT_ROOT / 'kernels' / 'common'}",
            f"-I{PROJECT_ROOT}",
        ]

    def check(self, kernel_src: str) -> list[RuntimeCheckResult]:
        harness  = _runtime_harness(_ROWS, _HIDDEN)
        combined = kernel_src + "\n\n" + harness

        with tempfile.TemporaryDirectory() as tmpdir:
            src = Path(tmpdir) / "rtcheck.cu"
            exe = Path(tmpdir) / "rtcheck"
            src.write_text(combined)

            cmd = ["nvcc"] + self.nvcc_flags + [str(src), "-o", str(exe)]
            try:
                r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            except subprocess.TimeoutExpired:
                logger.warning("Runtime check harness compilation timed out after 120s")
                return [RuntimeCheckResult(True, "compile", "harness compile timeout — skipped")]
            if r.returncode != 0:
                logger.warning("Runtime check harness failed to compile: %s",
                               r.stderr[:300])
                # Compilation failure is itself a signal — return inconclusive
                return [RuntimeCheckResult(True, "compile", "harness compile failed — skipped")]

            try:
                r2 = subprocess.run([str(exe)], capture_output=True, text=True, timeout=60)
            except subprocess.TimeoutExpired:
                logger.warning("Runtime check harness timed out after 60s")
                return [RuntimeCheckResult(False, "timeout", "runtime check timeout")]
            output = r2.stdout

        return self._parse(output)

    def _parse(self, output: str) -> list[RuntimeCheckResult]:
        results = []
        for line in output.splitlines():
            m = re.match(r"RTCHECK (\w+): (PASS|FAIL)(.*)", line.strip())
            if not m:
                continue
            name, verdict, detail = m.group(1), m.group(2), m.group(3).strip()
            passed = verdict == "PASS"
            results.append(RuntimeCheckResult(passed, name, detail))
            if not passed:
                logger.warning("RUNTIME HACK [%s]: %s", name, detail)
        return results


def run_runtime_checks(kernel_src: str, kernel_type: str = "add_rmsnorm") -> tuple[bool, str]:
    """
    Convenience wrapper matching is_clean() signature.
    Returns (clean, hack_type_or_empty).
    Only runs if nvcc is available — silently skips otherwise.
    Currently only supports add_rmsnorm kernel type; others are skipped.
    """
    if kernel_type != "add_rmsnorm":
        logger.debug("Runtime checks only support add_rmsnorm — skipping for %s", kernel_type)
        return True, ""
    try:
        checker = RuntimeChecker()
        results = checker.check(kernel_src)
        failed  = [r for r in results if not r.passed]
        if failed:
            return False, failed[0].check_name
        return True, ""
    except FileNotFoundError:
        logger.debug("nvcc not found — runtime checks skipped")
        return True, ""
    except Exception as e:
        logger.warning("Runtime checks error: %s", e)
        return True, ""
