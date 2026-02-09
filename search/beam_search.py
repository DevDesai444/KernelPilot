"""
beam_search.py — Main beam search orchestrator.
Ties together: RLM engine, profiler, diversity selection, and combination.
"""

from __future__ import annotations
import logging
import re
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

from rlm.engine import RLMEngine
from rlm.environment import RLMEnvironment, KernelCandidate
from profiler.kernel_profiler import KernelProfiler
from profiler.metrics import KernelMetrics, metrics_from_dict
from search.diversity_selector import DiversitySelector
from eval.benchmark import Benchmarker
from eval.hack_detector import is_clean
from eval.runtime_checks import run_runtime_checks
from eval.correctness import CorrectnessChecker

logger = logging.getLogger(__name__)


class BeamSearch:
    """
    Profiler-guided beam search for CUDA kernel optimization.

    Algorithm:
      0. Root LLM decomposes → selects strategies
      1. Generate N beams in parallel (sub-LLMs)
      2. Compile + profile each beam
      3. Label branch family per beam
      4. Select diverse survivors (1 per strategy family)
      5. Refine each survivor with targeted sub-LLM
      6. Repeat rounds 2-5 until budget or round limit
      7. Combine top-2 survivors → final kernel
    """

    def __init__(self, env: RLMEnvironment):
        self.env      = env
        self.engine   = RLMEngine(env)
        self.profiler = KernelProfiler(env.search_config, hw_spec=env.hw_spec)
        self.benchmarker = Benchmarker(env.search_config, kernel_type=env.kernel_type)
        self.selector = DiversitySelector(env.search_config)
        self.checker  = CorrectnessChecker(env.search_config)
        self.beam_w   = env.search_config["beam"]["width"]
        self.rounds   = env.search_config["beam"]["refine_rounds"]
        beam_cfg = env.search_config.get("beam", {})
        profiler_cfg = env.search_config.get("profiler", {})
        self.family_retire_gap = float(beam_cfg.get("family_retire_gap", 0.25))
        self.family_retire_ratio = float(beam_cfg.get("family_retire_ratio", 0.8))
        self.plateau_refine_attempts = int(beam_cfg.get("plateau_refine_attempts", 2))
        self.plateau_bad_rounds = int(beam_cfg.get("plateau_bad_rounds", 2))
        self.population_crossover = bool(beam_cfg.get("population_crossover", True))
        self.early_stop_min_improvement = float(beam_cfg.get("early_stop_min_improvement", 0.03))
        self.max_profile_workers = int(profiler_cfg.get("max_profile_workers", 2))
        self._env_lock = threading.Lock()  # guards shared env counters

    @staticmethod
    def _branch_family(candidate: KernelCandidate) -> str:
        return (
            candidate.branch_family
            or candidate.parent_strategy
            or candidate.strategy.split("__", 1)[0]
        )

    @staticmethod
    def _recent_bad_outcomes(candidate: KernelCandidate, limit: int) -> int:
        recent = list(candidate.refinement_history[-limit:])
        return sum(entry.get("outcome") in {"stagnant", "regression"} for entry in recent)

    def _is_plateaued(self, candidate: KernelCandidate) -> bool:
        if candidate.refine_attempts >= self.plateau_refine_attempts:
            return True
        return self._recent_bad_outcomes(candidate, self.plateau_bad_rounds) >= self.plateau_bad_rounds

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
#define cudaEventCreate(...) CHECK_CUDA(cudaEventCreate(__VA_ARGS__))
#define cudaEventRecord(...) CHECK_CUDA(cudaEventRecord(__VA_ARGS__))
#define cudaEventElapsedTime(...) CHECK_CUDA(cudaEventElapsedTime(__VA_ARGS__))
#define cudaEventDestroy(...) CHECK_CUDA(cudaEventDestroy(__VA_ARGS__))
#define cudaDeviceGetAttribute(...) CHECK_CUDA(cudaDeviceGetAttribute(__VA_ARGS__))
"""

    def _prune_stale_families(self, survivors: list[KernelCandidate]) -> list[KernelCandidate]:
        if len(survivors) <= 1:
            return survivors

        best = max(survivors, key=lambda c: c.speedup)
        best_family = self._branch_family(best)
        pruned = []

        for candidate in survivors:
            if candidate is best:
                pruned.append(candidate)
                continue

            family = self._branch_family(candidate)
            materially_behind = (
                candidate.speedup < best.speedup * self.family_retire_ratio
                or (best.speedup - candidate.speedup) > self.family_retire_gap
            )
            if (
                family != best_family
                and materially_behind
                and self._is_plateaued(candidate)
            ):
                logger.info(
                    "Pruning stale family [%s]: best=%s %.3fx candidate=%.3fx attempts=%d",
                    family,
                    best_family,
                    best.speedup,
                    candidate.speedup,
                    candidate.refine_attempts,
                )
                continue

            pruned.append(candidate)

        pruned.sort(key=lambda c: -c.speedup)
        return pruned[:self.beam_w]

    def _family_distinct_top_pair(self, candidates: list[KernelCandidate]) -> tuple[KernelCandidate, KernelCandidate] | None:
        ranked = [candidate for candidate in sorted(candidates, key=lambda c: -c.speedup) if candidate.is_viable()]
        if len(ranked) < 2:
            return None
        first = ranked[0]
        first_family = self._branch_family(first)
        for other in ranked[1:]:
            if self._branch_family(other) != first_family:
                return first, other
        return None

    def _spawn_crossover_candidate(
        self,
        survivors: list[KernelCandidate],
        round_num: int,
        problem_shape: tuple,
        baseline_us: float,
    ) -> tuple[KernelCandidate, KernelMetrics] | None:
        if not self.population_crossover or len(survivors) < 2 or self.env.over_budget():
            return None
        pair = self._family_distinct_top_pair(survivors)
        if pair is None:
            return None
        logger.info(
            "Injecting crossover candidate from families %s + %s",
            self._branch_family(pair[0]),
            self._branch_family(pair[1]),
        )
        try:
            merged = self.engine.combine(list(pair))
        except Exception as exc:
            logger.warning("Crossover combine failed: %s", exc)
            return None
        merged.round_num = round_num
        metrics = self._profile_candidate(merged, problem_shape, baseline_us)
        return merged, metrics or KernelMetrics()

    def _build_harness(self, problem_shape: tuple) -> str:
        kt = self.env.kernel_type
        if kt == "add_rmsnorm":
            return self._harness_add_rmsnorm(problem_shape)
        elif kt == "silu_mul":
            return self._harness_silu_mul(problem_shape)
        elif kt == "nvfp4_quantize":
            return self._harness_nvfp4_quantize(problem_shape)
        else:
            raise ValueError(f"Unknown kernel_type: {kt}")

    def _harness_add_rmsnorm(self, shape: tuple) -> str:
        rows, hidden = shape
        n, nb = rows * hidden, rows * hidden // 16
        input_bytes = n * 2 * 2 + hidden * 2  # (input+residual)*bf16 + weight
        return f"""
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <stdio.h>
#include <stdlib.h>
{self._cuda_harness_prelude()}
void launch_fused_add_rmsnorm_nvfp4(
    const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*,
    __nv_bfloat16*, unsigned char*, __nv_fp8_storage_t*, int, int, cudaStream_t);
int main(int argc, char** argv) {{
    int warmup=500, iters=100;
    for(int i=1;i<argc;++i){{ sscanf(argv[i],"--warmup=%d",&warmup); sscanf(argv[i],"--iters=%d",&iters); }}
    const int rows={rows}, hidden={hidden}, N={n}, nb={nb};
    // Dynamic L2 cache cycling (KernelArena/ThunderKittens 2.0 methodology)
    int l2_bytes=0;
    cudaDeviceGetAttribute(&l2_bytes, cudaDevAttrL2CacheSize, 0);
    int nbufs = 1;
    if ({input_bytes} > 0 && l2_bytes > 0 && {input_bytes} < l2_bytes * 3)
        nbufs = l2_bytes * 3 / {input_bytes} + 1;
    if (nbufs > 256) nbufs = 256;
    if (nbufs < 1) nbufs = 1;
    __nv_bfloat16 **di = (__nv_bfloat16**)malloc(nbufs*sizeof(void*));
    __nv_bfloat16 **dr = (__nv_bfloat16**)malloc(nbufs*sizeof(void*));
    __nv_bfloat16 *dw, *dro; unsigned char *dq; __nv_fp8_storage_t *ds;
    for(int b=0;b<nbufs;++b) {{ cudaMalloc(&di[b],N*2); cudaMalloc(&dr[b],N*2); }}
    cudaMalloc(&dw,hidden*2);
    cudaMalloc(&dro,N*2); cudaMalloc(&dq,N/2); cudaMalloc(&ds,nb);
    cudaStream_t s; cudaStreamCreate(&s);
    // Warmup with L2 cycling
    for(int i=0;i<warmup;++i) launch_fused_add_rmsnorm_nvfp4(di[i%nbufs],dr[i%nbufs],dw,dro,dq,ds,rows,hidden,s);
    cudaStreamSynchronize(s);
    // Timed reps — 2 CUDA events wrapping all reps (ThunderKittens convention)
    cudaEvent_t t0,t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0,s);
    for(int i=0;i<iters;++i) launch_fused_add_rmsnorm_nvfp4(di[i%nbufs],dr[i%nbufs],dw,dro,dq,ds,rows,hidden,s);
    cudaEventRecord(t1,s); cudaStreamSynchronize(s);
    float ms=0; cudaEventElapsedTime(&ms,t0,t1);
    printf("timing_us: %.3f\\n", ms*1000.f/iters);
    printf("l2_cycle_bufs: %d\\n", nbufs);
    for(int b=0;b<nbufs;++b) {{ cudaFree(di[b]); cudaFree(dr[b]); }}
    free(di); free(dr);
    cudaFree(dw); cudaFree(dro); cudaFree(dq); cudaFree(ds);
    return 0;
}}
"""

    def _harness_silu_mul(self, shape: tuple) -> str:
        b, m, k = shape
        n = b * m * k
        nb = n // 16
        input_bytes = n * 2 * 2  # (gate + up) * bf16
        return f"""
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
{self._cuda_harness_prelude()}
void launch_silu_mul_fp4quant(
    const __nv_bfloat16*, const __nv_bfloat16*,
    uint8_t*, __nv_fp8_storage_t*, int, cudaStream_t);
int main(int argc, char** argv) {{
    int warmup=500, iters=100;
    for(int i=1;i<argc;++i){{ sscanf(argv[i],"--warmup=%d",&warmup); sscanf(argv[i],"--iters=%d",&iters); }}
    const int N={n}, nb={nb};
    // Dynamic L2 cache cycling
    int l2_bytes=0;
    cudaDeviceGetAttribute(&l2_bytes, cudaDevAttrL2CacheSize, 0);
    int nbufs = 1;
    if ({input_bytes} > 0 && l2_bytes > 0 && {input_bytes} < l2_bytes * 3)
        nbufs = l2_bytes * 3 / {input_bytes} + 1;
    if (nbufs > 256) nbufs = 256;
    if (nbufs < 1) nbufs = 1;
    __nv_bfloat16 **dg = (__nv_bfloat16**)malloc(nbufs*sizeof(void*));
    __nv_bfloat16 **du = (__nv_bfloat16**)malloc(nbufs*sizeof(void*));
    uint8_t *dq; __nv_fp8_storage_t *ds;
    for(int b=0;b<nbufs;++b) {{ cudaMalloc(&dg[b],N*2); cudaMalloc(&du[b],N*2); }}
    cudaMalloc(&dq,N/2); cudaMalloc(&ds,nb);
    cudaStream_t s; cudaStreamCreate(&s);
    for(int i=0;i<warmup;++i) launch_silu_mul_fp4quant(dg[i%nbufs],du[i%nbufs],dq,ds,N,s);
    cudaStreamSynchronize(s);
    cudaEvent_t t0,t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0,s);
    for(int i=0;i<iters;++i) launch_silu_mul_fp4quant(dg[i%nbufs],du[i%nbufs],dq,ds,N,s);
    cudaEventRecord(t1,s); cudaStreamSynchronize(s);
    float ms=0; cudaEventElapsedTime(&ms,t0,t1);
    printf("timing_us: %.3f\\n", ms*1000.f/iters);
    printf("l2_cycle_bufs: %d\\n", nbufs);
    for(int b=0;b<nbufs;++b) {{ cudaFree(dg[b]); cudaFree(du[b]); }}
    free(dg); free(du);
    cudaFree(dq); cudaFree(ds);
    return 0;
}}
"""

    def _harness_nvfp4_quantize(self, shape: tuple) -> str:
        m, k = shape
        n = m * k
        nb = n // 16
        input_bytes = n * 2  # input * bf16
        return f"""
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
{self._cuda_harness_prelude()}
void launch_nvfp4_quantize_bf16(
    const __nv_bfloat16*, uint8_t*, __nv_fp8_storage_t*, int, cudaStream_t);
int main(int argc, char** argv) {{
    int warmup=500, iters=100;
    for(int i=1;i<argc;++i){{ sscanf(argv[i],"--warmup=%d",&warmup); sscanf(argv[i],"--iters=%d",&iters); }}
    const int N={n}, nb={nb};
    // Dynamic L2 cache cycling
    int l2_bytes=0;
    cudaDeviceGetAttribute(&l2_bytes, cudaDevAttrL2CacheSize, 0);
    int nbufs = 1;
    if ({input_bytes} > 0 && l2_bytes > 0 && {input_bytes} < l2_bytes * 3)
        nbufs = l2_bytes * 3 / {input_bytes} + 1;
    if (nbufs > 256) nbufs = 256;
    if (nbufs < 1) nbufs = 1;
    __nv_bfloat16 **din = (__nv_bfloat16**)malloc(nbufs*sizeof(void*));
    uint8_t *dpk; __nv_fp8_storage_t *dsc;
    for(int b=0;b<nbufs;++b) {{ cudaMalloc(&din[b],N*2); }}
    cudaMalloc(&dpk,N/2); cudaMalloc(&dsc,nb);
    cudaStream_t s; cudaStreamCreate(&s);
    for(int i=0;i<warmup;++i) launch_nvfp4_quantize_bf16(din[i%nbufs],dpk,dsc,N,s);
    cudaStreamSynchronize(s);
    cudaEvent_t t0,t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0,s);
    for(int i=0;i<iters;++i) launch_nvfp4_quantize_bf16(din[i%nbufs],dpk,dsc,N,s);
    cudaEventRecord(t1,s); cudaStreamSynchronize(s);
    float ms=0; cudaEventElapsedTime(&ms,t0,t1);
    printf("timing_us: %.3f\\n", ms*1000.f/iters);
    printf("l2_cycle_bufs: %d\\n", nbufs);
    for(int b=0;b<nbufs;++b) {{ cudaFree(din[b]); }}
    free(din);
    cudaFree(dpk); cudaFree(dsc);
    return 0;
}}
"""

    def measure_search_baseline(self, problem_shape: tuple) -> tuple[Optional[float], Optional[CompilerMetrics]]:
        """Measure reference kernel timing with the SAME harness used for candidates.

        Ensures symmetric measurement: both baseline and candidate see the same
        L2 cache cycling, same CUDA event timing, same C++ dispatch overhead.
        """
        harness = self._build_harness(problem_shape)
        safe_name = f"baseline_{self.env.kernel_type}_{int(time.time())}"
        ok, err, binary, baseline_cm = self.profiler.compile_kernel(
            kernel_src=self.env.kernel_src_raw, harness_src=harness,
            output_name=safe_name,
        )
        if not ok:
            logger.warning("Baseline compilation failed: %s", err[:200])
            return None, None
        timing = self._benchmark_with_graphs(self.env.kernel_src_raw, problem_shape)
        if timing is None:
            logger.warning("Graph benchmark baseline failed; falling back to binary event timing")
            timing = self.profiler.benchmark_timing(binary)
        if timing:
            logger.info("Search baseline (with L2 cycling): %.3f us", timing)
        self.env.baseline_naive_us = timing
        self.env.baseline_compiler_metrics = baseline_cm
        return timing, baseline_cm

    def _benchmark_with_graphs(self, kernel_src: str, problem_shape: tuple) -> Optional[float]:
        try:
            return self.benchmarker._compile_and_time(kernel_src, problem_shape)
        except Exception as exc:
            logger.warning("Graph benchmark failed for %s: %s", self.env.kernel_type, exc)
            return None

    @staticmethod
    def _timing_delta_pct(graph_timing_us: Optional[float], binary_timing_us: Optional[float]) -> Optional[float]:
        if graph_timing_us is None or binary_timing_us is None or binary_timing_us <= 0:
            return None
        return ((graph_timing_us - binary_timing_us) / binary_timing_us) * 100.0

    @staticmethod
    def _speedup_from_timing(baseline_us: float, timing_us: Optional[float]) -> float:
        if timing_us is None or timing_us <= 0 or baseline_us <= 0:
            return 0.0
        return baseline_us / timing_us

    def _profile_candidate(
        self,
        candidate: KernelCandidate,
        problem_shape: tuple,
        baseline_us: float,
    ) -> Optional[KernelMetrics]:
        clean, hack_type = is_clean(candidate.code)
        if not clean:
            logger.warning("Hack detected in candidate [%s]: %s — rejecting",
                           candidate.strategy, hack_type)
            candidate.compile_ok = False
            candidate.speedup    = 0.0
            candidate.bottleneck = "unknown"
            with self._env_lock:
                self.env.hack_rejections.append(
                    {"strategy": candidate.strategy, "hack_type": hack_type,
                     "round": candidate.round_num}
                )
            return None

        harness = self._build_harness(problem_shape)
        # Sanitize strategy name for filesystem (freeform names may have spaces/slashes)
        safe_strat = re.sub(r'[^a-zA-Z0-9_-]', '_', candidate.strategy)
        name    = f"{safe_strat}_r{candidate.round_num}_{int(time.time())}_{id(candidate)}"

        # Compile, check correctness, profile.
        # Note: counters reflect ALL attempts including inner refinement retries.
        ok = False
        metrics = None
        speedup = 0.0
        binary_speedup = None  # set inside timing block; used for routing at end
        with self._env_lock:
            self.env.total_attempts += 1

        compile_ok, err_msg, binary, compiler_metrics = self.profiler.compile_kernel(
            kernel_src=candidate.code, harness_src=harness, output_name=name,
        )
        if not compile_ok:
            candidate.compile_ok = False
            candidate.compile_error = err_msg[:800]
            logger.error("  Compile FAIL [%s]: %s", candidate.strategy, err_msg[:400])
        if compile_ok:
            candidate.compile_ok = True  # nvcc succeeded
            with self._env_lock:
                self.env.compile_passes += 1
            passed, max_err, msg = self.checker.check(candidate.code, problem_shape,
                                                          kernel_type=self.env.kernel_type)
            if not passed:
                candidate.compile_error = f"Correctness: {msg[:600]}"
                logger.warning("  Correctness FAIL [%s] (err=%.4f): %s",
                               candidate.strategy, max_err, msg[:200])
            else:
                with self._env_lock:
                    self.env.correctness_passes += 1
                candidate.correct = True
                graph_timing_us = self._benchmark_with_graphs(candidate.code, problem_shape)
                binary_timing_us = self.profiler.benchmark_timing(binary)
                timing_delta_pct = self._timing_delta_pct(graph_timing_us, binary_timing_us)
                graph_speedup = self._speedup_from_timing(baseline_us, graph_timing_us)
                binary_speedup = self._speedup_from_timing(baseline_us, binary_timing_us)

                if graph_timing_us is not None or binary_timing_us is not None:
                    graph_str = f"{graph_timing_us:.3f}" if graph_timing_us is not None else "n/a"
                    binary_str = f"{binary_timing_us:.3f}" if binary_timing_us is not None else "n/a"
                    delta_str = f"{timing_delta_pct:+.1f}%" if timing_delta_pct is not None else "n/a"
                    logger.info(
                        "  Timing AB [%s]: graph=%sus binary=%sus delta=%s",
                        candidate.strategy,
                        graph_str,
                        binary_str,
                        delta_str,
                    )
                    logger.info(
                        "  Speedup check [%s]: baseline=%.3fus source=%s graph=%.3fx binary=%.3fx",
                        candidate.strategy,
                        baseline_us,
                        self.env.baseline_source,
                        graph_speedup,
                        binary_speedup,
                    )
                    if timing_delta_pct is not None and abs(timing_delta_pct) >= 10.0:
                        logger.warning(
                            "  Timing-path mismatch [%s]: graph and binary differ by %.1f%%",
                            candidate.strategy,
                            timing_delta_pct,
                        )

                timing_us = graph_timing_us
                timing_path = "graph"
                # Fall back to binary if graph is None OR if mismatch is extreme
                # (>100% delta indicates broken graph capture, e.g. L2 cycling issue on large shapes)
                if timing_us is None:
                    logger.warning(
                        "Graph benchmark failed for [%s]; falling back to binary event timing",
                        candidate.strategy,
                    )
                    timing_us = binary_timing_us
                    timing_path = "binary_fallback"
                elif timing_delta_pct is not None and abs(timing_delta_pct) > 100.0 and binary_timing_us is not None:
                    logger.warning(
                        "  Graph timing suspect for [%s] (delta=%.1f%%); using binary timing %.3fus instead",
                        candidate.strategy, timing_delta_pct, binary_timing_us,
                    )
                    timing_us = binary_timing_us
                    timing_path = "binary_fallback"
                if timing_us is not None:
                    speedup = self._speedup_from_timing(baseline_us, timing_us)
                    metrics = self.profiler.profile(
                        binary, report_name=name,
                        kernel_src=candidate.code,
                        kernel_type=self.env.kernel_type,
                        problem_shape=problem_shape,
                        baseline_us=baseline_us,
                        timing_us=timing_us,
                        compiler_metrics=compiler_metrics,
                    )
                    if metrics:
                        metrics.duration_us = timing_us
                        metrics.speedup = speedup
                        metrics.graph_timing_us = graph_timing_us or 0.0
                        metrics.binary_timing_us = binary_timing_us or 0.0
                        metrics.timing_delta_pct = timing_delta_pct or 0.0
                        metrics.graph_speedup = graph_speedup
                        metrics.binary_speedup = binary_speedup
                        logger.info("  Profiler: occ=%.1f%% timing=%.1fus speedup=%.3fx",
                                    metrics.sm_occupancy, metrics.duration_us, metrics.speedup)
                    else:
                        logger.warning("  Profiler returned no metrics for [%s]", candidate.strategy)
                    candidate.metrics = {
                        "baseline_us": baseline_us,
                        "baseline_source": self.env.baseline_source,
                        "selected_timing_path": timing_path,
                        "selected_timing_us": timing_us,
                        "graph_timing_us": graph_timing_us,
                        "binary_timing_us": binary_timing_us,
                        "timing_delta_pct": timing_delta_pct,
                        "graph_speedup": graph_speedup,
                        "binary_speedup": binary_speedup,
                    }
                ok = True

        # Runtime hack checks — run after compile confirms the kernel is valid CUDA
        if ok:
            rt_clean, rt_hack = run_runtime_checks(candidate.code, kernel_type=self.env.kernel_type)
            if not rt_clean:
                logger.warning("Runtime hack detected in candidate [%s]: %s — rejecting",
                               candidate.strategy, rt_hack)
                candidate.compile_ok = False
                candidate.speedup    = 0.0
                candidate.bottleneck = "rejected"
                with self._env_lock:
                    self.env.hack_rejections.append(
                        {"strategy": candidate.strategy, "hack_type": f"runtime:{rt_hack}",
                         "round": candidate.round_num}
                    )
                return None

        # Always set speedup from timing, even if profiling failed
        candidate.speedup = speedup
        if metrics:
            metrics_dict = metrics.to_dict()
            metrics_dict.update(candidate.metrics)
            candidate.metrics = metrics_dict
            candidate.bottleneck = self._branch_family(candidate) or "unlabeled"
        if candidate.compile_ok and candidate.correct:
            # Use binary speedup for routing when available: graph timing is symmetric
            # with the FlashInfer baseline but both are inflated ~3-8% by the stream-s
            # pre-capture step, compressing ratios toward 1.0x. Binary timing is ground
            # truth. A kernel is "above baseline" if either measure says so.
            routing_speedup = max(speedup, binary_speedup if binary_speedup is not None else 0.0)
            route = "planner_tree" if routing_speedup >= 1.0 else "fixer_with_rag"
            candidate.feedback_route = route
            logger.info(
                "  route=%s graph=%.3fx binary=%s",
                route, speedup,
                f"{binary_speedup:.3f}x" if binary_speedup is not None else "n/a",
            )
        return metrics

    def _make_inner_profile_fn(self, problem_shape, baseline_us):
        """Create a profiling callback for the engine's inner refinement loop."""
        def fn(code, strategy, round_num):
            temp = KernelCandidate(
                code=code,
                strategy=f"{strategy}_inner",
                round_num=round_num,
                compile_ok=bool(code),
            )
            self._profile_candidate(temp, problem_shape, baseline_us)
            error = temp.compile_error
            if not error and not temp.compile_ok:
                error = "Rejected: code did not pass validation checks"
            if not error and not temp.correct:
                error = "Correctness failure: output mismatch"
            return {
                "compile_ok": temp.compile_ok,
                "correct": temp.correct,
                "speedup": temp.speedup,
                "binary_speedup": (temp.metrics or {}).get("binary_speedup"),
                "metrics": temp.metrics,
                "branch_family": temp.bottleneck,
                "error": error,
            }
        return fn

    def _profile_candidates_parallel(
        self,
        candidates: list,
        problem_shape: tuple,
        baseline_us: float,
    ) -> list:
        """Profile all candidates in parallel using a thread pool.
        Returns list of (candidate, KernelMetrics) tuples preserving input order."""
        if not candidates:
            return []

        results = [None] * len(candidates)

        max_workers = max(1, min(len(candidates), self.max_profile_workers))
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            future_to_idx = {
                pool.submit(self._profile_candidate, c, problem_shape, baseline_us): i
                for i, c in enumerate(candidates)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    m = future.result()
                except Exception as e:
                    logger.error("Profiling thread failed for [%s]: %s",
                                 candidates[idx].strategy, e)
                    m = None
                results[idx] = (candidates[idx], m or KernelMetrics())

        return results

    def run(self) -> KernelCandidate:
        """Execute the full beam search. Returns the best KernelCandidate."""
        env           = self.env
        problem_shape = env.problem_shapes[0]

        baseline_us = env.baseline_us_reported
        if baseline_us is None or baseline_us <= 0:
            raise RuntimeError(
                f"Invalid speedup baseline for {env.kernel_name}: {baseline_us!r}. "
                "Measure a valid FlashInfer or explicit reference baseline before search."
            )
        search_naive_us, baseline_cm = self.measure_search_baseline(problem_shape)
        if search_naive_us:
            logger.info("Naive reference kernel (harness): %.3f us", search_naive_us)
        if env.official_baseline:
            logger.info("FlashInfer baseline (speedup denominator): %.3f us", baseline_us)
        else:
            logger.warning(
                "Reference fallback baseline (UNOFFICIAL denominator): %.3f us",
                baseline_us,
            )

        logger.info("="*60)
        logger.info("Beam search: kernel=%s beam_width=%d rounds=%d",
                    env.kernel_name, self.beam_w, self.rounds)
        logger.info("="*60)

        # ── Round 0: Decompose + generate initial beams ────────────────────
        all_strategies = self.engine.run_decompose()
        strategies = all_strategies[:self.beam_w]
        self._reserve_strategies = all_strategies[self.beam_w:]
        logger.info("Active strategies: %s", [s.get("name", s) if isinstance(s, dict) else s for s in strategies])
        logger.info("Reserve strategies: %s", [s.get("name", s) if isinstance(s, dict) else s for s in self._reserve_strategies])
        env.current_round = 0

        kernel_slice = env.kernel_src
        profile_fn = self._make_inner_profile_fn(problem_shape, baseline_us)
        candidates = self.engine.run_generate_beams(
            strategies=strategies, kernel_slice=kernel_slice, round_num=0,
            profile_fn=profile_fn,
        )

        # Separate pre-profiled (from tool loop) vs unprofiled (from one-shot fallback)
        pre_profiled = [c for c in candidates if c.metrics]
        need_profiling = [c for c in candidates if not c.metrics]

        if need_profiling:
            logger.info("Profiling %d one-shot beams in parallel...", len(need_profiling))
            fresh_metrics = self._profile_candidates_parallel(
                need_profiling, problem_shape, baseline_us)
        else:
            fresh_metrics = []

        metrics_list = [
            (c, metrics_from_dict(c.metrics) if c.metrics else KernelMetrics())
            for c in pre_profiled
        ] + fresh_metrics

        for c, _ in metrics_list:
            env.optimization_history.record(c)
            logger.info("  %s", c.summary())

        # Collect failed round-0 strategies for retry — these were the LLM's
        # top picks but failed due to implementation bugs (not bad ideas)
        self._failed_strategies = []
        for (c, _), strat in zip(metrics_list, strategies):
            if not c.is_viable():
                error = c.compile_error or "Unknown failure"
                self._failed_strategies.append((strat, error))
        if self._failed_strategies:
            names = [s.get("name", s) if isinstance(s, dict) else s
                     for s, _ in self._failed_strategies]
            logger.info("Failed strategies saved for retry: %s", names)

        survivors = self.selector.select_survivors(metrics_list, max_survivors=self.beam_w)
        survivors = self._prune_stale_families(survivors)
        for s in survivors:
            if s.prev_metrics is None:
                s.prev_metrics = s.metrics
            # Initialize best-code tracking for refinement
            s.best_code = s.code
            s.best_speedup = s.speedup

        # ── Rounds 1-N: Refine ─────────────────────────────────────────────
        for round_num in range(1, self.rounds + 1):
            env.current_round = round_num

            if env.over_budget():
                logger.warning("Budget exhausted at round %d", round_num)
                break
            if not survivors:
                logger.warning("No viable survivors — stopping early")
                break

            round_start_best = max((s.speedup for s in survivors), default=0.0)
            logger.info("--- Round %d --- (best so far: %.3fx)", round_num, round_start_best)

            # Split survivors into refineable vs stagnant
            to_refine = []
            stagnant = []
            for s in survivors:
                if s.refine_attempts >= max(3, self.plateau_refine_attempts + 1):
                    stagnant.append(s)
                    logger.info("  Retiring stagnant [%s] after %d failed refine attempts",
                                s.strategy, s.refine_attempts)
                else:
                    to_refine.append(s)
                    # Don't increment here — increment based on outcome below

            # Refine non-stagnant survivors (inner loop profiles via callback)
            refined = []
            if to_refine:
                profile_fn = self._make_inner_profile_fn(problem_shape, baseline_us)
                refined = self.engine.run_refine_beams(
                    to_refine, round_num, profile_fn=profile_fn)

            # Fill stagnant slots: first retry failed strategies, then use reserves
            fresh = []
            slots_available = len(stagnant)
            best_code = max(survivors, key=lambda c: c.speedup).code if survivors else kernel_slice

            # Priority 1: retry failed strategies with error context
            if slots_available > 0 and self._failed_strategies:
                n_retry = min(slots_available, len(self._failed_strategies))
                to_retry = self._failed_strategies[:n_retry]
                self._failed_strategies = self._failed_strategies[n_retry:]

                retry_strats = []
                for strat, error in to_retry:
                    name = strat.get("name", strat) if isinstance(strat, dict) else strat
                    what = strat.get("what", "") if isinstance(strat, dict) else ""
                    retry_strats.append({
                        "name": name,
                        "what": (what + f"\n\nWARNING — Previous attempt FAILED:\n{error[:400]}\n"
                                 "Fix the bug. Common cause: __syncthreads() inside if/else "
                                 "causes deadlock — all threads in a block MUST hit every barrier."),
                    })

                logger.info("  Retrying %d failed strategies: %s",
                            n_retry, [s["name"] for s in retry_strats])
                fresh = self.engine.run_generate_beams(
                    strategies=retry_strats, kernel_slice=best_code,
                    round_num=round_num, profile_fn=profile_fn,
                )
                slots_available -= n_retry

            # Priority 2: fresh reserve strategies for remaining slots
            if slots_available > 0 and self._reserve_strategies:
                n_fresh = min(slots_available, len(self._reserve_strategies))
                fresh_strats = self._reserve_strategies[:n_fresh]
                self._reserve_strategies = self._reserve_strategies[n_fresh:]
                logger.info("  Replacing %d stagnant beams with fresh strategies: %s",
                            n_fresh, [s.get("name", s) if isinstance(s, dict) else s for s in fresh_strats])
                fresh += self.engine.run_generate_beams(
                    strategies=fresh_strats, kernel_slice=best_code,
                    round_num=round_num, profile_fn=profile_fn,
                )

            # Refined candidates are already profiled by the inner loop.
            # Fresh beams may be pre-profiled (tool loop) or need profiling (one-shot).
            refined_metrics = [
                (c, metrics_from_dict(c.metrics) if c.metrics else KernelMetrics())
                for c in refined
            ]
            fresh_pre_profiled = [c for c in fresh if c.metrics]
            fresh_need_profiling = [c for c in fresh if not c.metrics]
            if fresh_need_profiling:
                fresh_metrics = self._profile_candidates_parallel(
                    fresh_need_profiling, problem_shape, baseline_us)
            else:
                fresh_metrics = []

            # Combine: refined (already profiled) + fresh pre-profiled + fresh just-profiled
            new_metrics = refined_metrics + [
                (c, metrics_from_dict(c.metrics) if c.metrics else KernelMetrics())
                for c in fresh_pre_profiled
            ] + fresh_metrics

            for c, _ in new_metrics:
                env.optimization_history.record(c)
                logger.info("  New: %s", c.summary())

            # Track refinement outcomes on parent survivors + build history.
            # A parent may now produce multiple child branches once it crosses 1.0x.
            parent_map = {parent.strategy: parent for parent in to_refine}
            refined_groups = {}
            for refined_c, metrics in refined_metrics:
                refined_groups.setdefault(refined_c.parent_strategy or "", []).append((refined_c, metrics))

            for parent in to_refine:
                children = refined_groups.get(parent.strategy, [])
                if not children:
                    continue

                best_child, _ = max(
                    children,
                    key=lambda item: (
                        int(item[0].compile_ok and item[0].correct),
                        item[0].speedup,
                    ),
                )

                for child, _ in children:
                    entry = {
                        "round": round_num,
                        "strategy": child.strategy,
                        "speedup": child.speedup,
                        "strategy_desc": getattr(child, "strategy_desc", ""),
                        "branch": (child.plan_branch or {}).get("name", ""),
                    }
                    if not child.compile_ok:
                        entry["outcome"] = "compile_fail"
                    elif not child.correct:
                        entry["outcome"] = "correctness_fail"
                    elif child.speedup < parent.speedup - 0.001:
                        entry["outcome"] = "regression"
                    elif child.speedup > parent.speedup + 0.02:
                        entry["outcome"] = "improved"
                    else:
                        entry["outcome"] = "stagnant"
                    parent.refinement_history.append(entry)
                    child.refinement_history = list(parent.refinement_history)

                if not best_child.compile_ok:
                    parent.last_refine_error = best_child.compile_error or "Compile failure"
                    parent.refine_attempts += 1
                elif not best_child.correct:
                    parent.last_refine_error = (
                        best_child.compile_error
                        or "Correctness failure (output mismatch or kernel hung)"
                    )
                    parent.refine_attempts += 1
                elif best_child.speedup < parent.speedup - 0.001:
                    msg = f"Your refinement was SLOWER: {best_child.speedup:.3f}x vs {parent.speedup:.3f}x."
                    rm = best_child.metrics or {}
                    rc = rm.get("_compiler", {})
                    pm = parent.metrics or {}
                    pc = pm.get("_compiler", {})
                    if rc and pc:
                        r_regs = rc.get("registers_per_thread", 0)
                        p_regs = pc.get("registers_per_thread", 0)
                        r_occ = rm.get("sm_occupancy", 0)
                        p_occ = pm.get("sm_occupancy", 0)
                        if r_regs != p_regs or r_occ != p_occ:
                            msg += f" Registers: {p_regs}->{r_regs}, Occupancy: {p_occ:.0f}%->{r_occ:.0f}%."
                    parent.last_refine_error = msg
                    parent.refine_attempts += 1
                elif best_child.speedup > parent.speedup + 0.02:
                    parent.last_refine_error = ""
                    parent.refine_attempts = 0
                    if best_child.speedup > (parent.best_speedup or 0):
                        best_child.best_code = best_child.code
                        best_child.best_speedup = best_child.speedup
                else:
                    parent.last_refine_error = (
                        f"Refinement produced same speedup ({best_child.speedup:.3f}x vs "
                        f"{parent.speedup:.3f}x). Review the metrics to see if your change actually affected hardware execution."
                    )
                    parent.refine_attempts += 1

                if best_child.metrics:
                    parent.prev_metrics = best_child.metrics

            # Problem 2: only promote refinements that actually improved
            improved_new = []
            for refined_c, m in refined_metrics:
                parent = parent_map.get(refined_c.parent_strategy)
                if parent and refined_c.speedup > parent.speedup:
                    improved_new.append((refined_c, m))
            for fresh_c, m in [
                (c, metrics_from_dict(c.metrics) if c.metrics else KernelMetrics())
                for c in fresh_pre_profiled
            ] + fresh_metrics:
                improved_new.append((fresh_c, m))

            crossover_candidate = self._spawn_crossover_candidate(
                survivors, round_num, problem_shape, baseline_us
            )
            if crossover_candidate is not None:
                improved_new.append(crossover_candidate)

            all_candidates = [
                (s, metrics_from_dict(s.metrics) if s.metrics else KernelMetrics())
                for s in survivors
            ] + improved_new
            prev_survivor_ids = {id(s) for s in survivors}
            survivors = self.selector.select_survivors(all_candidates, max_survivors=self.beam_w)
            survivors = self._prune_stale_families(survivors)
            # Only set prev_metrics on NEW entrants; carried-over survivors keep theirs
            for s in survivors:
                if id(s) not in prev_survivor_ids:
                    s.prev_metrics = s.metrics

            # Early stopping: if best speedup improved by less than threshold, stop
            round_end_best = max((s.speedup for s in survivors), default=0.0)
            improvement = round_end_best - round_start_best
            if round_num >= 1 and improvement < self.early_stop_min_improvement:
                logger.info(
                    "Early stop at round %d: improvement %.4fx < threshold %.4fx (best=%.3fx)",
                    round_num, improvement, self.early_stop_min_improvement, round_end_best,
                )
                break

        # ── Final: Combine Orthogonal Survivors (Tournament Bracket) ─────────────
        top_for_combine = self.selector.select_for_combination(survivors)
        logger.info("Combining %d top beams via Tournament Bracket", len(top_for_combine))

        current_pool = top_for_combine
        while len(current_pool) > 1:
            next_pool = []
            for i in range(0, len(current_pool), 2):
                if i + 1 < len(current_pool):
                    merged = self.engine.combine([current_pool[i], current_pool[i+1]])
                    self._profile_candidate(merged, problem_shape, baseline_us)
                    
                    # If regression or compilation failure, fallback to the best parent of the pair
                    best_parent = max(current_pool[i], current_pool[i+1], key=lambda c: c.speedup)
                    if not merged.compile_ok or merged.speedup < best_parent.speedup * 0.95:
                        logger.warning("Merge regressed (%.3fx < %.3fx) — reverting to best parent", 
                                       merged.speedup, best_parent.speedup)
                        next_pool.append(best_parent)
                    else:
                        logger.info("Merge succeeded! Speedup compounded: %.3fx", merged.speedup)
                        next_pool.append(merged)
                else:
                    # Odd-man out advances to next bracket automatically
                    next_pool.append(current_pool[i])
            current_pool = next_pool

        final = current_pool[0] if current_pool else (survivors[0] if survivors else candidates[0])

        logger.info("="*60)
        logger.info("Search complete: %s", final.summary())
        logger.info("Total API cost: $%.4f", env.total_api_cost_usd)
        logger.info("="*60)

        # Clean up async client to prevent 'Event loop is closed' errors
        self.engine.close()

        return final
