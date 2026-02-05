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

