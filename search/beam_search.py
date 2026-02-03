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
