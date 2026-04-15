"""
run.py — Entry point: solve all WaferBench NVFP4 kernel problems.

Usage:
    python run.py                          # optimize all kernels
    python run.py --kernel add_rmsnorm     # single kernel
    python run.py --dry-run                # validate setup, no LLM calls
    python run.py --beam-width 2 --rounds 2
"""

from __future__ import annotations
import argparse
import logging
import os
import subprocess
import sys
import time
import yaml
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from rlm.env_loader import load_project_env
from rlm.environment import RLMEnvironment
from search.beam_search import BeamSearch
from eval.correctness import CorrectnessChecker
from eval.benchmark import Benchmarker, geometric_mean
from eval.waferbench_format import format_submission, save_submission, print_submission_summary
from eval import flashinfer_ref

load_project_env(PROJECT_ROOT)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("rlm_optimizer.log"),
    ],
)
logger = logging.getLogger(__name__)

WAFERBENCH_KERNELS = [
    {
        "name":        "add_rmsnorm_fp4quant_b128xh2048",
        "src":         "kernels/reference/add_rmsnorm.cu",
        "kernel_type": "add_rmsnorm",
        "description": "Fused Add+RMSNorm+NVFP4 quant (128×2048)",
        "shape":       (128, 2048),
    },
    {
        "name":        "add_rmsnorm_fp4quant_b128xh4096",
        "src":         "kernels/reference/add_rmsnorm.cu",
        "kernel_type": "add_rmsnorm",
        "description": "Fused Add+RMSNorm+NVFP4 quant (128×4096)",
        "shape":       (128, 4096),
    },
    {
        "name":        "add_rmsnorm_fp4quant_b128xh8192",
        "src":         "kernels/reference/add_rmsnorm.cu",
        "kernel_type": "add_rmsnorm",
        "description": "Fused Add+RMSNorm+NVFP4 quant (128×8192)",
        "shape":       (128, 8192),
    },
    {
        "name":        "nvfp4_quantize_m128xk14336",
        "src":         "kernels/reference/nvfp4_quantize.cu",
        "kernel_type": "nvfp4_quantize",
        "description": "BF16→NVFP4 block quantization (128×14336)",
        "shape":       (128, 14336),
    },
    {
        "name":        "silu_mul_fp4quant_b8xm256xk7168",
        "src":         "kernels/reference/silu_mul.cu",
        "kernel_type": "silu_mul",
        "description": "Fused SiLU×Mul+NVFP4 quant (8×256×7168)",
        "shape":       (8, 256, 7168),
    },
    {
        "name":        "silu_mul_fp4quant_b8xm256xk14336",
        "src":         "kernels/reference/silu_mul.cu",
        "kernel_type": "silu_mul",
        "description": "Fused SiLU×Mul+NVFP4 quant (8×256×14336)",
        "shape":       (8, 256, 14336),
    },
]


def optimize_kernel(
    kernel_def: dict,
    config: dict,
    dry_run: bool = False,
    output_dir: Path = None,
    allow_reference_baseline: bool = False,
) -> dict:
    name        = kernel_def["name"]
    src_path    = PROJECT_ROOT / kernel_def["src"]
    kernel_type = kernel_def["kernel_type"]
    shape       = kernel_def["shape"]

    logger.info("\n%s\nOptimizing: %s\n%s", "="*60, name, "="*60)

    if not src_path.exists():
        logger.error("Kernel source not found: %s", src_path)
        return {}

    require_flashinfer = bool(config.get("eval", {}).get("require_flashinfer_baseline", True))
    baseline, baseline_source = flashinfer_ref.measure_baseline_with_source(kernel_type, shape)
    official_baseline = baseline_source == "flashinfer"
    if official_baseline:
        logger.info("FlashInfer baseline for %s: %.2f us", name, baseline)
    else:
        if require_flashinfer and not allow_reference_baseline:
            logger.error(
                "FlashInfer baseline is required for %s but unavailable. "
                "Re-run with --allow-reference-baseline only for unofficial debugging.",
                name,
            )
            return {
                "kernel_name": name,
                "status": "flashinfer_baseline_required",
                "metadata": {"official_baseline": False, "baseline_source": "unavailable"},
            }
        logger.warning("FlashInfer unavailable — using reference kernel baseline (UNOFFICIAL)")
        benchmarker_tmp = Benchmarker(config, kernel_type=kernel_type)
        baseline = benchmarker_tmp._compile_and_time(src_path.read_text(), shape)
        if baseline is None:
            logger.error("Cannot measure baseline for %s", name)
            return {
                "kernel_name": name,
                "status": "baseline_measurement_failed",
                "metadata": {"official_baseline": False, "baseline_source": "reference_fallback_failed"},
            }
        baseline_source = "reference_fallback"
        logger.info("Reference kernel baseline for %s: %.2f us", name, baseline)

    env = RLMEnvironment(kernel_name=name, kernel_src_path=str(src_path),
                         kernel_type=kernel_type, problem_shape=shape)
    env.baseline_us_reported = baseline
    env.baseline_source = baseline_source
    env.official_baseline = official_baseline

    overrides = config.get("_overrides", {})
    if "beam_width" in overrides:
        env.search_config["beam"]["width"] = overrides["beam_width"]
    if "rounds" in overrides:
        env.search_config["beam"]["refine_rounds"] = overrides["rounds"]

    logger.info(env.state_summary())

    if dry_run:
        logger.info("[DRY RUN] Skipping LLM calls for %s", name)
        return {"kernel_name": name, "dry_run": True, "status": "skipped"}

    if not os.environ.get("ANTHROPIC_API_KEY"):
        logger.error("ANTHROPIC_API_KEY not set")
        sys.exit(1)

    start_time = time.time()
    beam_search = BeamSearch(env)
    best        = beam_search.run()
    elapsed     = time.time() - start_time
    logger.info("Beam search completed in %.1f seconds", elapsed)

    checker  = CorrectnessChecker(env.search_config)
    passed, max_err, msg = checker.check(best.code, env.problem_shapes[0],
                                          kernel_type=kernel_type)
    best.correct = passed
    logger.info("Correctness: %s (max_err=%.6f)", "PASS" if passed else "FAIL", max_err)

    if not passed:
        logger.error("Correctness FAILED — falling back to reference kernel")
        best.code = src_path.read_text()

    benchmarker = Benchmarker(env.search_config, kernel_type=kernel_type)

    # Reuse the baseline measured at start (before GPU was hot from search)
    # Re-measuring here gives inflated numbers due to thermal throttling
    baseline_per_shape = {}
    for s in env.problem_shapes:
        baseline_per_shape[s] = baseline
    logger.info("Using search-time baseline for eval: %.2f us", baseline)

    bench_results = benchmarker.benchmark(best.code, baseline_per_shape)

    submission = format_submission(
        kernel_name=name,
        kernel_src=best.code,
        benchmark_results=bench_results,
        metadata={
            "strategy":         best.strategy,
            "bottleneck":       best.bottleneck,
            "baseline_source":  env.baseline_source,
            "official_baseline": env.official_baseline,
            "search_rounds":    env.current_round,
            "api_cost_usd":     env.total_api_cost_usd,
            "elapsed_seconds":  elapsed,
            "correct":          passed,
            "max_abs_err":      max_err,
            "hack_rejections":  env.hack_rejections,
            "total_attempts":       env.total_attempts,
            "compile_passes":       env.compile_passes,
            "correctness_passes":   env.correctness_passes,
        },
    )

    if output_dir:
        save_submission(submission, output_dir, name)
        logger.info("Saved to %s", output_dir)

    print_submission_summary(submission)
    return submission


def _lock_gpu_clocks():
    """Lock GPU clocks to max frequency for stable benchmarking.

    Prevents thermal throttling from inflating/deflating speedup numbers
    across runs. Falls back gracefully if nvidia-smi isn't available or
    permissions are insufficient.
    """
    try:
        # Query max supported clocks
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=clocks.max.sm", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if r.returncode == 0:
            max_clock = int(r.stdout.strip().split("\n")[0])
            # Lock SM clocks to max
            r2 = subprocess.run(
                ["nvidia-smi", "-lgc", str(max_clock)],
                capture_output=True, text=True, timeout=10,
            )
            if r2.returncode == 0:
                logger.info("GPU clocks locked to %d MHz for stable benchmarking", max_clock)
            else:
                logger.warning("Could not lock GPU clocks (need root?): %s", r2.stderr.strip())
        else:
            logger.warning("Could not query GPU max clocks: %s", r.stderr.strip())
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError) as e:
        logger.warning("GPU clock locking unavailable: %s", e)


