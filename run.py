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


