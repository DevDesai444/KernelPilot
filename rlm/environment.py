"""
environment.py — RLM environment state management.
Loads kernel source, hardware spec, manages optimization state across rounds.
"""

from __future__ import annotations
import logging
import re
import yaml
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

PROJECT_ROOT = Path(__file__).parent.parent
logger = logging.getLogger(__name__)


@dataclass
class KernelCandidate:
    """Represents a single kernel variant at any point in the search."""
    code: str
    strategy: str
    round_num: int = 0
    metrics: dict = field(default_factory=dict)
    speedup: float = 1.0
    bottleneck: str = "unknown"
    compile_ok: bool = False
    correct: bool = False
    profile_report_path: Optional[str] = None
    prev_metrics: Optional[dict] = None  # parent's metrics for delta comparison
    compile_error: str = ""  # compiler error message for reflection
    last_refine_error: str = ""  # error from last failed refinement attempt
    refinement_history: list = field(default_factory=list)  # [{round, strategy, outcome, speedup}]
    refine_attempts: int = 0  # times this candidate was refined without improvement
    best_code: str = ""       # code that achieved best_speedup (for refinement base)
    best_speedup: float = 0.0 # best speedup seen for this beam lineage
    strategy_context: str = "" # original strategy description — anchors refinement direction
    parent_strategy: str = ""
    branch_family: str = ""
    plan_branch: dict = field(default_factory=dict)
    feedback_route: str = ""

    def is_viable(self) -> bool:
        return self.compile_ok and self.correct

    def summary(self) -> str:
        return (
            f"[{self.strategy}] round={self.round_num} "
            f"speedup={self.speedup:.3f}x "
            f"family={self.bottleneck} "
            f"compile={'ok' if self.compile_ok else 'FAIL'} "
            f"correct={'yes' if self.correct else 'NO'}"
        )


@dataclass
class OptimizationHistory:
    """Tracks everything tried across all rounds and strategies."""
    entries: list = field(default_factory=list)

    def record(self, candidate: KernelCandidate, notes: str = "") -> None:
        self.entries.append({
            "timestamp": time.time(),
            "strategy": candidate.strategy,
            "round": candidate.round_num,
            "speedup": candidate.speedup,
            "family": candidate.bottleneck,
            "compile_ok": candidate.compile_ok,
            "correct": candidate.correct,
            "notes": notes,
        })

    def best_speedup(self) -> float:
        viable = [e["speedup"] for e in self.entries
                  if e["compile_ok"] and e["correct"]]
        return max(viable) if viable else 1.0

    def strategies_tried(self) -> list:
        return list({e["strategy"] for e in self.entries})

    def to_summary_str(self) -> str:
        lines = []
        for e in self.entries:
            lines.append(
                f"  round={e['round']} strategy={e['strategy']} "
                f"speedup={e['speedup']:.3f}x "
                f"ok={e['compile_ok']} correct={e['correct']}"
            )
        return "\n".join(lines) if lines else "  (none)"


class RLMEnvironment:
    """
    Shared state object for the RLM REPL.
    Root LLM reads/writes this; sub-LLMs get slices of it.
    """

    def __init__(
        self,
        kernel_name: str,
        kernel_src_path: str,
        config_path: str = None,
        kernel_type: str = "add_rmsnorm",
        problem_shape: tuple = None,
    ):
        self.kernel_name = kernel_name
        self.kernel_src_path = Path(kernel_src_path)
        self.kernel_src_raw: str = self.kernel_src_path.read_text()
        self.kernel_src: str = self._expand_local_includes(self.kernel_src_raw)
        self.kernel_type: str = kernel_type

        hw_spec_path = PROJECT_ROOT / "config" / "b200_spec.yaml"
        with open(hw_spec_path) as f:
            self.hw_spec: dict = yaml.safe_load(f)

        search_cfg_path = config_path or PROJECT_ROOT / "config" / "search_config.yaml"
        with open(search_cfg_path) as f:
            self.search_config: dict = yaml.safe_load(f)

        self.profile_report: Optional[dict] = None
        self.baseline_us: Optional[float] = None
        self.baseline_us_reported: Optional[float] = None
        self.baseline_source: str = "unknown"
        self.official_baseline: bool = False
        self.baseline_naive_us: Optional[float] = None
        self.baseline_compiler_metrics = None  # CompilerMetrics from reference kernel
        self._pricing_warnings: set[str] = set()

        # Task-specific shape takes priority over config shapes
        if problem_shape is not None:
            self.problem_shapes: list = [problem_shape]
        else:
            self.problem_shapes: list = [
                tuple(s) for s in self.search_config["eval"]["problem_shapes"]
            ]

        self.optimization_history = OptimizationHistory()
        self.current_round: int = 0
        self.total_api_cost_usd: float = 0.0
        self.candidates: list = []
        self.hack_rejections: list = []
        # Pass rate tracking (first-try, no retries)
        self.total_attempts: int = 0
        self.compile_passes: int = 0
        self.correctness_passes: int = 0

    # ── Preprocessing ──────────────────────────────────────────────────────────

    def _expand_local_includes(self, src: str) -> str:
        """Expand local #include directives so the LLM sees helper function signatures."""
        lines = src.split("\n")
        result = []
        for line in lines:
            stripped = line.strip()
            if stripped.startswith('#include "') and stripped.endswith('"'):
                rel_path = stripped[len('#include "'):-1]
                header = self.kernel_src_path.parent / rel_path
                if header.exists():
                    result.append(f"// === expanded from {rel_path} ===")
                    result.append(header.read_text())
                    result.append(f"// === end {rel_path} ===")
                    continue
            result.append(line)
        return "\n".join(result)

    # ── Kernel source navigation ──────────────────────────────────────────────

    def find_kernel_function(self, pattern: str = r"__global__\s+void\s+(\w+)") -> list:
        return re.findall(pattern, self.kernel_src)

    def find_hot_loop(self) -> tuple:
        lines = self.kernel_src.split("\n")
        hot_start = 0
        depth_max = 0
        depth = 0
        for i, line in enumerate(lines):
            depth += line.count("{") - line.count("}")
            if "for" in line and ("[" in line or "load" in line.lower()):
                if depth > depth_max:
                    depth_max = depth
                    hot_start = i
        hot_end = min(hot_start + 30, len(lines))
        return hot_start, hot_end

    def get_hot_loop_src(self) -> str:
        s, e = self.find_hot_loop()
        lines = self.kernel_src.split("\n")
        return "\n".join(lines[s:e])

    def get_kernel_slice(self, start: int, end: int) -> str:
        return self.kernel_src[start:end]

    def _strip_comments(self, src: str) -> str:
        src = re.sub(r"/\*.*?\*/", "", src, flags=re.S)
        src = re.sub(r"//.*", "", src)
        return src

    def _extract_int_constants(self, src: str) -> dict[str, int]:
        constants: dict[str, int] = {}
        for name, expr in re.findall(r"^\s*#define\s+([A-Za-z_]\w*)\s+([^\n]+)$", src, flags=re.M):
            value = self._eval_int_expr(expr.strip(), constants)
            if value is not None:
                constants[name] = value
        if "BLOCK_THREADS" in constants:
            constants["blockDim.x"] = constants["BLOCK_THREADS"]
        return constants

    def _eval_int_expr(self, expr: str, constants: dict[str, int]) -> Optional[int]:
        expr = expr.strip()
        if not expr:
            return None
        for name, value in sorted(constants.items(), key=lambda item: -len(item[0])):
            expr = expr.replace(name, str(value))
        expr = re.sub(r"\b(static_cast|reinterpret_cast|const)\b", "", expr)
        expr = re.sub(r"\((?:int|unsigned|size_t|long|short)\)", "", expr)
        if re.search(r"[A-Za-z_]", expr):
            return None
        if not re.fullmatch(r"[0-9xXa-fA-F+\-*/%<>&|() \t]+", expr):
            return None
        try:
            value = eval(expr, {"__builtins__": {}}, {})
        except Exception:
            return None
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, (int, float)):
            return int(value)
        return None

    def _find_matching_delim(self, text: str, start: int, open_ch: str, close_ch: str) -> int:
        depth = 0
        for idx in range(start, len(text)):
            if text[idx] == open_ch:
                depth += 1
            elif text[idx] == close_ch:
                depth -= 1
                if depth == 0:
                    return idx
        return -1

