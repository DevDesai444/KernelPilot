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

    def _estimate_for_loop_iterations(self, header: str, constants: dict[str, int]) -> Optional[int]:
        parts = [part.strip() for part in header.split(";")]
        if len(parts) != 3:
            return None
        init, cond, update = parts
        init_match = re.search(r"([A-Za-z_]\w*)\s*=\s*(.+)$", init)
        if not init_match:
            return None
        var = init_match.group(1)
        start = self._eval_int_expr(init_match.group(2), constants)
        if start is None:
            return None

        if re.fullmatch(rf"{re.escape(var)}\s*>>=\s*1", update) or re.fullmatch(
            rf"{re.escape(var)}\s*=\s*{re.escape(var)}\s*>>\s*1", update
        ):
            if re.search(rf"\b{re.escape(var)}\b\s*>\s*0", cond):
                count = 0
                value = start
                while value > 0:
                    count += 1
                    value >>= 1
                return count

        if re.fullmatch(rf"{re.escape(var)}\s*/=\s*2", update) or re.fullmatch(
            rf"{re.escape(var)}\s*=\s*{re.escape(var)}\s*/\s*2", update
        ):
            if re.search(rf"\b{re.escape(var)}\b\s*>\s*0", cond):
                count = 0
                value = start
                while value > 0:
                    count += 1
                    value //= 2
                return count

        step = None
        if re.fullmatch(rf"(?:\+\+{re.escape(var)}|{re.escape(var)}\+\+)", update):
            step = 1
        else:
            step_match = re.fullmatch(rf"{re.escape(var)}\s*\+=\s*(.+)", update)
            if step_match:
                step = self._eval_int_expr(step_match.group(1), constants)
        if step is not None and step > 0:
            cond_match = re.search(rf"\b{re.escape(var)}\b\s*(<|<=)\s*(.+)$", cond)
            if cond_match:
                end = self._eval_int_expr(cond_match.group(2), constants)
                if end is not None:
                    if cond_match.group(1) == "<=":
                        end += 1
                    if start < end:
                        return max(0, (end - start + step - 1) // step)
        return None

    def _count_runtime_syncs(self, src: str, constants: dict[str, int]) -> int:
        total = 0
        cursor = 0
        while cursor < len(src):
            sync_idx = src.find("__syncthreads", cursor)
            for_match = re.search(r"\bfor\s*\(", src[cursor:])
            for_idx = cursor + for_match.start() if for_match else -1

            if sync_idx == -1 and for_idx == -1:
                break
            if sync_idx != -1 and (for_idx == -1 or sync_idx < for_idx):
                total += 1
                cursor = sync_idx + len("__syncthreads")
                continue

            header_open = src.find("(", for_idx)
            header_close = self._find_matching_delim(src, header_open, "(", ")")
            if header_open == -1 or header_close == -1:
                cursor = for_idx + 3
                continue
            header = src[header_open + 1:header_close]
            body_start = header_close + 1
            while body_start < len(src) and src[body_start].isspace():
                body_start += 1

            if body_start < len(src) and src[body_start] == "{":
                body_end = self._find_matching_delim(src, body_start, "{", "}")
                if body_end == -1:
                    cursor = body_start + 1
                    continue
                body = src[body_start + 1:body_end]
                cursor = body_end + 1
            else:
                stmt_end = src.find(";", body_start)
                if stmt_end == -1:
                    cursor = body_start + 1
                    continue
                body = src[body_start:stmt_end + 1]
                cursor = stmt_end + 1

            body_syncs = self._count_runtime_syncs(body, constants)
            iterations = self._estimate_for_loop_iterations(header, constants)
            total += body_syncs * max(iterations or 1, 1)
        return total

    def estimate_runtime_syncthreads(self) -> tuple[int, int]:
        src = self._strip_comments(self.kernel_src_raw)
        constants = self._extract_int_constants(src)
        source_occurrences = src.count("__syncthreads")
        runtime_estimate = self._count_runtime_syncs(src, constants)
        return source_occurrences, max(runtime_estimate, source_occurrences)

    def count_memory_ops(self) -> dict:
        # Use raw source (without expanded includes) to analyze the actual kernel code
        src = self._strip_comments(self.kernel_src_raw)
        syncthreads_source, syncthreads_runtime = self.estimate_runtime_syncthreads()
        return {
            "loads":       len(re.findall(r"\b(?:__ldg|ld\.global|tex1Dfetch)\b", src)),
            "stores":      len(re.findall(r"\b(?:__stg|st\.global|atomicAdd)\b", src)),
            "float4":      src.count("float4"),
            "cp_async":    src.count("cp.async"),
            "tma":         src.count("tma_load") + src.count("tcgen05"),
            "syncthreads": syncthreads_runtime,
            "syncthreads_source": syncthreads_source,
            "shfl":        src.count("__shfl"),
        }

    def detect_missing_optimizations(self) -> list:
        """Kernel-type-aware optimization detection.

        Uses kernel-specific ideal strategy selection instead of keyword grep.
        Falls back to legacy grep-based detection for unknown kernel types.
        """
        from search.strategy_bank import select_for_kernel, KERNEL_IDEAL_STRATEGIES

        tried = self.optimization_history.strategies_tried()

        # Use kernel-aware selection if we know the kernel type
        if self.kernel_type in KERNEL_IDEAL_STRATEGIES:
            return select_for_kernel(
                kernel_type=self.kernel_type,
                tried=tried,
                beam_width=self.search_config["beam"]["width"],
            )

        # Legacy fallback for unknown kernel types
        ops = self.count_memory_ops()
        enabled = self.search_config["strategies"]["enabled"]
        missing = []
        if ops["float4"] == 0 and "vectorize_loads" in enabled:
            missing.append("vectorize_loads")
        if ops["tma"] == 0 and "tma_prefetch" in enabled:
            missing.append("tma_prefetch")
        if ops["syncthreads"] > 2 and ops["shfl"] == 0 and "warp_reduction" in enabled:
            missing.append("warp_reduction")
        if "fuse_passes" in enabled and self.kernel_src.count("__global__") > 1:
            missing.append("fuse_passes")
        return missing

    # ── Cost tracking ─────────────────────────────────────────────────────────

    def record_api_cost(self, tokens_in: int, tokens_out: int, model: str) -> float:
        _costs_per_million = {
            "claude-3-5-sonnet-20241022": {"in": 3.0,  "out": 15.0},
            "claude-sonnet-4-6":         {"in": 3.0,  "out": 15.0},
            "claude-opus-4-6":           {"in": 15.0, "out": 75.0},
            "claude-haiku-4-6":          {"in": 0.25, "out": 1.25},
            "claude-3-haiku-20240307":    {"in": 0.25, "out": 1.25},
            "claude-haiku-4-5-20251001":  {"in": 0.25, "out": 1.25},
            "claude-3-opus-20240229":     {"in": 15.0, "out": 75.0},
        }
        pricing_cfg = self.search_config.get("pricing", {})
        configured = pricing_cfg.get(model)
        if configured is not None:
            p = configured
        elif model in _costs_per_million:
            p = _costs_per_million[model]
        else:
            default_pricing = pricing_cfg.get("default_model_pricing")
            if default_pricing is not None:
                p = default_pricing
                if model not in self._pricing_warnings:
                    logger.warning(
                        "Using configured default pricing for unknown model '%s': in=%s out=%s",
                        model, p.get("in"), p.get("out"),
                    )
                    self._pricing_warnings.add(model)
            else:
                p = {
                    "in": max(item["in"] for item in _costs_per_million.values()),
                    "out": max(item["out"] for item in _costs_per_million.values()),
                }
                if model not in self._pricing_warnings:
                    logger.warning(
                        "Unknown model '%s' has no configured pricing; using conservative fallback in=%s out=%s",
                        model, p["in"], p["out"],
                    )
                    self._pricing_warnings.add(model)
        cost = (tokens_in * p["in"] + tokens_out * p["out"]) / 1_000_000
        self.total_api_cost_usd += cost
        return cost

    def budget_remaining(self) -> float:
        return self.search_config["cost_control"]["max_total_api_cost_usd"] - self.total_api_cost_usd

    def over_budget(self) -> bool:
        return self.total_api_cost_usd >= self.search_config["cost_control"]["max_total_api_cost_usd"]

    # ── State summary ─────────────────────────────────────────────────────────

    def state_summary(self) -> str:
        ops = self.count_memory_ops()
        missing = self.detect_missing_optimizations()
        return (
            f"=== RLM Environment State ===\n"
            f"Kernel:        {self.kernel_name}\n"
            f"Source:        {self.kernel_src_path} ({len(self.kernel_src)} chars)\n"
            f"Round:         {self.current_round}\n"
            f"Candidates:    {len(self.candidates)}\n"
            f"Best speedup:  {self.optimization_history.best_speedup():.3f}x\n"
            f"API cost:      ${self.total_api_cost_usd:.4f} / "
            f"${self.search_config['cost_control']['max_total_api_cost_usd']:.2f}\n"
            f"Memory ops:    loads={ops['loads']} stores={ops['stores']} "
            f"float4={ops['float4']} tma={ops['tma']}\n"
            f"Sync ops:      syncthreads~={ops['syncthreads']} "
            f"(src={ops['syncthreads_source']}) shfl={ops['shfl']}\n"
            f"Missing opts:  {', '.join(missing) or 'none detected'}\n"
            f"Strategies tried: {', '.join(self.optimization_history.strategies_tried()) or 'none'}\n"
            f"History:\n{self.optimization_history.to_summary_str()}\n"
        )
