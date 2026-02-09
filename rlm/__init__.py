"""rlm — Recursive Language Model kernel optimization engine."""
from .environment import RLMEnvironment, KernelCandidate, OptimizationHistory

try:  # Optional so utility modules can import without the full runtime deps.
    from .engine import RLMEngine
except ModuleNotFoundError:  # pragma: no cover - depends on local env
    RLMEngine = None

__all__ = ["RLMEngine", "RLMEnvironment", "KernelCandidate", "OptimizationHistory"]
