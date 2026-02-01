"""
diversity_selector.py — Family-aware beam diversity preservation.
Keeps the best candidate from each strategy family rather than top-K globally.
"""

from __future__ import annotations
import logging

logger = logging.getLogger(__name__)


class DiversitySelector:
    """
    Selects survivors while preserving diversity across strategy families.
    Avoids collapsing all beams to the same optimization direction.
    """

    def __init__(self, config: dict):
        self.mode          = config["beam"].get("diversity_mode", "family_diverse")
        self.combine_top_k = config["beam"].get("combine_top_k", 2)

    @staticmethod
    def _family(candidate) -> str:
        return (
            getattr(candidate, "branch_family", "")
            or getattr(candidate, "parent_strategy", "")
            or candidate.strategy.split("__", 1)[0]
        )

    def select_survivors(
        self,
        candidates_with_metrics: list,
        max_survivors: int = 4,
    ) -> list:
        """Select diverse survivors; at most max_survivors, one per strategy family."""
        if self.mode == "family_diverse":
            return self._cluster_select(candidates_with_metrics, max_survivors)
        return self._top_k_select(candidates_with_metrics, max_survivors)

