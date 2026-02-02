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

    def _cluster_select(self, candidates_with_metrics: list, max_survivors: int) -> list:
        clusters = {}
        for candidate, metrics in candidates_with_metrics:
            if not candidate.is_viable():
                continue
            family = self._family(candidate)
            clusters.setdefault(family, []).append((candidate, metrics))

        survivors = []
        if len(clusters) == 1:
            group = list(clusters.values())[0]
            family = list(clusters.keys())[0]
            group.sort(key=lambda x: -x[0].speedup)
            for c, m in group[:max_survivors]:
                c.bottleneck = family or "unclustered"
                c.metrics = m.to_dict()
                survivors.append(c)
                logger.info("Family %s: %s speedup=%.3fx", family, c.strategy, c.speedup)
        else:
            for family, group in clusters.items():
                best_c, best_m = max(group, key=lambda x: x[0].speedup)
                best_c.bottleneck = family or "unclustered"
                best_c.metrics    = best_m.to_dict()
                survivors.append(best_c)
                logger.info("Family %s: best=%s speedup=%.3fx",
                            family, best_c.strategy, best_c.speedup)

        survivors.sort(key=lambda c: -c.speedup)
        return survivors[:max_survivors]

    def _top_k_select(self, candidates_with_metrics: list, max_survivors: int) -> list:
        viable = [(c, m) for c, m in candidates_with_metrics if c.is_viable()]
        viable.sort(key=lambda x: -x[0].speedup)
        survivors = []
        for candidate, metrics in viable[:max_survivors]:
            candidate.bottleneck = self._family(candidate) or "unclustered"
            candidate.metrics    = metrics.to_dict()
            survivors.append(candidate)
        return survivors

    def select_for_combination(self, survivors: list) -> list:
        """Select top-K for combination, preferring orthogonal strategy families."""
        if len(survivors) <= self.combine_top_k:
            return survivors

        selected = []
        seen_families = set()
        for c in sorted(survivors, key=lambda x: -x.speedup):
            family = self._family(c)
            if family not in seen_families:
                selected.append(c)
                seen_families.add(family)
            if len(selected) >= self.combine_top_k:
                break

        for c in sorted(survivors, key=lambda x: -x.speedup):
            if c not in selected:
                selected.append(c)
            if len(selected) >= self.combine_top_k:
                break

        return selected
