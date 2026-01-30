"""
strategy_bank.py — Predefined optimization strategies with metadata.
Includes kernel-type-aware selection to pick the best 4 per kernel.
"""

from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class Strategy:
    name: str
    display_name: str
    description: str
    targets_bottleneck: list
    priority: int
    applicable_kernels: list = field(default_factory=list)  # empty = all
    requires: list = field(default_factory=list)
    conflicts: list = field(default_factory=list)

    def applies_to(self, bottleneck: str) -> bool:
        return bottleneck in self.targets_bottleneck or not self.targets_bottleneck

