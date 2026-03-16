"""Opt-in deterministic spectral curriculum controller."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.qec.analysis.curriculum_metrics import curriculum_progress


_TIER_NAMES = (
    "tier_0_easy",
    "tier_1_intermediate",
    "tier_2_hard",
    "tier_3_frontier",
)


@dataclass
class SpectralCurriculumController:
    """Track deterministic curriculum-tier progression."""

    current_tier: int = 0
    tier_progress: float = 0.0
    tier_success_rate: float = 0.0

    def _tier_name(self) -> str:
        tier = int(min(max(self.current_tier, 0), len(_TIER_NAMES) - 1))
        return _TIER_NAMES[tier]

    def filter_candidates(
        self,
        candidates: list[dict[str, Any]],
        region_tiers: dict[str, list[int]],
    ) -> list[dict[str, Any]]:
        """Return candidates that belong to the current tier only."""
        allowed = set(region_tiers.get(self._tier_name(), []))
        if not allowed:
            return []
        selected: list[dict[str, Any]] = []
        for candidate in candidates:
            region_index = int(candidate.get("region_index", -1))
            if region_index in allowed:
                selected.append(candidate)
        return selected

    def update_progress(
        self,
        successful_codes: int,
        attempted_codes: int,
        success_threshold: float,
    ) -> dict[str, float | bool]:
        """Update progress metrics for the current tier."""
        metrics = curriculum_progress(
            successful_codes=successful_codes,
            attempted_codes=attempted_codes,
            threshold=success_threshold,
        )
        self.tier_success_rate = float(np.float64(metrics["success_rate"]))
        self.tier_progress = float(np.float64(metrics["progress"]))
        return metrics

    def advance_tier_if_ready(self, should_advance: bool) -> bool:
        """Advance one tier when threshold condition is met."""
        if not should_advance:
            return False
        if self.current_tier >= len(_TIER_NAMES) - 1:
            return False
        self.current_tier += 1
        self.tier_progress = 0.0
        self.tier_success_rate = 0.0
        return True

    # Backward-compatible method aliases for prior review iteration names.
    def select_curriculum_targets(
        self,
        candidates: list[dict[str, Any]],
        region_tiers: dict[str, list[int]],
    ) -> list[dict[str, Any]]:
        return self.filter_candidates(candidates, region_tiers)

    def update_curriculum_progress(
        self,
        successful_codes: int,
        attempted_codes: int,
        success_threshold: float,
    ) -> dict[str, float | bool]:
        return self.update_progress(successful_codes, attempted_codes, success_threshold)

    def advance_curriculum(self, should_advance: bool) -> bool:
        return self.advance_tier_if_ready(should_advance)
