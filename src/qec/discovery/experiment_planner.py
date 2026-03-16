"""Deterministic spectral experiment planner for closed-loop discovery."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.qec.analysis.experiment_targets import (
    detect_high_uncertainty_regions,
    generate_experiment_targets,
)
from src.qec.analysis.phase_diagram_uncertainty import estimate_phase_uncertainty


class SpectralExperimentPlanner:
    """Design deterministic follow-up experiments from phase uncertainty."""

    def __init__(self, *, uncertainty_threshold: float = 0.2, max_targets: int = 10) -> None:
        self.uncertainty_threshold = float(np.float64(uncertainty_threshold))
        self.max_targets = int(max_targets)

    def plan_experiments(
        self,
        phase_surface: dict[str, Any],
        landscape_memory: object | None,
    ) -> dict[str, Any]:
        """Plan experiments from phase uncertainty and return JSON-safe payload."""
        del landscape_memory
        uncertainty = estimate_phase_uncertainty(phase_surface)
        regions = detect_high_uncertainty_regions(uncertainty, self.uncertainty_threshold)
        targets = generate_experiment_targets(regions, max_targets=self.max_targets)

        uncertainty_map = np.asarray(uncertainty.get("uncertainty_map", []), dtype=np.float64)
        score = float(np.mean(uncertainty_map)) if uncertainty_map.size else 0.0

        return {
            "uncertainty_map": uncertainty,
            "regions": regions,
            "planned_targets": targets,
            "phase_uncertainty_score": float(np.float64(score)),
            "planner_iteration": 1,
        }
