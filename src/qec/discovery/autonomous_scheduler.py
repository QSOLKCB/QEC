"""v43.0.0 — Autonomous deterministic discovery scheduler."""

from __future__ import annotations

from typing import Any

from src.qec.analysis.landscape_gaps import detect_landscape_gaps
from src.qec.discovery.experiment_targets import choose_experiment_target


def schedule_next_experiment(memory, gap_radius: float, max_gaps: int) -> dict[str, Any]:
    """Schedule the next autonomous experiment from landscape memory."""
    gaps = detect_landscape_gaps(memory, gap_radius=float(gap_radius), max_gaps=int(max_gaps))
    target = choose_experiment_target(gaps)
    return {
        "target_spectrum": target,
        "gap_count": int(len(gaps)),
        "strategy": "landscape_exploration",
    }
