"""Deterministic policy for adaptive basin exploration strategy selection."""

from __future__ import annotations


def choose_exploration_strategy(state: str) -> str:
    """Map exploration state to deterministic mutation strategy."""
    if state == "LOCAL_OPTIMIZATION":
        return "GRADIENT"
    if state == "BASIN_STAGNATION":
        return "ESCAPE"
    if state == "BASIN_TRANSITION":
        return "NB_EIGENMODE"
    return "RANDOM_EXPLORATION"


def apply_early_exploration_guard(
    strategy: str,
    *,
    recent_basin_discovery_rate: float,
    threshold: float = 0.5,
) -> str:
    """Temporarily disable escape while new-basin discovery is rapid."""
    if strategy == "ESCAPE" and float(recent_basin_discovery_rate) > float(threshold):
        return "GRADIENT"
    return strategy


def apply_escape_feedback_bias(
    strategy: str,
    *,
    escape_success_rate: float,
    low_success_threshold: float = 0.1,
) -> str:
    """Bias away from ESCAPE when escape effectiveness is very low."""
    if strategy == "ESCAPE" and float(escape_success_rate) < float(low_success_threshold):
        return "NB_EIGENMODE"
    return strategy


__all__ = ["choose_exploration_strategy", "apply_early_exploration_guard", "apply_escape_feedback_bias"]
