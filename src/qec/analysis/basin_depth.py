"""v14.5.0 — Deterministic basin-depth scoring for adaptive beam planning."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BasinDepthConfig:
    """Configuration for deterministic basin-depth scoring."""

    w1: float = 1.0
    w2: float = 1.0
    w3: float = 1.0
    w4: float = 1.0
    slope_window: int = 3
    precision: int = 12


def compute_smoothed_energy_slope(energy_deltas: list[float], *, last_k: int = 3, precision: int = 12) -> float:
    """Compute deterministic mean slope over the trailing energy deltas."""
    if not energy_deltas:
        return 0.0
    k = min(max(1, int(last_k)), len(energy_deltas))
    trailing = np.asarray(energy_deltas[-k:], dtype=np.float64)
    return round(float(np.mean(trailing)), int(precision))


def compute_basin_depth(
    *,
    flow_ipr: float,
    edge_reuse_rate: float,
    unstable_mode_persistence: float,
    energy_deltas: list[float] | tuple[float, ...],
    config: BasinDepthConfig | None = None,
) -> dict[str, float]:
    """Compute deterministic basin depth from diagnostic signals."""
    cfg = config if config is not None else BasinDepthConfig()
    energy_slope_smoothed = compute_smoothed_energy_slope(
        [float(x) for x in energy_deltas],
        last_k=cfg.slope_window,
        precision=cfg.precision,
    )
    depth = (
        float(cfg.w1) * float(flow_ipr)
        + float(cfg.w2) * float(edge_reuse_rate)
        + float(cfg.w3) * float(unstable_mode_persistence)
        - float(cfg.w4) * abs(float(energy_slope_smoothed))
    )
    return {
        "basin_depth": round(float(depth), cfg.precision),
        "energy_slope_smoothed": energy_slope_smoothed,
    }


__all__ = [
    "BasinDepthConfig",
    "compute_basin_depth",
    "compute_smoothed_energy_slope",
]
