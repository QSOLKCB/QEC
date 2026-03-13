"""Deterministic basin-depth scoring and adaptive beam-width mapping (v14.5.0)."""

from __future__ import annotations

import math
from typing import Mapping

DEFAULT_W1_FLOW_IPR = 1.0
DEFAULT_W2_EDGE_REUSE = 1.0
DEFAULT_W3_UNSTABLE_PERSISTENCE = 0.5
DEFAULT_W4_ENERGY_SLOPE = 0.5
DEFAULT_DEPTH_SCALE = 3.0
DEFAULT_BEAM_MIN = 3
DEFAULT_BEAM_MAX = 10


def compute_basin_depth(
    *,
    flow_ipr: float,
    edge_reuse_rate: float,
    unstable_mode_persistence: float,
    energy_slope: float,
    w1: float = DEFAULT_W1_FLOW_IPR,
    w2: float = DEFAULT_W2_EDGE_REUSE,
    w3: float = DEFAULT_W3_UNSTABLE_PERSISTENCE,
    w4: float = DEFAULT_W4_ENERGY_SLOPE,
) -> float:
    """Compute deterministic scalar depth score for a metastable basin."""
    return float(
        float(w1) * float(flow_ipr)
        + float(w2) * float(edge_reuse_rate)
        + float(w3) * float(unstable_mode_persistence)
        - float(w4) * abs(float(energy_slope))
    )


def basin_depth_from_diagnostics(
    diagnostics: Mapping[str, float] | None,
    *,
    w1: float = DEFAULT_W1_FLOW_IPR,
    w2: float = DEFAULT_W2_EDGE_REUSE,
    w3: float = DEFAULT_W3_UNSTABLE_PERSISTENCE,
    w4: float = DEFAULT_W4_ENERGY_SLOPE,
) -> float:
    """Extract and compute basin depth from diagnostics outputs."""
    d = diagnostics or {}
    return compute_basin_depth(
        flow_ipr=float(d.get("flow_ipr", 0.0)),
        edge_reuse_rate=float(d.get("edge_reuse_rate", 0.0)),
        unstable_mode_persistence=float(d.get("unstable_mode_persistence", 0.0)),
        energy_slope=float(d.get("energy_slope", 0.0)),
        w1=w1,
        w2=w2,
        w3=w3,
        w4=w4,
    )


def adaptive_beam_width(
    basin_depth: float,
    *,
    beam_min: int = DEFAULT_BEAM_MIN,
    beam_max: int = DEFAULT_BEAM_MAX,
    depth_scale: float = DEFAULT_DEPTH_SCALE,
) -> int:
    """Map basin depth to bounded beam width using deterministic sigmoid scaling."""
    lo = int(beam_min)
    hi = int(beam_max)
    if lo < 1:
        lo = 1
    if hi < lo:
        hi = lo
    x = float(depth_scale) * float(basin_depth)
    sigma = 1.0 / (1.0 + math.exp(-x))
    width = lo + int(math.floor((hi - lo) * sigma))
    if width < lo:
        return lo
    if width > hi:
        return hi
    return int(width)
