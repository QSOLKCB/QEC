"""Deterministic diagnostic helper for metastable spectral-basin switching."""

from __future__ import annotations


def detect_basin_switch(
    prev_signature: dict[str, float],
    new_signature: dict[str, float],
    *,
    spectral_radius_eps: float = 1e-9,
    ipr_delta_threshold: float = 0.05,
) -> str:
    """Classify whether a signature transition indicates a metastable switch."""
    prev_radius = float(prev_signature.get("spectral_radius", 0.0))
    new_radius = float(new_signature.get("spectral_radius", 0.0))
    prev_ipr = float(prev_signature.get("mode_ipr", 0.0))
    new_ipr = float(new_signature.get("mode_ipr", 0.0))

    if abs(new_radius - prev_radius) < spectral_radius_eps and (new_ipr - prev_ipr) > ipr_delta_threshold:
        return "metastable_switch"
    return "stable"
