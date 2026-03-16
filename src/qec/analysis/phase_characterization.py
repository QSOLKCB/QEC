"""Deterministic spectral phase characterization helpers."""

from __future__ import annotations

from typing import Any

import numpy as np


def _to_vector(values: Any) -> np.ndarray:
    """Convert arbitrary inputs to a flat float64 vector."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return np.zeros((0,), dtype=np.float64)
    return arr.reshape(-1).astype(np.float64, copy=False)


def _graph_density(graph: Any) -> np.float64:
    """Return deterministic density estimate from parity-check matrix."""
    H = np.asarray(graph, dtype=np.float64)
    if H.ndim != 2 or H.size == 0:
        return np.float64(0.0)
    return np.float64(np.sum(H > 0.5) / max(H.size, 1))


def compute_phase_metrics(
    graph: Any,
    spectrum: Any,
    decoder_stats: dict[str, Any] | None,
) -> dict[str, float]:
    """Compute deterministic float64 diagnostics for a discovered phase."""
    spec = _to_vector(spectrum)
    stats = decoder_stats or {}

    spectral_radius = (
        np.float64(np.max(np.abs(spec)))
        if spec.size > 0
        else np.float64(stats.get("spectral_radius", 0.0))
    )

    bethe_hessian_min_eigenvalue = (
        np.float64(np.min(spec))
        if spec.size > 0
        else np.float64(stats.get("bethe_margin", 0.0))
    )

    graph_density = _graph_density(graph)
    trap_count = np.float64(stats.get("trapping_set_count", 0.0))
    trap_signal = np.float64(stats.get("instability_score", 0.0))
    trapping_density = np.float64(
        np.clip(
            graph_density * np.float64(0.65)
            + np.float64(0.25) * np.tanh(trap_count / np.float64(10.0))
            + np.float64(0.10) * np.clip(trap_signal, 0.0, 1.0),
            0.0,
            1.0,
        )
    )

    bp_stability_score = np.float64(
        1.0
        / (
            1.0
            + np.abs(spectral_radius - np.float64(1.0))
            + np.maximum(np.float64(0.0), -bethe_hessian_min_eigenvalue)
            + np.float64(0.5) * trapping_density
        )
    )

    threshold_term = np.float64(np.exp(-np.abs(spectral_radius - np.float64(1.0))))
    estimated_threshold = np.float64(
        np.clip(
            np.float64(0.50) * bp_stability_score
            + np.float64(0.30) * (np.float64(1.0) - trapping_density)
            + np.float64(0.20) * threshold_term,
            0.0,
            1.0,
        )
    )

    return {
        "bp_stability_score": float(np.float64(bp_stability_score)),
        "trapping_density": float(np.float64(trapping_density)),
        "estimated_threshold": float(np.float64(estimated_threshold)),
        "spectral_radius": float(np.float64(spectral_radius)),
        "bethe_hessian_min_eigenvalue": float(np.float64(bethe_hessian_min_eigenvalue)),
    }


def classify_phase(metrics: dict[str, Any]) -> dict[str, str]:
    """Classify a phase with deterministic threshold rules.

    Rule order is intentionally fixed to keep labels reproducible:
    1) High trapping density dominates all other evidence.
    2) Spectral instability (large radius or negative Bethe minimum).
    3) Stable BP regime requires high stability and threshold.
    4) Remaining cases are fragile BP behavior.
    """
    trapping_density = np.float64(metrics.get("trapping_density", 0.0))
    spectral_radius = np.float64(metrics.get("spectral_radius", 0.0))
    bethe_min = np.float64(metrics.get("bethe_hessian_min_eigenvalue", 0.0))
    bp_stability = np.float64(metrics.get("bp_stability_score", 0.0))
    estimated_threshold = np.float64(metrics.get("estimated_threshold", 0.0))

    if trapping_density >= np.float64(0.35):
        label = "trapping_dominated_phase"
    elif spectral_radius >= np.float64(1.25) or bethe_min <= np.float64(-0.20):
        label = "spectral_instability_phase"
    elif bp_stability >= np.float64(0.70) and estimated_threshold >= np.float64(0.45):
        label = "stable_bp_phase"
    else:
        label = "fragile_bp_phase"

    return {"phase_label": str(label)}


def build_phase_profile(
    phase_id: int,
    metrics: dict[str, Any],
    label: dict[str, Any] | str,
) -> dict[str, Any]:
    """Build deterministic, stable-order phase profile record."""
    phase_label = label.get("phase_label", "") if isinstance(label, dict) else label
    return {
        "phase_id": int(phase_id),
        "phase_label": str(phase_label),
        "bp_stability_score": float(np.float64(metrics.get("bp_stability_score", 0.0))),
        "trapping_density": float(np.float64(metrics.get("trapping_density", 0.0))),
        "estimated_threshold": float(np.float64(metrics.get("estimated_threshold", 0.0))),
        "spectral_radius": float(np.float64(metrics.get("spectral_radius", 0.0))),
        "bethe_min_eigenvalue": float(np.float64(metrics.get("bethe_hessian_min_eigenvalue", 0.0))),
    }
