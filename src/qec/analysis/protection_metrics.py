"""v108.2.0 — Deterministic chain protection-metrics abstraction layer."""

from __future__ import annotations

from typing import Any

from qec.analysis.minimal_chain_experiments import DIFFUSION_STEPS, run_minimal_chain_experiment

ROBUSTNESS_WEIGHT_LOCAL_IMMUNITY = 0.4
ROBUSTNESS_WEIGHT_ENDPOINT_RETENTION = 0.3
ROBUSTNESS_WEIGHT_PARITY_LOCK = 0.3

PROTECTION_CLASS_MODERATE_THRESHOLD = 0.34
PROTECTION_CLASS_STRONG_THRESHOLD = 0.67


def run_protection_metrics(
    chain_length: int,
    perturbation_magnitude: float = 1.0,
    diffusion_steps: int = DIFFUSION_STEPS,
) -> dict[str, Any]:
    """Run deterministic boundary-vs-center protection metrics."""
    if chain_length < 3:
        raise ValueError("chain_length must be >= 3")
    boundary_result = run_minimal_chain_experiment(
        chain_length=chain_length,
        perturbation_index=0,
        perturbation_magnitude=perturbation_magnitude,
        diffusion_steps=diffusion_steps,
    )
    center_result = run_minimal_chain_experiment(
        chain_length=chain_length,
        perturbation_index=chain_length // 2,
        perturbation_magnitude=perturbation_magnitude,
        diffusion_steps=diffusion_steps,
    )

    local_immunity_score = _clamp01(1.0 - float(center_result["protection_hint_score"]))
    endpoint_retention_advantage = _clamp01(
        float(boundary_result["endpoint_signal_strength"]) - float(center_result["endpoint_signal_strength"])
    )
    parity_lock_advantage = _clamp01(
        float(boundary_result["coherence_response"]["parity_stability_score"])
        - float(center_result["coherence_response"]["parity_stability_score"])
    )
    robustness_score = _clamp01(
        ROBUSTNESS_WEIGHT_LOCAL_IMMUNITY * local_immunity_score
        + ROBUSTNESS_WEIGHT_ENDPOINT_RETENTION * endpoint_retention_advantage
        + ROBUSTNESS_WEIGHT_PARITY_LOCK * parity_lock_advantage
    )

    return {
        "chain_length": int(chain_length),
        "boundary_result": boundary_result,
        "center_result": center_result,
        "local_immunity_score": round(local_immunity_score, 12),
        "endpoint_retention_advantage": round(endpoint_retention_advantage, 12),
        "parity_lock_advantage": round(parity_lock_advantage, 12),
        "robustness_score": round(robustness_score, 12),
        "protection_class": _protection_class(robustness_score),
    }


def _protection_class(robustness_score: float) -> str:
    if robustness_score >= PROTECTION_CLASS_STRONG_THRESHOLD:
        return "strong"
    if robustness_score >= PROTECTION_CLASS_MODERATE_THRESHOLD:
        return "moderate"
    return "weak"


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


__all__ = ["run_protection_metrics"]
