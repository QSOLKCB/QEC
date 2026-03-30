"""v108.1.0 — Minimal deterministic chain perturbation experiments.

This module provides a compact, analysis-first experiment for comparing
boundary and interior additive perturbations on a deterministic 1D chain.
No randomness or in-place input mutation is used.
"""

from __future__ import annotations

from typing import Any

from qec.analysis.parity_coherence import run_parity_coherence_analysis

DIFFUSION_STEPS = 4


def run_minimal_chain_experiment(
    chain_length: int,
    perturbation_index: int,
    perturbation_magnitude: float = 1.0,
) -> dict[str, Any]:
    """Run a deterministic additive perturbation experiment on a 1D chain."""
    if chain_length < 3:
        raise ValueError("chain_length must be >= 3")
    if perturbation_index < 0 or perturbation_index >= chain_length:
        raise ValueError("perturbation_index out of range")

    initial_chain = [0.0] * chain_length
    initial_chain[perturbation_index] = float(perturbation_magnitude)

    final_chain = list(initial_chain)
    for _ in range(DIFFUSION_STEPS):
        final_chain = _diffusion_step(final_chain)

    # Boundedness in [0, 1] is an architectural invariant for analysis signals.
    endpoint_signal_strength = _clamp01(_endpoint_signal_strength(final_chain))
    interior_signal_strength = _clamp01(_interior_signal_strength(final_chain))
    signal_asymmetry = _clamp01(abs(endpoint_signal_strength - interior_signal_strength))

    coherence = run_parity_coherence_analysis(final_chain)
    parity_response = coherence["parity_stability_score"]

    protection_hint_score = _clamp01(
        endpoint_signal_strength / (endpoint_signal_strength + interior_signal_strength + 1e-12)
    )

    return {
        "chain_length": int(chain_length),
        "perturbation_index": int(perturbation_index),
        "is_boundary_perturbation": bool(perturbation_index in (0, chain_length - 1)),
        "initial_chain": initial_chain,
        "final_chain": final_chain,
        "endpoint_signal_strength": round(endpoint_signal_strength, 12),
        "interior_signal_strength": round(interior_signal_strength, 12),
        "signal_asymmetry": round(signal_asymmetry, 12),
        "coherence_response": coherence,
        "parity_response": round(float(parity_response), 12),
        "protection_hint_score": round(protection_hint_score, 12),
    }


def _diffusion_step(chain: list[float]) -> list[float]:
    n = len(chain)
    out: list[float] = []
    for i in range(n):
        total = chain[i]
        count = 1
        if i > 0:
            total += chain[i - 1]
            count += 1
        if i < n - 1:
            total += chain[i + 1]
            count += 1
        out.append(total / count)
    return out


def _endpoint_signal_strength(chain: list[float]) -> float:
    return (abs(chain[0]) + abs(chain[-1])) / 2.0


def _interior_signal_strength(chain: list[float]) -> float:
    interior = chain[1:-1]
    if not interior:
        return 0.0
    return sum(abs(v) for v in interior) / len(interior)


def _clamp01(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


__all__ = ["run_minimal_chain_experiment"]
