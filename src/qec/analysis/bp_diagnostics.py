"""v23.1.0 — Deterministic BP convergence diagnostics helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


_RESIDUAL_CONVERGENCE_EPS = 1e-8
_ROUND = 12


@dataclass(frozen=True)
class BPDiagnostics:
    iterations_to_converge: int
    converged: bool
    final_residual: float
    residual_history: list[float]
    syndrome_weight_history: list[int]


def collect_bp_diagnostics(history: dict[str, Any]) -> BPDiagnostics:
    """Extract deterministic BP diagnostics from a history mapping.

    Expected keys in ``history`` are ``residuals`` and optional
    ``syndrome_weights``.
    """
    residuals_raw = history.get("residuals", [])
    residuals = [round(float(v), _ROUND) for v in residuals_raw]

    syndrome_raw = history.get("syndrome_weights", [])
    syndrome_weights = [int(v) for v in syndrome_raw]

    final_residual = float(residuals[-1]) if residuals else 0.0
    converged = bool(final_residual < _RESIDUAL_CONVERGENCE_EPS)

    return BPDiagnostics(
        iterations_to_converge=len(residuals),
        converged=converged,
        final_residual=round(final_residual, _ROUND),
        residual_history=list(residuals),
        syndrome_weight_history=list(syndrome_weights),
    )

