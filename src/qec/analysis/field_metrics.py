"""Deterministic field metrics layer.

Provides physics-inspired metrics describing stability, symmetry,
nonlinearity, structural change, and oscillation/periodicity.

All functions are pure, deterministic, and never mutate inputs.
Dependencies: numpy (stdlib math used where sufficient).

Note: Core implementations live in qec.core.metrics.
This module re-exports them for public API compatibility.
"""

from qec.core.metrics import (  # noqa: F401
    PHI,
    compute_complexity,
    compute_curvature,
    compute_field_metrics,
    compute_nonlinear_response,
    compute_phi_alignment,
    compute_resonance,
    compute_symmetry_score,
    compute_triality_balance,
)
