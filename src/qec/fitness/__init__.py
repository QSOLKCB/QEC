"""
v10.0.0 — Fitness Engine.

Computes composite fitness scores for LDPC/QLDPC parity-check matrices
using spectral metrics, girth analysis, ACE spectrum, and expansion
properties.

Layer 3 — Fitness.
Does not import or modify the decoder (Layer 1).
Fully deterministic: no randomness, no global state, no input mutation.
"""

from qec.fitness.spectral_metrics import (
    compute_nbt_spectral_radius,
    compute_girth_spectrum,
    compute_ace_spectrum,
    estimate_eigenvector_ipr,
)
from qec.fitness.fitness_engine import FitnessEngine

__all__ = [
    "compute_nbt_spectral_radius",
    "compute_girth_spectrum",
    "compute_ace_spectrum",
    "estimate_eigenvector_ipr",
    "FitnessEngine",
]
