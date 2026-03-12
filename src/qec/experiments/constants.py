"""Shared constants for experiment strategies and metric names."""

from __future__ import annotations


MUTATION_STRATEGIES = [
    "baseline",
    "random_swap",
    "nb_swap",
    "nb_ipr_swap",
    "nb_gradient",
]

ABLATION_METRICS = [
    "fer",
    "spectral_radius",
    "ipr",
    "runtime",
]

