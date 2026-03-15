"""Deterministic spectral signature value object."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SpectralSignature:
    """Compact spectral signature used by memory utilities."""

    negative_modes: int
    max_ipr: float
    transport_imbalance: float
