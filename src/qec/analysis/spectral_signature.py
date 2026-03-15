"""Deterministic spectral signature container helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.qec.analysis.spectral_entropy import spectral_entropy


@dataclass(frozen=True)
class SpectralSignature:
    """Compact deterministic spectral signature."""

    spectral_radius: float
    mode_ipr: float
    support_fraction: float
    topk_mass_fraction: float
    spectral_entropy: float



def with_entropy(
    *,
    spectral_radius: float,
    mode_ipr: float,
    support_fraction: float,
    topk_mass_fraction: float,
    nb_eigenvalues: np.ndarray,
    precision: int = 12,
) -> SpectralSignature:
    """Build a signature with deterministic entropy rounding."""
    return SpectralSignature(
        spectral_radius=round(float(spectral_radius), precision),
        mode_ipr=round(float(mode_ipr), precision),
        support_fraction=round(float(support_fraction), precision),
        topk_mass_fraction=round(float(topk_mass_fraction), precision),
        spectral_entropy=spectral_entropy(nb_eigenvalues, precision=precision),
    )
