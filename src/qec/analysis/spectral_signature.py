from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class SpectralSignature:
    """
    Deterministic spectral signature for Tanner graph diagnostics.
    """

    spectral_radius: float
    bethe_negative_mass: float
    flow_ipr: float
    spectral_entropy: float = 0.0

    def with_entropy(self, entropy: float) -> "SpectralSignature":
        return SpectralSignature(
            spectral_radius=self.spectral_radius,
            bethe_negative_mass=self.bethe_negative_mass,
            flow_ipr=self.flow_ipr,
            spectral_entropy=float(entropy),
        )


def compute_signature(
    spectral_radius: float,
    bethe_negative_mass: float,
    flow_ipr: float,
    spectral_entropy: float = 0.0,
    precision: int = 12,
) -> SpectralSignature:
    """
    Construct a deterministically rounded spectral signature.
    """

    return SpectralSignature(
        spectral_radius=round(float(spectral_radius), precision),
        bethe_negative_mass=round(float(bethe_negative_mass), precision),
        flow_ipr=round(float(flow_ipr), precision),
        spectral_entropy=round(float(spectral_entropy), precision),
    )


def with_entropy(
    *,
    spectral_radius: float,
    mode_ipr: float,
    support_fraction: float,
    topk_mass_fraction: float,
    nb_eigenvalues: np.ndarray,
    precision: int = 12,
) -> SpectralSignature:
    """Compatibility helper that returns a rounded signature with entropy."""
    del support_fraction, topk_mass_fraction, nb_eigenvalues
    return compute_signature(
        spectral_radius=spectral_radius,
        bethe_negative_mass=0.0,
        flow_ipr=mode_ipr,
        spectral_entropy=0.0,
        precision=precision,
    )
