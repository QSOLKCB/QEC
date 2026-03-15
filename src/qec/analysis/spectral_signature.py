"""Deterministic spectral signatures for Tanner-graph diversity tracking."""

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
import scipy.sparse

from src.qec.analysis.nonbacktracking_flow import NonBacktrackingFlowAnalyzer
from src.qec.diagnostics.non_backtracking_spectrum import compute_non_backtracking_spectrum

_ROUND = 12


@dataclass(frozen=True, eq=False)
class SpectralSignature:
    """Compact deterministic signature of a Tanner graph state."""

    nb_spectrum: np.ndarray
    bh_negative_modes: int
    bh_energy: float
    max_ipr: float


    @property
    def nb_spectrum_magnitudes(self) -> np.ndarray:
        return self.nb_spectrum

    @property
    def bh_negative_mass(self) -> float:
        return self.bh_energy

    @property
    def flow_ipr(self) -> float:
        return self.max_ipr

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SpectralSignature):
            return False
        return (
            np.array_equal(self.nb_spectrum, other.nb_spectrum)
            and self.bh_negative_modes == other.bh_negative_modes
            and float(self.bh_energy) == float(other.bh_energy)
            and float(self.max_ipr) == float(other.max_ipr)
        )


def _round64(value: float, precision: int = _ROUND) -> float:
    return float(np.round(np.float64(value), int(precision)))


def _to_dense64(H: np.ndarray | scipy.sparse.spmatrix) -> np.ndarray:
    if scipy.sparse.issparse(H):
        return np.asarray(H.toarray(), dtype=np.float64)
    return np.asarray(H, dtype=np.float64)


def _tanner_adjacency(H: np.ndarray) -> np.ndarray:
    m, n = H.shape
    top = np.concatenate([np.zeros((n, n), dtype=np.float64), H.T], axis=1)
    bottom = np.concatenate([H, np.zeros((m, m), dtype=np.float64)], axis=1)
    return np.concatenate([top, bottom], axis=0)


def compute_signature(H: np.ndarray | scipy.sparse.spmatrix, *, num_nb_values: int = 6, precision: int = _ROUND) -> SpectralSignature:
    """Compute deterministic spectral signature for a parity-check matrix."""
    H_arr = _to_dense64(H)

    nb = compute_non_backtracking_spectrum(H_arr)
    eigvals = np.asarray(nb.get("nb_eigenvalues", []), dtype=np.float64)
    if eigvals.size == 0:
        mags = np.zeros((0,), dtype=np.float64)
    else:
        complex_vals = eigvals[:, 0] + 1j * eigvals[:, 1]
        mags = np.abs(complex_vals).astype(np.float64, copy=False)
    k = max(1, int(num_nb_values))
    if mags.size >= k:
        nb_spectrum = mags[:k]
    else:
        nb_spectrum = np.pad(mags, (0, k - mags.size), mode="constant")
    nb_spectrum = np.asarray(np.round(nb_spectrum, int(precision)), dtype=np.float64)

    A = _tanner_adjacency(H_arr)
    if A.size == 0:
        bh_negative_modes = 0
        bh_energy = 0.0
    else:
        eigvals_A = np.linalg.eigvalsh(A)
        max_eigval_A = float(np.max(np.abs(eigvals_A))) if eigvals_A.size else 0.0
        r_used = float(np.sqrt(max_eigval_A)) if max_eigval_A > 0.0 else 1.0
        D = np.diag(np.sum(A, axis=1, dtype=np.float64)).astype(np.float64, copy=False)
        H_B = (r_used * r_used - 1.0) * np.eye(A.shape[0], dtype=np.float64) - r_used * A + D
        bh_eigvals = np.linalg.eigvalsh(H_B)
        neg = bh_eigvals[bh_eigvals < 0.0]
        bh_negative_modes = int(neg.size)
        bh_energy = _round64(float(np.sum(np.abs(neg), dtype=np.float64)), precision)

    flow = NonBacktrackingFlowAnalyzer().compute_flow(H_arr)
    max_ipr = _round64(float(flow.get("flow_localization", 0.0)), precision)

    return SpectralSignature(
        nb_spectrum=nb_spectrum,
        bh_negative_modes=int(bh_negative_modes),
        bh_energy=float(bh_energy),
        max_ipr=float(max_ipr),
    )
