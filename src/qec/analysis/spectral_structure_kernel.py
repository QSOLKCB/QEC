"""v143.2 — Spectral / Random Matrix Structure Kernel (SPHAERA Phase 3).

Attribution:
This module incorporates concepts from:
Marc Brendecke (2026)
Quantum Sphaera Companion v3.30.0
DOI: https://doi.org/10.5281/zenodo.19682951
License: CC-BY-4.0
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any

import numpy as np

from qec.analysis.canonical_hashing import canonical_json, sha256_hex
from qec.analysis.ensemble_consistency_engine import EnsembleClass, EnsembleConsistencyReceipt
from qec.analysis.generalized_invariant_detector import InvariantDetectionReceipt
from qec.analysis.invariant_geometry_embedding import InvariantGeometryReceipt

SPECTRAL_STRUCTURE_KERNEL_VERSION = "v143.2"
_CONTROL_MODE = "spectral_structure_advisory"
_ROUND_DIGITS = 12

_DYNAMICS_RANK: dict[str, int] = {
    "rigid": 0,
    "structured": 1,
    "coupled": 2,
    "highly_coupled": 3,
}


def _round_stable(value: float) -> float:
    return round(float(value), _ROUND_DIGITS)


def _finite_float(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    output = float(value)
    if not math.isfinite(output):
        raise ValueError(f"{name} must be finite")
    return output


def _bounded01(value: float, name: str) -> float:
    output = _finite_float(value, name)
    if output < 0.0 or output > 1.0:
        raise ValueError(f"{name} must be in [0,1]")
    return output


def _validate_hash(value: str, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
        raise ValueError(f"{name} must be 64-char lowercase sha256 hex")
    return value


def _ordered_ensembles(ensembles: tuple[EnsembleClass, ...]) -> tuple[EnsembleClass, ...]:
    return tuple(sorted(ensembles, key=lambda item: item.class_id))


def _centered_vector(centroid_vector: tuple[float, ...]) -> tuple[float, ...]:
    return tuple(_round_stable(float(value) - 0.5) for value in centroid_vector)


def _cosine_similarity(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    if len(left) != len(right):
        raise ValueError("embedding dimension mismatch")
    dot = 0.0
    left_norm_sq = 0.0
    right_norm_sq = 0.0
    for left_value, right_value in zip(left, right):
        dot += left_value * right_value
        left_norm_sq += left_value * left_value
        right_norm_sq += right_value * right_value
    if left_norm_sq == 0.0 or right_norm_sq == 0.0:
        return 0.0
    similarity = dot / math.sqrt(left_norm_sq * right_norm_sq)
    return _round_stable(max(-1.0, min(1.0, similarity)))


def _build_ensemble_operator(ensembles: tuple[EnsembleClass, ...]) -> np.ndarray:
    count = len(ensembles)
    matrix = np.zeros((count, count), dtype=np.float64)

    centered = tuple(_centered_vector(ensemble.centroid_vector) for ensemble in ensembles)
    diagonal_values = tuple(_round_stable(max(0.0, min(1.0, 1.0 - float(ensemble.mean_deviation)))) for ensemble in ensembles)

    for row in range(count):
        matrix[row, row] = diagonal_values[row]
        for col in range(row + 1, count):
            value = _cosine_similarity(centered[row], centered[col])
            matrix[row, col] = value
            matrix[col, row] = value

    return matrix


def _classify_dynamics(
    spectral_dispersion_score: float,
    coupling_density_score: float,
    diagonal_dominance_score: float,
) -> str:
    if diagonal_dominance_score >= 0.9 and coupling_density_score < 0.2:
        return "rigid"
    if spectral_dispersion_score < 0.4:
        return "structured"
    if coupling_density_score < 0.7:
        return "coupled"
    return "highly_coupled"


@dataclass(frozen=True)
class SpectralStructureReceipt:
    ensemble_operator_shape: tuple[int, int]
    trace_value: float
    frobenius_norm: float
    diagonal_energy: float
    off_diagonal_energy: float
    spectral_radius_proxy: float
    spectral_gap_proxy: float
    spectral_dispersion_score: float
    coupling_density_score: float
    diagonal_dominance_score: float
    ensemble_symmetry_score: float
    dynamics_label: str
    dynamics_rank: int
    geometry_receipt_stable_hash: str
    ensemble_receipt_stable_hash: str
    version: str
    control_mode: str
    observatory_only: bool
    stable_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.ensemble_operator_shape, tuple) or len(self.ensemble_operator_shape) != 2:
            raise ValueError("ensemble_operator_shape must be tuple[int, int]")
        rows, cols = self.ensemble_operator_shape
        if any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in (rows, cols)):
            raise ValueError("ensemble_operator_shape must be tuple[int, int]")

        object.__setattr__(self, "trace_value", _finite_float(self.trace_value, "trace_value"))
        object.__setattr__(self, "frobenius_norm", _finite_float(self.frobenius_norm, "frobenius_norm"))
        object.__setattr__(self, "diagonal_energy", _finite_float(self.diagonal_energy, "diagonal_energy"))
        object.__setattr__(self, "off_diagonal_energy", _finite_float(self.off_diagonal_energy, "off_diagonal_energy"))
        object.__setattr__(self, "spectral_radius_proxy", _finite_float(self.spectral_radius_proxy, "spectral_radius_proxy"))
        object.__setattr__(self, "spectral_gap_proxy", _finite_float(self.spectral_gap_proxy, "spectral_gap_proxy"))
        object.__setattr__(self, "spectral_dispersion_score", _bounded01(self.spectral_dispersion_score, "spectral_dispersion_score"))
        object.__setattr__(self, "coupling_density_score", _bounded01(self.coupling_density_score, "coupling_density_score"))
        object.__setattr__(self, "diagonal_dominance_score", _bounded01(self.diagonal_dominance_score, "diagonal_dominance_score"))
        object.__setattr__(self, "ensemble_symmetry_score", _bounded01(self.ensemble_symmetry_score, "ensemble_symmetry_score"))

        if self.dynamics_label not in _DYNAMICS_RANK:
            raise ValueError("invalid dynamics_label")
        if self.dynamics_rank != _DYNAMICS_RANK[self.dynamics_label]:
            raise ValueError("dynamics_rank mismatch")

        _validate_hash(self.geometry_receipt_stable_hash, "geometry_receipt_stable_hash")
        _validate_hash(self.ensemble_receipt_stable_hash, "ensemble_receipt_stable_hash")

        if not isinstance(self.version, str) or not self.version:
            raise ValueError("version must be non-empty str")
        if self.control_mode != _CONTROL_MODE:
            raise ValueError(f"control_mode must be {_CONTROL_MODE!r}")
        if self.observatory_only is not True:
            raise ValueError("observatory_only must be True")

        object.__setattr__(self, "stable_hash", sha256_hex(self._payload_without_hash()))

    def _payload_without_hash(self) -> dict[str, Any]:
        return {
            "ensemble_operator_shape": self.ensemble_operator_shape,
            "trace_value": self.trace_value,
            "frobenius_norm": self.frobenius_norm,
            "diagonal_energy": self.diagonal_energy,
            "off_diagonal_energy": self.off_diagonal_energy,
            "spectral_radius_proxy": self.spectral_radius_proxy,
            "spectral_gap_proxy": self.spectral_gap_proxy,
            "spectral_dispersion_score": self.spectral_dispersion_score,
            "coupling_density_score": self.coupling_density_score,
            "diagonal_dominance_score": self.diagonal_dominance_score,
            "ensemble_symmetry_score": self.ensemble_symmetry_score,
            "dynamics_label": self.dynamics_label,
            "dynamics_rank": self.dynamics_rank,
            "geometry_receipt_stable_hash": self.geometry_receipt_stable_hash,
            "ensemble_receipt_stable_hash": self.ensemble_receipt_stable_hash,
            "version": self.version,
            "control_mode": self.control_mode,
            "observatory_only": self.observatory_only,
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self._payload_without_hash()
        payload["stable_hash"] = self.stable_hash
        return payload

    def to_canonical_json(self) -> str:
        return canonical_json(self.to_dict())


def evaluate_spectral_structure_kernel(
    ensemble_receipt: EnsembleConsistencyReceipt,
    geometry_receipt: InvariantGeometryReceipt,
    invariant_receipt: InvariantDetectionReceipt | None = None,
    *,
    version: str = SPECTRAL_STRUCTURE_KERNEL_VERSION,
) -> SpectralStructureReceipt:
    if not isinstance(ensemble_receipt, EnsembleConsistencyReceipt):
        raise ValueError("invalid input type")
    if not isinstance(geometry_receipt, InvariantGeometryReceipt):
        raise ValueError("invalid input type")
    if invariant_receipt is not None and not isinstance(invariant_receipt, InvariantDetectionReceipt):
        raise ValueError("invalid input type")
    if invariant_receipt is not None:
        expected_invariant_hash = ensemble_receipt.invariant_receipt_stable_hash
        provided_invariant_hash = invariant_receipt.stable_hash
        if provided_invariant_hash != expected_invariant_hash:
            raise ValueError(
                "invariant receipt stable_hash does not match "
                "ensemble_receipt.invariant_receipt_stable_hash"
            )
    if not isinstance(version, str) or not version:
        raise ValueError("version must be non-empty str")

    ensembles = _ordered_ensembles(ensemble_receipt.ensembles)
    operator = _build_ensemble_operator(ensembles)
    count = operator.shape[0]

    diagonal = np.diag(operator)
    diagonal_energy = _round_stable(float(np.sum(np.square(diagonal), dtype=np.float64)))
    frobenius_sq = _round_stable(float(np.sum(np.square(operator), dtype=np.float64)))
    off_diagonal_energy = _round_stable(max(0.0, frobenius_sq - diagonal_energy))

    trace_value = _round_stable(float(np.trace(operator, dtype=np.float64)))
    frobenius_norm = _round_stable(math.sqrt(frobenius_sq))

    row_abs_sums = np.sum(np.abs(operator), axis=1, dtype=np.float64)
    spectral_radius_proxy = _round_stable(float(np.max(row_abs_sums)) if count > 0 else 0.0)

    eigenvalues = np.linalg.eigvalsh(operator.astype(np.float64, copy=False)) if count > 0 else np.array([], dtype=np.float64)
    eigenvalues_sorted = tuple(_round_stable(float(value)) for value in np.sort(eigenvalues)[::-1])
    if not eigenvalues_sorted:
        spectral_gap_proxy = 0.0
        eigen_std = 0.0
    elif len(eigenvalues_sorted) == 1:
        spectral_gap_proxy = abs(eigenvalues_sorted[0])
        eigen_std = 0.0
    else:
        spectral_gap_proxy = abs(eigenvalues_sorted[0] - eigenvalues_sorted[1])
        eigen_std = float(np.std(np.asarray(eigenvalues_sorted, dtype=np.float64), dtype=np.float64))
    spectral_gap_proxy = _round_stable(spectral_gap_proxy)

    total_energy = diagonal_energy + off_diagonal_energy
    if total_energy <= 0.0:
        coupling_density_score = 0.0
        diagonal_dominance_score = 1.0
    else:
        coupling_density_score = _round_stable(max(0.0, min(1.0, off_diagonal_energy / total_energy)))
        diagonal_dominance_score = _round_stable(max(0.0, min(1.0, diagonal_energy / total_energy)))

    dispersion_denominator = max(1.0, float(count))
    spectral_dispersion_score = _round_stable(max(0.0, min(1.0, eigen_std / dispersion_denominator)))

    asymmetry = float(np.max(np.abs(operator - operator.T))) if count > 0 else 0.0
    order_score = 1.0 if tuple(item.class_id for item in ensembles) == tuple(sorted(item.class_id for item in ensembles)) else 0.0
    ensemble_symmetry_score = _round_stable(max(0.0, min(1.0, (1.0 - min(1.0, asymmetry)) * order_score)))

    dynamics_label = _classify_dynamics(
        spectral_dispersion_score=spectral_dispersion_score,
        coupling_density_score=coupling_density_score,
        diagonal_dominance_score=diagonal_dominance_score,
    )

    return SpectralStructureReceipt(
        ensemble_operator_shape=(int(operator.shape[0]), int(operator.shape[1])),
        trace_value=trace_value,
        frobenius_norm=frobenius_norm,
        diagonal_energy=diagonal_energy,
        off_diagonal_energy=off_diagonal_energy,
        spectral_radius_proxy=spectral_radius_proxy,
        spectral_gap_proxy=spectral_gap_proxy,
        spectral_dispersion_score=spectral_dispersion_score,
        coupling_density_score=coupling_density_score,
        diagonal_dominance_score=diagonal_dominance_score,
        ensemble_symmetry_score=ensemble_symmetry_score,
        dynamics_label=dynamics_label,
        dynamics_rank=_DYNAMICS_RANK[dynamics_label],
        geometry_receipt_stable_hash=geometry_receipt.stable_hash,
        ensemble_receipt_stable_hash=ensemble_receipt.stable_hash,
        version=version,
        control_mode=_CONTROL_MODE,
        observatory_only=True,
    )


__all__ = [
    "SPECTRAL_STRUCTURE_KERNEL_VERSION",
    "SpectralStructureReceipt",
    "evaluate_spectral_structure_kernel",
]
