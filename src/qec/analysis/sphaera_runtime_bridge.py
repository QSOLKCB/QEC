"""v143.4 — Sphaera Integration Bridge (SPHAERA Final Phase).

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

from qec.analysis.canonical_hashing import canonical_json, sha256_hex
from qec.analysis.convergence_engine import CONVERGENCE_ENGINE_VERSION, evaluate_convergence_engine
from qec.analysis.ensemble_consistency_engine import ENSEMBLE_CONSISTENCY_ENGINE_VERSION, EnsembleConsistencyReceipt, evaluate_ensemble_consistency_engine
from qec.analysis.generalized_invariant_detector import GENERALIZED_INVARIANT_DETECTOR_VERSION, InvariantDetectionReceipt, evaluate_generalized_invariant_detector
from qec.analysis.invariant_geometry_embedding import INVARIANT_GEOMETRY_EMBEDDING_VERSION, InvariantGeometryReceipt, evaluate_invariant_geometry_embedding
from qec.analysis.iterative_system_abstraction_layer import IterativeExecutionReceipt
from qec.analysis.self_determination_kernel import SELF_DETERMINATION_KERNEL_VERSION, SelfDeterminationReceipt, evaluate_self_determination_kernel
from qec.analysis.spectral_structure_kernel import SPECTRAL_STRUCTURE_KERNEL_VERSION, SpectralStructureReceipt, evaluate_spectral_structure_kernel

SPHAERA_RUNTIME_BRIDGE_VERSION = "v143.4"
_CONTROL_MODE = "sphaera_runtime_bridge_advisory"


def _bounded01(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"{name} must be finite")
    if out < 0.0 or out > 1.0:
        raise ValueError(f"{name} must be in [0,1]")
    return out


def _clamp01(value: float) -> float:
    if value <= 0.0:
        return 0.0
    if value >= 1.0:
        return 1.0
    return float(value)


def _round_stable(value: float) -> float:
    return round(float(value), 12)


def _derive_global_state_label(*, dynamics_label: str, selected_transition_id: str) -> str:
    if selected_transition_id == "no_admissible_transition":
        return "terminal_state"
    if dynamics_label == "rigid":
        return "stable_equilibrium"
    if dynamics_label == "structured":
        return "structured_equilibrium"
    if dynamics_label == "coupled":
        return "adaptive_state"
    return "dynamic_state"


def _validate_lineage(
    *,
    invariant_receipt: InvariantDetectionReceipt,
    geometry_receipt: InvariantGeometryReceipt,
    ensemble_receipt: EnsembleConsistencyReceipt,
    spectral_receipt: SpectralStructureReceipt,
    self_determination_receipt: SelfDeterminationReceipt,
) -> None:
    if ensemble_receipt.invariant_receipt_stable_hash != invariant_receipt.stable_hash:
        raise ValueError("lineage mismatch: ensemble -> invariant")
    if spectral_receipt.geometry_receipt_stable_hash != geometry_receipt.stable_hash:
        raise ValueError("lineage mismatch: spectral -> geometry")
    if spectral_receipt.ensemble_receipt_stable_hash != ensemble_receipt.stable_hash:
        raise ValueError("lineage mismatch: spectral -> ensemble")
    if spectral_receipt.invariant_receipt_stable_hash != invariant_receipt.stable_hash:
        raise ValueError("lineage mismatch: spectral -> invariant")
    if self_determination_receipt.spectral_receipt_stable_hash != spectral_receipt.stable_hash:
        raise ValueError("lineage mismatch: self_determination -> spectral")
    if self_determination_receipt.ensemble_receipt_stable_hash != ensemble_receipt.stable_hash:
        raise ValueError("lineage mismatch: self_determination -> ensemble")
    if self_determination_receipt.geometry_receipt_stable_hash != geometry_receipt.stable_hash:
        raise ValueError("lineage mismatch: self_determination -> geometry")
    if self_determination_receipt.invariant_receipt_stable_hash != invariant_receipt.stable_hash:
        raise ValueError("lineage mismatch: self_determination -> invariant")


@dataclass(frozen=True)
class SphaeraRuntimeReceipt:
    invariant_receipt: InvariantDetectionReceipt
    geometry_receipt: InvariantGeometryReceipt
    ensemble_receipt: EnsembleConsistencyReceipt
    spectral_receipt: SpectralStructureReceipt
    self_determination_receipt: SelfDeterminationReceipt
    global_state_label: str
    coherence_score: float
    invariant_hash: str
    geometry_hash: str
    ensemble_hash: str
    spectral_hash: str
    self_determination_hash: str
    version: str
    control_mode: str
    observatory_only: bool
    stable_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.invariant_receipt, InvariantDetectionReceipt):
            raise ValueError("invariant_receipt must be InvariantDetectionReceipt")
        if not isinstance(self.geometry_receipt, InvariantGeometryReceipt):
            raise ValueError("geometry_receipt must be InvariantGeometryReceipt")
        if not isinstance(self.ensemble_receipt, EnsembleConsistencyReceipt):
            raise ValueError("ensemble_receipt must be EnsembleConsistencyReceipt")
        if not isinstance(self.spectral_receipt, SpectralStructureReceipt):
            raise ValueError("spectral_receipt must be SpectralStructureReceipt")
        if not isinstance(self.self_determination_receipt, SelfDeterminationReceipt):
            raise ValueError("self_determination_receipt must be SelfDeterminationReceipt")

        if not isinstance(self.global_state_label, str) or not self.global_state_label:
            raise ValueError("global_state_label must be non-empty str")
        object.__setattr__(self, "coherence_score", _bounded01(self.coherence_score, "coherence_score"))

        expected_hashes = {
            "invariant_hash": self.invariant_receipt.stable_hash,
            "geometry_hash": self.geometry_receipt.stable_hash,
            "ensemble_hash": self.ensemble_receipt.stable_hash,
            "spectral_hash": self.spectral_receipt.stable_hash,
            "self_determination_hash": self.self_determination_receipt.stable_hash,
        }
        for field_name, expected in expected_hashes.items():
            if getattr(self, field_name) != expected:
                raise ValueError(f"{field_name} mismatch")

        if not isinstance(self.version, str) or not self.version:
            raise ValueError("version must be non-empty str")
        if self.control_mode != _CONTROL_MODE:
            raise ValueError(f"control_mode must be {_CONTROL_MODE!r}")
        if self.observatory_only is not True:
            raise ValueError("observatory_only must be True")

        object.__setattr__(self, "stable_hash", sha256_hex(self._payload_without_hash()))

    def _payload_without_hash(self) -> dict[str, Any]:
        return {
            "invariant_receipt": self.invariant_receipt.to_dict(),
            "geometry_receipt": self.geometry_receipt.to_dict(),
            "ensemble_receipt": self.ensemble_receipt.to_dict(),
            "spectral_receipt": self.spectral_receipt.to_dict(),
            "self_determination_receipt": self.self_determination_receipt.to_dict(),
            "global_state_label": self.global_state_label,
            "coherence_score": self.coherence_score,
            "invariant_hash": self.invariant_hash,
            "geometry_hash": self.geometry_hash,
            "ensemble_hash": self.ensemble_hash,
            "spectral_hash": self.spectral_hash,
            "self_determination_hash": self.self_determination_hash,
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


def evaluate_sphaera_runtime_bridge(
    execution_state: IterativeExecutionReceipt,
    *,
    invariant_receipt: InvariantDetectionReceipt | None = None,
    geometry_receipt: InvariantGeometryReceipt | None = None,
    ensemble_receipt: EnsembleConsistencyReceipt | None = None,
    spectral_receipt: SpectralStructureReceipt | None = None,
    self_determination_receipt: SelfDeterminationReceipt | None = None,
    version: str = SPHAERA_RUNTIME_BRIDGE_VERSION,
) -> SphaeraRuntimeReceipt:
    if not isinstance(execution_state, IterativeExecutionReceipt):
        raise ValueError("invalid input type")
    if invariant_receipt is not None and not isinstance(invariant_receipt, InvariantDetectionReceipt):
        raise ValueError("invalid input type")
    if geometry_receipt is not None and not isinstance(geometry_receipt, InvariantGeometryReceipt):
        raise ValueError("invalid input type")
    if ensemble_receipt is not None and not isinstance(ensemble_receipt, EnsembleConsistencyReceipt):
        raise ValueError("invalid input type")
    if spectral_receipt is not None and not isinstance(spectral_receipt, SpectralStructureReceipt):
        raise ValueError("invalid input type")
    if self_determination_receipt is not None and not isinstance(self_determination_receipt, SelfDeterminationReceipt):
        raise ValueError("invalid input type")
    if not isinstance(version, str) or not version:
        raise ValueError("version must be non-empty str")

    if invariant_receipt is None:
        invariant_receipt = evaluate_generalized_invariant_detector(
            execution_state,
            version=GENERALIZED_INVARIANT_DETECTOR_VERSION,
        )

    if geometry_receipt is None:
        convergence_receipt = evaluate_convergence_engine(
            execution_state,
            invariant_receipt,
            version=CONVERGENCE_ENGINE_VERSION,
        )
        geometry_receipt = evaluate_invariant_geometry_embedding(
            invariant_receipt,
            convergence_receipt,
            execution_state.trace,
            version=INVARIANT_GEOMETRY_EMBEDDING_VERSION,
        )

    if ensemble_receipt is None:
        ensemble_receipt = evaluate_ensemble_consistency_engine(
            geometry_receipt,
            invariant_receipt,
            execution_trace=execution_state.trace,
            version=ENSEMBLE_CONSISTENCY_ENGINE_VERSION,
        )

    if spectral_receipt is None:
        spectral_receipt = evaluate_spectral_structure_kernel(
            ensemble_receipt,
            geometry_receipt,
            invariant_receipt,
            version=SPECTRAL_STRUCTURE_KERNEL_VERSION,
        )

    if self_determination_receipt is None:
        self_determination_receipt = evaluate_self_determination_kernel(
            spectral_receipt,
            ensemble_receipt,
            geometry_receipt,
            invariant_receipt,
            version=SELF_DETERMINATION_KERNEL_VERSION,
        )

    _validate_lineage(
        invariant_receipt=invariant_receipt,
        geometry_receipt=geometry_receipt,
        ensemble_receipt=ensemble_receipt,
        spectral_receipt=spectral_receipt,
        self_determination_receipt=self_determination_receipt,
    )

    coherence_score = _round_stable(
        _clamp01(
            0.4 * ensemble_receipt.global_consistency_score
            + 0.3 * (1.0 - spectral_receipt.spectral_dispersion_score)
            + 0.3 * self_determination_receipt.selection_confidence
        )
    )

    global_state_label = _derive_global_state_label(
        dynamics_label=spectral_receipt.dynamics_label,
        selected_transition_id=self_determination_receipt.selected_transition_id,
    )

    return SphaeraRuntimeReceipt(
        invariant_receipt=invariant_receipt,
        geometry_receipt=geometry_receipt,
        ensemble_receipt=ensemble_receipt,
        spectral_receipt=spectral_receipt,
        self_determination_receipt=self_determination_receipt,
        global_state_label=global_state_label,
        coherence_score=coherence_score,
        invariant_hash=invariant_receipt.stable_hash,
        geometry_hash=geometry_receipt.stable_hash,
        ensemble_hash=ensemble_receipt.stable_hash,
        spectral_hash=spectral_receipt.stable_hash,
        self_determination_hash=self_determination_receipt.stable_hash,
        version=version,
        control_mode=_CONTROL_MODE,
        observatory_only=True,
    )


__all__ = [
    "SPHAERA_RUNTIME_BRIDGE_VERSION",
    "SphaeraRuntimeReceipt",
    "evaluate_sphaera_runtime_bridge",
]
