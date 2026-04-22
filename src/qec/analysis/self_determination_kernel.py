"""v143.3 — Self-Determination Kernel (SPHAERA Phase 4).

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
from qec.analysis.ensemble_consistency_engine import EnsembleConsistencyReceipt
from qec.analysis.generalized_invariant_detector import InvariantDetectionReceipt
from qec.analysis.invariant_geometry_embedding import InvariantGeometryReceipt
from qec.analysis.spectral_structure_kernel import SpectralStructureReceipt

SELF_DETERMINATION_KERNEL_VERSION = "v143.3"
_CONTROL_MODE = "self_determination_advisory"
_ROUND_DIGITS = 12
_CONSISTENCY_THRESHOLD = 0.6

_W_CONSISTENCY = 0.5
_W_SPECTRAL = 0.3
_W_INVARIANT = 0.2


def _round_stable(value: float) -> float:
    return round(float(value), _ROUND_DIGITS)


def _bounded01(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    output = float(value)
    if not math.isfinite(output):
        raise ValueError(f"{name} must be finite")
    if output < 0.0 or output > 1.0:
        raise ValueError(f"{name} must be in [0,1]")
    return output


def _validate_hash(value: str, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
        raise ValueError(f"{name} must be 64-char lowercase sha256 hex")
    return value


def _clamp01(value: float) -> float:
    if value <= 0.0:
        return 0.0
    if value >= 1.0:
        return 1.0
    return float(value)


def _posture_label(
    spectral_dispersion_score: float,
    coupling_density_score: float,
    diagonal_dominance_score: float,
    global_consistency_score: float,
) -> str:
    if diagonal_dominance_score >= 0.9 and coupling_density_score < 0.2 and global_consistency_score >= 0.9:
        return "rigid_stable"
    if diagonal_dominance_score >= 0.6 and spectral_dispersion_score < 0.45 and global_consistency_score >= 0.75:
        return "structured_stable"
    if coupling_density_score < 0.7 and global_consistency_score >= 0.5:
        return "coupled_adaptive"
    return "highly_coupled_dynamic"


@dataclass(frozen=True)
class TransitionOption:
    transition_id: str
    description: str
    priority_score: float
    admissible: bool

    def __post_init__(self) -> None:
        if not isinstance(self.transition_id, str) or not self.transition_id:
            raise ValueError("transition_id must be non-empty str")
        if not isinstance(self.description, str) or not self.description:
            raise ValueError("description must be non-empty str")
        object.__setattr__(self, "priority_score", _bounded01(self.priority_score, "priority_score"))
        if not isinstance(self.admissible, bool):
            raise ValueError("admissible must be bool")

    def to_dict(self) -> dict[str, Any]:
        return {
            "transition_id": self.transition_id,
            "description": self.description,
            "priority_score": self.priority_score,
            "admissible": self.admissible,
        }

    def to_canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    def stable_hash(self) -> str:
        return sha256_hex(self.to_dict())


@dataclass(frozen=True)
class SelfDeterminationReceipt:
    allowed_transitions: tuple[TransitionOption, ...]
    selected_transition_id: str
    selected_transition_score: float
    selection_confidence: float
    posture_label: str
    transition_count: int
    admissible_count: int
    spectral_receipt_stable_hash: str
    ensemble_receipt_stable_hash: str
    geometry_receipt_stable_hash: str
    invariant_receipt_stable_hash: str | None
    version: str
    control_mode: str
    observatory_only: bool
    stable_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.allowed_transitions, tuple):
            raise ValueError("allowed_transitions must be tuple[TransitionOption, ...]")
        for transition in self.allowed_transitions:
            if not isinstance(transition, TransitionOption):
                raise ValueError("allowed_transitions must be tuple[TransitionOption, ...]")
        transition_ids = tuple(transition.transition_id for transition in self.allowed_transitions)
        if tuple(sorted(transition_ids)) != transition_ids:
            raise ValueError("allowed_transitions must be sorted by transition_id")
        if len(set(transition_ids)) != len(transition_ids):
            raise ValueError("transition_id values must be unique")

        if not isinstance(self.selected_transition_id, str):
            raise ValueError("selected_transition_id must be str")

        object.__setattr__(self, "selected_transition_score", _bounded01(self.selected_transition_score, "selected_transition_score"))
        object.__setattr__(self, "selection_confidence", _bounded01(self.selection_confidence, "selection_confidence"))

        if not isinstance(self.posture_label, str) or not self.posture_label:
            raise ValueError("posture_label must be non-empty str")
        if isinstance(self.transition_count, bool) or not isinstance(self.transition_count, int):
            raise ValueError("transition_count must be int")
        if isinstance(self.admissible_count, bool) or not isinstance(self.admissible_count, int):
            raise ValueError("admissible_count must be int")
        if self.transition_count != len(self.allowed_transitions):
            raise ValueError("transition_count mismatch")
        if self.admissible_count < 0 or self.admissible_count > self.transition_count:
            raise ValueError("admissible_count out of range")

        _validate_hash(self.spectral_receipt_stable_hash, "spectral_receipt_stable_hash")
        _validate_hash(self.ensemble_receipt_stable_hash, "ensemble_receipt_stable_hash")
        _validate_hash(self.geometry_receipt_stable_hash, "geometry_receipt_stable_hash")
        if self.invariant_receipt_stable_hash is not None:
            _validate_hash(self.invariant_receipt_stable_hash, "invariant_receipt_stable_hash")

        if not isinstance(self.version, str) or not self.version:
            raise ValueError("version must be non-empty str")
        if self.control_mode != _CONTROL_MODE:
            raise ValueError(f"control_mode must be {_CONTROL_MODE!r}")
        if self.observatory_only is not True:
            raise ValueError("observatory_only must be True")

        object.__setattr__(self, "stable_hash", sha256_hex(self._payload_without_hash()))

    def _payload_without_hash(self) -> dict[str, Any]:
        return {
            "allowed_transitions": tuple(transition.to_dict() for transition in self.allowed_transitions),
            "selected_transition_id": self.selected_transition_id,
            "selected_transition_score": self.selected_transition_score,
            "selection_confidence": self.selection_confidence,
            "posture_label": self.posture_label,
            "transition_count": self.transition_count,
            "admissible_count": self.admissible_count,
            "spectral_receipt_stable_hash": self.spectral_receipt_stable_hash,
            "ensemble_receipt_stable_hash": self.ensemble_receipt_stable_hash,
            "geometry_receipt_stable_hash": self.geometry_receipt_stable_hash,
            "invariant_receipt_stable_hash": self.invariant_receipt_stable_hash,
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


def _build_transition_space(posture_label: str) -> tuple[tuple[str, str], ...]:
    if posture_label == "rigid_stable":
        return (
            ("maintain_invariant_anchor", "Maintain high-invariance anchor posture."),
            ("micro_adjust_consistency", "Apply micro-adjustments to preserve consistency structure."),
            ("structured_coupling_probe", "Probe structured coupling while preserving invariant geometry."),
        )
    if posture_label == "structured_stable":
        return (
            ("maintain_structured_baseline", "Maintain structured baseline under spectral constraints."),
            ("reduce_dispersion", "Reduce spectral dispersion while preserving admissible geometry."),
            ("controlled_coupling_increase", "Increase coupling in a controlled, invariant-safe manner."),
        )
    if posture_label == "coupled_adaptive":
        return (
            ("adaptive_balance", "Balance coupling and consistency under invariant constraints."),
            ("stabilize_diagonal_dominance", "Increase diagonal dominance without violating invariants."),
            ("spectral_recompression", "Recompress spectral spread while keeping ensemble agreement."),
        )
    return (
        ("reduce_coupling_pressure", "Reduce coupling pressure to recover deterministic structure."),
        ("consistency_repair", "Repair ensemble consistency while preserving invariant equivalence."),
        ("safe_hold", "Hold current posture when no stronger transition is admissible."),
    )


def _construct_allowed_transitions(
    *,
    posture_label: str,
    spectral: SpectralStructureReceipt,
    ensemble: EnsembleConsistencyReceipt,
    geometry: InvariantGeometryReceipt,
) -> tuple[TransitionOption, ...]:
    invariance_strength = _round_stable((geometry.geometric_consistency_score + geometry.embedding_stability_score) * 0.5)
    spectral_stability = _round_stable((spectral.diagonal_dominance_score + (1.0 - spectral.coupling_density_score)) * 0.5)

    output: list[TransitionOption] = []
    for transition_id, description in _build_transition_space(posture_label):
        priority = _round_stable(
            0.4 * ensemble.global_consistency_score
            + 0.35 * spectral_stability
            + 0.25 * invariance_strength
        )

        admissible = True

        if ensemble.global_consistency_score < 0.2 and invariance_strength < 0.25:
            admissible = False

        if invariance_strength < 0.3 and transition_id not in {"safe_hold", "maintain_invariant_anchor", "consistency_repair"}:
            admissible = False

        if ensemble.global_consistency_score < _CONSISTENCY_THRESHOLD and transition_id in {
            "controlled_coupling_increase",
            "adaptive_balance",
            "structured_coupling_probe",
        }:
            admissible = False

        if spectral.diagonal_dominance_score < 0.2 and transition_id in {
            "controlled_coupling_increase",
            "structured_coupling_probe",
        }:
            admissible = False

        if spectral.coupling_density_score > 0.85 and transition_id in {
            "controlled_coupling_increase",
            "adaptive_balance",
        }:
            admissible = False

        output.append(
            TransitionOption(
                transition_id=transition_id,
                description=description,
                priority_score=priority,
                admissible=admissible,
            )
        )

    return tuple(sorted(output, key=lambda item: item.transition_id))


def _selection_score(
    *,
    transition: TransitionOption,
    spectral: SpectralStructureReceipt,
    ensemble: EnsembleConsistencyReceipt,
    geometry: InvariantGeometryReceipt,
) -> float:
    consistency_alignment = ensemble.global_consistency_score
    spectral_alignment = _round_stable((1.0 - spectral.spectral_dispersion_score + spectral.diagonal_dominance_score) * 0.5)
    invariant_preservation = _round_stable((geometry.geometric_consistency_score + geometry.embedding_stability_score) * 0.5)

    score = (
        _W_CONSISTENCY * consistency_alignment
        + _W_SPECTRAL * spectral_alignment
        + _W_INVARIANT * invariant_preservation
    )
    score = _round_stable(_clamp01(score) * transition.priority_score)
    return _round_stable(_clamp01(score))


def evaluate_self_determination_kernel(
    spectral_receipt: SpectralStructureReceipt,
    ensemble_receipt: EnsembleConsistencyReceipt,
    geometry_receipt: InvariantGeometryReceipt,
    invariant_receipt: InvariantDetectionReceipt | None = None,
    *,
    version: str = SELF_DETERMINATION_KERNEL_VERSION,
) -> SelfDeterminationReceipt:
    if not isinstance(spectral_receipt, SpectralStructureReceipt):
        raise ValueError("invalid input type")
    if not isinstance(ensemble_receipt, EnsembleConsistencyReceipt):
        raise ValueError("invalid input type")
    if not isinstance(geometry_receipt, InvariantGeometryReceipt):
        raise ValueError("invalid input type")
    if invariant_receipt is not None and not isinstance(invariant_receipt, InvariantDetectionReceipt):
        raise ValueError("invalid input type")
    if not isinstance(version, str) or not version:
        raise ValueError("version must be non-empty str")

    if spectral_receipt.ensemble_receipt_stable_hash != ensemble_receipt.stable_hash:
        raise ValueError("spectral_receipt and ensemble_receipt stable_hash mismatch")
    if spectral_receipt.geometry_receipt_stable_hash != geometry_receipt.stable_hash:
        raise ValueError("spectral_receipt and geometry_receipt stable_hash mismatch")
    if spectral_receipt.invariant_receipt_stable_hash != ensemble_receipt.invariant_receipt_stable_hash:
        raise ValueError("spectral and ensemble invariant_receipt_stable_hash mismatch")

    invariant_hash: str | None = None
    if invariant_receipt is not None:
        if invariant_receipt.stable_hash != spectral_receipt.invariant_receipt_stable_hash:
            raise ValueError("invariant receipt stable_hash mismatch")
        invariant_hash = invariant_receipt.stable_hash

    posture_label = _posture_label(
        spectral_dispersion_score=spectral_receipt.spectral_dispersion_score,
        coupling_density_score=spectral_receipt.coupling_density_score,
        diagonal_dominance_score=spectral_receipt.diagonal_dominance_score,
        global_consistency_score=ensemble_receipt.global_consistency_score,
    )

    transitions = _construct_allowed_transitions(
        posture_label=posture_label,
        spectral=spectral_receipt,
        ensemble=ensemble_receipt,
        geometry=geometry_receipt,
    )

    scored_transitions: list[tuple[float, TransitionOption]] = []
    for transition in transitions:
        score = _selection_score(
            transition=transition,
            spectral=spectral_receipt,
            ensemble=ensemble_receipt,
            geometry=geometry_receipt,
        )
        scored_transitions.append((score, transition))

    scored_transitions_sorted = sorted(
        scored_transitions,
        key=lambda item: (-item[0], item[1].transition_id),
    )

    selected_transition_id = "no_admissible_transition"
    selected_transition_score = 0.0
    for score, transition in scored_transitions_sorted:
        if transition.admissible:
            selected_transition_id = transition.transition_id
            selected_transition_score = _round_stable(score)
            break

    max_score = max((score for score, _ in scored_transitions_sorted), default=0.0)
    min_score = min((score for score, _ in scored_transitions_sorted), default=0.0)
    if selected_transition_id == "no_admissible_transition":
        selection_confidence = 0.0
    elif max_score <= min_score:
        selection_confidence = 1.0
    else:
        selection_confidence = _round_stable((selected_transition_score - min_score) / (max_score - min_score))
    selection_confidence = _round_stable(_clamp01(selection_confidence))

    admissible_count = sum(1 for transition in transitions if transition.admissible)

    return SelfDeterminationReceipt(
        allowed_transitions=transitions,
        selected_transition_id=selected_transition_id,
        selected_transition_score=_round_stable(_clamp01(selected_transition_score)),
        selection_confidence=selection_confidence,
        posture_label=posture_label,
        transition_count=len(transitions),
        admissible_count=admissible_count,
        spectral_receipt_stable_hash=spectral_receipt.stable_hash,
        ensemble_receipt_stable_hash=ensemble_receipt.stable_hash,
        geometry_receipt_stable_hash=geometry_receipt.stable_hash,
        invariant_receipt_stable_hash=invariant_hash,
        version=version,
        control_mode=_CONTROL_MODE,
        observatory_only=True,
    )


__all__ = [
    "SELF_DETERMINATION_KERNEL_VERSION",
    "TransitionOption",
    "SelfDeterminationReceipt",
    "evaluate_self_determination_kernel",
]
