from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

from .canonical_hashing import canonical_bytes, canonical_json, sha256_hex
from .perturbation_matrix import (
    EnergyMatrixReceipt,
    PerturbationMatrix,
    validate_energy_matrix_receipt,
    validate_perturbation_matrix,
)

_ERR_INVALID_INPUT = "INVALID_INPUT"
_ERR_INVALID_HASH_FORMAT = "INVALID_HASH_FORMAT"
_ERR_HASH_MISMATCH = "HASH_MISMATCH"
_ERR_INVALID_STRESS_MODE = "INVALID_STRESS_MODE"
_ERR_INVALID_STRESS_CLASSIFICATION = "INVALID_STRESS_CLASSIFICATION"
_ERR_INVALID_STABILITY_MODE = "INVALID_STABILITY_MODE"
_ERR_INVALID_STABILITY_CLASS = "INVALID_STABILITY_CLASS"
_ERR_STRESS_CLASSIFICATION_MISMATCH = "STRESS_CLASSIFICATION_MISMATCH"
_ERR_STABILITY_CLASS_MISMATCH = "STABILITY_CLASS_MISMATCH"
_ERR_STRESS_RECEIPT_MISMATCH = "STRESS_RECEIPT_MISMATCH"
_ERR_STABILITY_PROOF_MISMATCH = "STABILITY_PROOF_MISMATCH"
_ERR_COUNT_MISMATCH = "COUNT_MISMATCH"
_ERR_IMPACT_SCORE_MISMATCH = "IMPACT_SCORE_MISMATCH"

_MAX_OPERATION_TYPE_COUNTS = 128
_MAX_TARGET_ARTIFACT_TYPE_COUNTS = 10_000
_MAX_ABS_TOTAL_IMPACT_SCORE = 1_000_000_000_000

_STRESS_MODE = "DETERMINISTIC_SEMANTIC_STRESS"
_STABILITY_MODE = "DETERMINISTIC_PERTURBATION_STABILITY"

_ALLOWED_STRESS_CLASSIFICATIONS = {"STRESS_STABLE", "STRESS_CHANGED", "STRESS_HEAVY", "STRESS_INVALID"}
_ALLOWED_STABILITY_CLASSES = {"PERTURBATION_STABLE", "PERTURBATION_CHANGED", "PERTURBATION_HEAVY", "PERTURBATION_INVALID"}

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _validate_sha256_hex(value: object) -> None:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(_ERR_INVALID_HASH_FORMAT)


def _validate_non_bool_int(value: object, *, min_value: int | None = None, max_value: int | None = None, err: str = _ERR_INVALID_INPUT) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(err)
    if min_value is not None and value < min_value:
        raise ValueError(err)
    if max_value is not None and value > max_value:
        raise ValueError(err)
    return value


def _validate_counts(value: object, *, max_length: int) -> tuple[tuple[str, int], ...]:
    if not isinstance(value, tuple) or len(value) > max_length:
        raise ValueError(_ERR_INVALID_INPUT)
    out: list[tuple[str, int]] = []
    seen: set[str] = set()
    last: str | None = None
    for item in value:
        if not isinstance(item, tuple) or len(item) != 2:
            raise ValueError(_ERR_INVALID_INPUT)
        key, count = item
        if not isinstance(key, str) or not key:
            raise ValueError(_ERR_INVALID_INPUT)
        if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
            raise ValueError(_ERR_INVALID_INPUT)
        if key in seen or (last is not None and key < last):
            raise ValueError(_ERR_INVALID_INPUT)
        seen.add(key)
        last = key
        out.append((key, count))
    return tuple(out)





def _derive_stress_classification(total_integer_impact_score: int, changed_entry_count: int) -> str:
    if total_integer_impact_score == 0 and changed_entry_count == 0:
        return "STRESS_STABLE"
    if total_integer_impact_score > 0 and changed_entry_count > 0 and total_integer_impact_score < 100:
        return "STRESS_CHANGED"
    if total_integer_impact_score >= 100:
        return "STRESS_HEAVY"
    return "STRESS_INVALID"


def _derive_stability(stress_classification: str, total_integer_impact_score: int, changed_entry_count: int) -> tuple[bool, str]:
    if stress_classification == "STRESS_STABLE" and total_integer_impact_score == 0 and changed_entry_count == 0:
        return (True, "PERTURBATION_STABLE")
    if total_integer_impact_score > 0 and total_integer_impact_score < 100:
        return (False, "PERTURBATION_CHANGED")
    if total_integer_impact_score >= 100:
        return (False, "PERTURBATION_HEAVY")
    return (False, "PERTURBATION_INVALID")


def _semantic_stress_receipt_payload(perturbation_matrix_hash: str, energy_matrix_receipt_hash: str, stress_mode: str, changed_entry_count: int, unchanged_entry_count: int, total_integer_impact_score: int, operation_type_counts: tuple[tuple[str, int], ...], target_artifact_type_counts: tuple[tuple[str, int], ...], stress_classification: str) -> dict[str, Any]:
    return {
        "perturbation_matrix_hash": perturbation_matrix_hash,
        "energy_matrix_receipt_hash": energy_matrix_receipt_hash,
        "stress_mode": stress_mode,
        "changed_entry_count": changed_entry_count,
        "unchanged_entry_count": unchanged_entry_count,
        "total_integer_impact_score": total_integer_impact_score,
        "operation_type_counts": [[k, v] for k, v in operation_type_counts],
        "target_artifact_type_counts": [[k, v] for k, v in target_artifact_type_counts],
        "stress_classification": stress_classification,
    }


def _perturbation_stability_proof_payload(perturbation_matrix_hash: str, energy_matrix_receipt_hash: str, semantic_stress_receipt_hash: str, stability_mode: str, stability_class: str, replay_stable: bool, changed_entry_count: int, unchanged_entry_count: int, total_integer_impact_score: int) -> dict[str, Any]:
    return {
        "perturbation_matrix_hash": perturbation_matrix_hash,
        "energy_matrix_receipt_hash": energy_matrix_receipt_hash,
        "semantic_stress_receipt_hash": semantic_stress_receipt_hash,
        "stability_mode": stability_mode,
        "stability_class": stability_class,
        "replay_stable": replay_stable,
        "changed_entry_count": changed_entry_count,
        "unchanged_entry_count": unchanged_entry_count,
        "total_integer_impact_score": total_integer_impact_score,
    }


@dataclass(frozen=True)
class SemanticStressReceipt:
    perturbation_matrix_hash: str
    energy_matrix_receipt_hash: str
    stress_mode: str
    changed_entry_count: int
    unchanged_entry_count: int
    total_integer_impact_score: int
    operation_type_counts: tuple[tuple[str, int], ...]
    target_artifact_type_counts: tuple[tuple[str, int], ...]
    stress_classification: str
    semantic_stress_receipt_hash: str

    def __post_init__(self) -> None:
        validate_semantic_stress_receipt(self)

    def _hash_payload(self) -> dict[str, Any]:
        return _semantic_stress_receipt_payload(
            self.perturbation_matrix_hash,
            self.energy_matrix_receipt_hash,
            self.stress_mode,
            self.changed_entry_count,
            self.unchanged_entry_count,
            self.total_integer_impact_score,
            self.operation_type_counts,
            self.target_artifact_type_counts,
            self.stress_classification,
        )

    def to_dict(self) -> dict[str, Any]:
        return {**self._hash_payload(), "semantic_stress_receipt_hash": self.semantic_stress_receipt_hash}

    def to_canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return canonical_bytes(self.to_dict())


@dataclass(frozen=True)
class PerturbationStabilityProof:
    perturbation_matrix_hash: str
    energy_matrix_receipt_hash: str
    semantic_stress_receipt_hash: str
    stability_mode: str
    stability_class: str
    replay_stable: bool
    changed_entry_count: int
    unchanged_entry_count: int
    total_integer_impact_score: int
    perturbation_stability_proof_hash: str

    def __post_init__(self) -> None:
        validate_perturbation_stability_proof(self)

    def _hash_payload(self) -> dict[str, Any]:
        return _perturbation_stability_proof_payload(
            self.perturbation_matrix_hash,
            self.energy_matrix_receipt_hash,
            self.semantic_stress_receipt_hash,
            self.stability_mode,
            self.stability_class,
            self.replay_stable,
            self.changed_entry_count,
            self.unchanged_entry_count,
            self.total_integer_impact_score,
        )

    def to_dict(self) -> dict[str, Any]:
        return {**self._hash_payload(), "perturbation_stability_proof_hash": self.perturbation_stability_proof_hash}

    def to_canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return canonical_bytes(self.to_dict())


def build_semantic_stress_receipt(perturbation_matrix: PerturbationMatrix, energy_matrix_receipt: EnergyMatrixReceipt) -> SemanticStressReceipt:
    validate_perturbation_matrix(perturbation_matrix)
    validate_energy_matrix_receipt(energy_matrix_receipt)
    if perturbation_matrix.perturbation_matrix_hash != energy_matrix_receipt.perturbation_matrix_hash:
        raise ValueError(_ERR_STRESS_RECEIPT_MISMATCH)
    if perturbation_matrix.changed_entry_count != energy_matrix_receipt.changed_entry_count or perturbation_matrix.unchanged_entry_count != energy_matrix_receipt.unchanged_entry_count:
        raise ValueError(_ERR_COUNT_MISMATCH)
    if sum(e.integer_impact_score for e in perturbation_matrix.entries) != energy_matrix_receipt.total_integer_impact_score:
        raise ValueError(_ERR_IMPACT_SCORE_MISMATCH)
    stress_classification = _derive_stress_classification(energy_matrix_receipt.total_integer_impact_score, energy_matrix_receipt.changed_entry_count)
    payload = _semantic_stress_receipt_payload(
        perturbation_matrix.perturbation_matrix_hash,
        energy_matrix_receipt.energy_matrix_receipt_hash,
        _STRESS_MODE,
        energy_matrix_receipt.changed_entry_count,
        energy_matrix_receipt.unchanged_entry_count,
        energy_matrix_receipt.total_integer_impact_score,
        energy_matrix_receipt.operation_type_counts,
        energy_matrix_receipt.target_artifact_type_counts,
        stress_classification,
    )
    return SemanticStressReceipt(
        perturbation_matrix_hash=perturbation_matrix.perturbation_matrix_hash,
        energy_matrix_receipt_hash=energy_matrix_receipt.energy_matrix_receipt_hash,
        stress_mode=_STRESS_MODE,
        changed_entry_count=energy_matrix_receipt.changed_entry_count,
        unchanged_entry_count=energy_matrix_receipt.unchanged_entry_count,
        total_integer_impact_score=energy_matrix_receipt.total_integer_impact_score,
        operation_type_counts=energy_matrix_receipt.operation_type_counts,
        target_artifact_type_counts=energy_matrix_receipt.target_artifact_type_counts,
        stress_classification=stress_classification,
        semantic_stress_receipt_hash=sha256_hex(payload),
    )


def build_perturbation_stability_proof(perturbation_matrix: PerturbationMatrix, energy_matrix_receipt: EnergyMatrixReceipt, semantic_stress_receipt: SemanticStressReceipt) -> PerturbationStabilityProof:
    validate_perturbation_matrix(perturbation_matrix)
    validate_energy_matrix_receipt(energy_matrix_receipt)
    validate_semantic_stress_receipt(semantic_stress_receipt)
    if perturbation_matrix.perturbation_matrix_hash != energy_matrix_receipt.perturbation_matrix_hash:
        raise ValueError(_ERR_STABILITY_PROOF_MISMATCH)
    if semantic_stress_receipt.perturbation_matrix_hash != perturbation_matrix.perturbation_matrix_hash:
        raise ValueError(_ERR_STABILITY_PROOF_MISMATCH)
    if semantic_stress_receipt.energy_matrix_receipt_hash != energy_matrix_receipt.energy_matrix_receipt_hash:
        raise ValueError(_ERR_STABILITY_PROOF_MISMATCH)
    rebuilt_stress = build_semantic_stress_receipt(perturbation_matrix, energy_matrix_receipt)
    if rebuilt_stress.to_dict() != semantic_stress_receipt.to_dict():
        raise ValueError(_ERR_STRESS_RECEIPT_MISMATCH)
    replay_stable, stability_class = _derive_stability(
        semantic_stress_receipt.stress_classification,
        semantic_stress_receipt.total_integer_impact_score,
        semantic_stress_receipt.changed_entry_count,
    )
    payload = _perturbation_stability_proof_payload(
        perturbation_matrix.perturbation_matrix_hash,
        energy_matrix_receipt.energy_matrix_receipt_hash,
        semantic_stress_receipt.semantic_stress_receipt_hash,
        _STABILITY_MODE,
        stability_class,
        replay_stable,
        semantic_stress_receipt.changed_entry_count,
        semantic_stress_receipt.unchanged_entry_count,
        semantic_stress_receipt.total_integer_impact_score,
    )
    return PerturbationStabilityProof(**payload, perturbation_stability_proof_hash=sha256_hex(payload))


def validate_semantic_stress_receipt(receipt: SemanticStressReceipt) -> bool:
    if not isinstance(receipt, SemanticStressReceipt):
        raise ValueError(_ERR_INVALID_INPUT)
    _validate_sha256_hex(receipt.perturbation_matrix_hash)
    _validate_sha256_hex(receipt.energy_matrix_receipt_hash)
    if receipt.stress_mode != _STRESS_MODE:
        raise ValueError(_ERR_INVALID_STRESS_MODE)
    _validate_non_bool_int(receipt.changed_entry_count, min_value=0, err=_ERR_INVALID_INPUT)
    _validate_non_bool_int(receipt.unchanged_entry_count, min_value=0, err=_ERR_INVALID_INPUT)
    _validate_non_bool_int(receipt.total_integer_impact_score, min_value=-_MAX_ABS_TOTAL_IMPACT_SCORE, max_value=_MAX_ABS_TOTAL_IMPACT_SCORE, err=_ERR_INVALID_INPUT)
    _validate_counts(receipt.operation_type_counts, max_length=_MAX_OPERATION_TYPE_COUNTS)
    _validate_counts(receipt.target_artifact_type_counts, max_length=_MAX_TARGET_ARTIFACT_TYPE_COUNTS)
    if receipt.stress_classification not in _ALLOWED_STRESS_CLASSIFICATIONS:
        raise ValueError(_ERR_INVALID_STRESS_CLASSIFICATION)
    expected_class = _derive_stress_classification(receipt.total_integer_impact_score, receipt.changed_entry_count)
    if receipt.stress_classification != expected_class:
        raise ValueError(_ERR_STRESS_CLASSIFICATION_MISMATCH)
    _validate_sha256_hex(receipt.semantic_stress_receipt_hash)
    if sha256_hex(receipt._hash_payload()) != receipt.semantic_stress_receipt_hash:
        raise ValueError(_ERR_HASH_MISMATCH)
    return True


def validate_perturbation_stability_proof(proof: PerturbationStabilityProof) -> bool:
    if not isinstance(proof, PerturbationStabilityProof):
        raise ValueError(_ERR_INVALID_INPUT)
    _validate_sha256_hex(proof.perturbation_matrix_hash)
    _validate_sha256_hex(proof.energy_matrix_receipt_hash)
    _validate_sha256_hex(proof.semantic_stress_receipt_hash)
    if proof.stability_mode != _STABILITY_MODE:
        raise ValueError(_ERR_INVALID_STABILITY_MODE)
    if proof.stability_class not in _ALLOWED_STABILITY_CLASSES:
        raise ValueError(_ERR_INVALID_STABILITY_CLASS)
    if not isinstance(proof.replay_stable, bool):
        raise ValueError(_ERR_INVALID_INPUT)
    _validate_non_bool_int(proof.changed_entry_count, min_value=0, err=_ERR_INVALID_INPUT)
    _validate_non_bool_int(proof.unchanged_entry_count, min_value=0, err=_ERR_INVALID_INPUT)
    _validate_non_bool_int(proof.total_integer_impact_score, min_value=-_MAX_ABS_TOTAL_IMPACT_SCORE, max_value=_MAX_ABS_TOTAL_IMPACT_SCORE, err=_ERR_INVALID_INPUT)
    expected_stress_class = _derive_stress_classification(proof.total_integer_impact_score, proof.changed_entry_count)
    expected_replay_stable, expected_class = _derive_stability(expected_stress_class, proof.total_integer_impact_score, proof.changed_entry_count)
    if proof.replay_stable != expected_replay_stable or proof.stability_class != expected_class:
        raise ValueError(_ERR_STABILITY_CLASS_MISMATCH)
    _validate_sha256_hex(proof.perturbation_stability_proof_hash)
    if sha256_hex(proof._hash_payload()) != proof.perturbation_stability_proof_hash:
        raise ValueError(_ERR_HASH_MISMATCH)
    return True


def validate_semantic_stress_receipt_with_matrix(receipt: SemanticStressReceipt, perturbation_matrix: PerturbationMatrix, energy_matrix_receipt: EnergyMatrixReceipt) -> bool:
    validate_semantic_stress_receipt(receipt)
    validate_perturbation_matrix(perturbation_matrix)
    validate_energy_matrix_receipt(energy_matrix_receipt)
    rebuilt = build_semantic_stress_receipt(perturbation_matrix, energy_matrix_receipt)
    if rebuilt.to_dict() != receipt.to_dict():
        raise ValueError(_ERR_STRESS_RECEIPT_MISMATCH)
    return True


def validate_perturbation_stability_proof_with_receipts(proof: PerturbationStabilityProof, perturbation_matrix: PerturbationMatrix, energy_matrix_receipt: EnergyMatrixReceipt, semantic_stress_receipt: SemanticStressReceipt) -> bool:
    validate_perturbation_stability_proof(proof)
    validate_perturbation_matrix(perturbation_matrix)
    validate_energy_matrix_receipt(energy_matrix_receipt)
    validate_semantic_stress_receipt(semantic_stress_receipt)
    rebuilt = build_perturbation_stability_proof(perturbation_matrix, energy_matrix_receipt, semantic_stress_receipt)
    if rebuilt.to_dict() != proof.to_dict():
        raise ValueError(_ERR_STABILITY_PROOF_MISMATCH)
    return True
