from qec.analysis.canonical_hashing import canonical_json
from dataclasses import FrozenInstanceError, replace

import pytest

from qec.analysis.perturbation_contract import build_perturbation_contract, execute_perturbation_contract
from qec.analysis.perturbation_matrix import build_energy_matrix_receipt, build_perturbation_matrix, build_perturbation_matrix_entry
from qec.analysis.semantic_stress_receipt import (
    build_perturbation_stability_proof,
    build_semantic_stress_receipt,
    validate_perturbation_stability_proof,
    validate_perturbation_stability_proof_with_receipts,
    validate_semantic_stress_receipt,
    validate_semantic_stress_receipt_with_matrix,
)


def _artifact(v: int) -> dict[str, object]:
    return {"state": {"value": v}}


def _bundle(changed_count: int, unchanged_count: int = 0):
    entries = []
    idx = 0
    for _ in range(changed_count):
        before = _artifact(idx)
        contract = build_perturbation_contract("Artifact", "a" * 64, ["state","value"], "REPLACE_VALUE", {"value": idx + 1})
        result = execute_perturbation_contract(contract, canonical_json(before))
        entries.append(build_perturbation_matrix_entry(idx, 0, f"ENTRY_{idx}", contract, result))
        idx += 1
    for _ in range(unchanged_count):
        before = _artifact(idx)
        contract = build_perturbation_contract("Artifact", "b" * 64, ["state","value"], "REPLACE_VALUE", {"value": idx})
        result = execute_perturbation_contract(contract, canonical_json(before))
        entries.append(build_perturbation_matrix_entry(idx, 0, f"ENTRY_{idx}", contract, result))
        idx += 1
    matrix = build_perturbation_matrix("STRESS_MATRIX", entries)
    energy = build_energy_matrix_receipt(matrix)
    stress = build_semantic_stress_receipt(matrix, energy)
    proof = build_perturbation_stability_proof(matrix, energy, stress)
    return matrix, energy, stress, proof


def test_semantic_stress_receipt_and_proof_basics_and_hashes():
    matrix0, energy0, stress0, proof0 = _bundle(0, 1)
    matrix1, energy1, stress1, proof1 = _bundle(1, 0)
    matrix2, energy2, stress2, proof2 = _bundle(100, 0)

    assert build_semantic_stress_receipt(matrix1, energy1).semantic_stress_receipt_hash == stress1.semantic_stress_receipt_hash
    assert build_perturbation_stability_proof(matrix1, energy1, stress1).perturbation_stability_proof_hash == proof1.perturbation_stability_proof_hash

    assert stress0.stress_classification == "STRESS_STABLE"
    assert stress1.stress_classification == "STRESS_CHANGED"
    assert stress2.stress_classification == "STRESS_HEAVY"

    assert proof0.stability_class == "PERTURBATION_STABLE" and proof0.replay_stable is True
    assert proof1.stability_class == "PERTURBATION_CHANGED" and proof1.replay_stable is False
    assert proof2.stability_class == "PERTURBATION_HEAVY" and proof2.replay_stable is False

    assert stress1.to_canonical_json() == build_semantic_stress_receipt(matrix1, energy1).to_canonical_json()
    assert proof1.to_canonical_bytes() == build_perturbation_stability_proof(matrix1, energy1, stress1).to_canonical_bytes()

    with pytest.raises(FrozenInstanceError):
        stress1.changed_entry_count = 3
    with pytest.raises(FrozenInstanceError):
        proof1.replay_stable = True


def test_semantic_stress_failures_and_complete_validation():
    matrix, energy, stress, _ = _bundle(2, 1)

    with pytest.raises(ValueError, match="INVALID_HASH_FORMAT"):
        validate_semantic_stress_receipt(replace(stress, semantic_stress_receipt_hash="bad"))
    with pytest.raises(ValueError, match="HASH_MISMATCH"):
        validate_semantic_stress_receipt(replace(stress, semantic_stress_receipt_hash="0" * 64))
    with pytest.raises(ValueError, match="INVALID_STRESS_MODE"):
        validate_semantic_stress_receipt(replace(stress, stress_mode="X"))
    with pytest.raises(ValueError, match="INVALID_STRESS_CLASSIFICATION"):
        validate_semantic_stress_receipt(replace(stress, stress_classification="X"))
    with pytest.raises(ValueError, match="COUNT_MISMATCH|INVALID_INPUT|STRESS_CLASSIFICATION_MISMATCH"):
        validate_semantic_stress_receipt_with_matrix(replace(stress, changed_entry_count=0), matrix, energy)
    with pytest.raises(ValueError, match="OPERATION_COUNTS_MISMATCH|STRESS_RECEIPT_MISMATCH|HASH_MISMATCH"):
        validate_semantic_stress_receipt_with_matrix(replace(stress, operation_type_counts=(("REPLACE_VALUE", 1),)), matrix, energy)
    with pytest.raises(ValueError, match="TARGET_COUNTS_MISMATCH|STRESS_RECEIPT_MISMATCH|HASH_MISMATCH"):
        validate_semantic_stress_receipt_with_matrix(replace(stress, target_artifact_type_counts=(("Artifact", 1),)), matrix, energy)
    with pytest.raises(ValueError, match="IMPACT_SCORE_MISMATCH|STRESS_CLASSIFICATION_MISMATCH"):
        validate_semantic_stress_receipt_with_matrix(replace(stress, total_integer_impact_score=0), matrix, energy)
    with pytest.raises(ValueError, match="STRESS_RECEIPT_MISMATCH|HASH_MISMATCH"):
        validate_semantic_stress_receipt_with_matrix(replace(stress, perturbation_matrix_hash="1" * 64), matrix, energy)
    with pytest.raises(ValueError, match="STRESS_RECEIPT_MISMATCH|HASH_MISMATCH"):
        validate_semantic_stress_receipt_with_matrix(replace(stress, energy_matrix_receipt_hash="2" * 64), matrix, energy)

    assert validate_semantic_stress_receipt_with_matrix(stress, matrix, energy) is True
    wrong_matrix, wrong_energy, _, _ = _bundle(1, 0)
    with pytest.raises(ValueError, match="STRESS_RECEIPT_MISMATCH"):
        validate_semantic_stress_receipt_with_matrix(stress, wrong_matrix, wrong_energy)


def test_proof_failures_complete_validation_and_scope_scan():
    matrix, energy, stress, proof = _bundle(2, 1)
    assert validate_perturbation_stability_proof_with_receipts(proof, matrix, energy, stress) is True

    with pytest.raises(ValueError, match="INVALID_HASH_FORMAT"):
        validate_perturbation_stability_proof(replace(proof, perturbation_stability_proof_hash="bad"))
    with pytest.raises(ValueError, match="HASH_MISMATCH"):
        validate_perturbation_stability_proof(replace(proof, perturbation_stability_proof_hash="f" * 64))
    with pytest.raises(ValueError, match="INVALID_STABILITY_MODE"):
        validate_perturbation_stability_proof(replace(proof, stability_mode="X"))
    with pytest.raises(ValueError, match="INVALID_STABILITY_CLASS"):
        validate_perturbation_stability_proof(replace(proof, stability_class="X"))
    with pytest.raises(ValueError, match="STABILITY_CLASS_MISMATCH"):
        validate_perturbation_stability_proof(replace(proof, stability_class="PERTURBATION_STABLE"))
    with pytest.raises(ValueError, match="STABILITY_PROOF_MISMATCH|HASH_MISMATCH"):
        validate_perturbation_stability_proof_with_receipts(replace(proof, semantic_stress_receipt_hash="3" * 64), matrix, energy, stress)
    with pytest.raises(ValueError, match="STABILITY_PROOF_MISMATCH|HASH_MISMATCH"):
        validate_perturbation_stability_proof_with_receipts(replace(proof, perturbation_matrix_hash="4" * 64), matrix, energy, stress)
    with pytest.raises(ValueError, match="STABILITY_PROOF_MISMATCH|HASH_MISMATCH"):
        validate_perturbation_stability_proof_with_receipts(replace(proof, energy_matrix_receipt_hash="5" * 64), matrix, energy, stress)

    wrong_matrix, wrong_energy, wrong_stress, _ = _bundle(1, 0)
    with pytest.raises(ValueError, match="STABILITY_PROOF_MISMATCH"):
        validate_perturbation_stability_proof_with_receipts(proof, wrong_matrix, wrong_energy, stress)
    with pytest.raises(ValueError, match="STABILITY_PROOF_MISMATCH"):
        validate_perturbation_stability_proof_with_receipts(proof, matrix, energy, wrong_stress)

    with pytest.raises(ValueError, match="HASH_MISMATCH|STRESS_CLASSIFICATION_MISMATCH"):
        replace(build_semantic_stress_receipt(matrix, energy), total_integer_impact_score=build_semantic_stress_receipt(matrix, energy).total_integer_impact_score + 1)

    with pytest.raises(ValueError, match="STABILITY_CLASS_MISMATCH|HASH_MISMATCH"):
        replace(proof, replay_stable=not proof.replay_stable)

    banned = [
        "gameplay", "render", "step_world", "execute_action", "run_game", "importlib", "__import__(", "subprocess",
        "exec(", "eval(", "random", "time.time", "datetime.now", "probability", "probabilistic", "neural", "learned_policy",
        "physical_energy", "joule", "voltage", "current", "substrate", "recursive", "global_truth",
    ]
    with open("src/qec/analysis/semantic_stress_receipt.py", "r", encoding="utf-8") as f:
        src = f.read().lower()
    for token in banned:
        assert token not in src


def test_stable_proof_with_flipped_replay_stable_or_stability_class():
    """Regression test: ensure proofs with changed_entry_count==0 and total_integer_impact_score==0
    but flipped replay_stable or stability_class are rejected by validate_perturbation_stability_proof."""
    matrix, energy, stress, proof = _bundle(0, 1)
    assert proof.changed_entry_count == 0
    assert proof.total_integer_impact_score == 0
    assert proof.replay_stable is True
    assert proof.stability_class == "PERTURBATION_STABLE"

    with pytest.raises(ValueError, match="STABILITY_CLASS_MISMATCH|HASH_MISMATCH"):
        validate_perturbation_stability_proof(replace(proof, replay_stable=False))

    with pytest.raises(ValueError, match="STABILITY_CLASS_MISMATCH|HASH_MISMATCH"):
        validate_perturbation_stability_proof(replace(proof, stability_class="PERTURBATION_INVALID"))

    with pytest.raises(ValueError, match="STABILITY_CLASS_MISMATCH|HASH_MISMATCH"):
        validate_perturbation_stability_proof(replace(proof, stability_class="PERTURBATION_CHANGED"))

    with pytest.raises(ValueError, match="STABILITY_CLASS_MISMATCH|HASH_MISMATCH"):
        validate_perturbation_stability_proof(replace(proof, stability_class="PERTURBATION_HEAVY"))
