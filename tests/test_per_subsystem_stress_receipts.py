from qec.analysis.canonical_hashing import canonical_json
from dataclasses import FrozenInstanceError, replace
import pytest

from qec.analysis.perturbation_contract import build_perturbation_contract, execute_perturbation_contract
from qec.analysis.perturbation_matrix import build_energy_matrix_receipt, build_perturbation_matrix, build_perturbation_matrix_entry
from qec.analysis.semantic_stress_receipt import (
    build_perturbation_stability_proof,
    build_semantic_stress_receipt,
)
from qec.analysis.per_subsystem_stress_receipts import (
    build_layer_activation_stability_receipt,
    build_router_stress_receipt,
    validate_layer_activation_stability_receipt,
    validate_router_stress_receipt,
)


def _artifact(v: int, artifact_type: str = "LayerActivation") -> dict[str, object]:
    return {"state": {"value": v}, "type": artifact_type}


def _bundle(changed_count: int, unchanged_count: int = 0, artifact_type: str = "LayerActivation"):
    entries = []
    idx = 0
    for _ in range(changed_count):
        before = _artifact(idx, artifact_type)
        contract = build_perturbation_contract(artifact_type, "a" * 64, ["state","value"], "REPLACE_VALUE", {"value": idx + 1})
        result = execute_perturbation_contract(contract, canonical_json(before))
        entries.append(build_perturbation_matrix_entry(idx, 0, f"ENTRY_{idx}", contract, result))
        idx += 1
    for _ in range(unchanged_count):
        before = _artifact(idx, artifact_type)
        contract = build_perturbation_contract(artifact_type, "b" * 64, ["state","value"], "REPLACE_VALUE", {"value": idx})
        result = execute_perturbation_contract(contract, canonical_json(before))
        entries.append(build_perturbation_matrix_entry(idx, 0, f"ENTRY_{idx}", contract, result))
        idx += 1
    matrix = build_perturbation_matrix("SUBSYSTEM_STRESS_MATRIX", entries)
    energy = build_energy_matrix_receipt(matrix)
    stress = build_semantic_stress_receipt(matrix, energy)
    proof = build_perturbation_stability_proof(matrix, energy, stress)
    return matrix, energy, stress, proof


def test_layer_activation_stability_receipt_basics():
    """Test basic construction and hash stability for LayerActivationStabilityReceipt"""
    matrix, energy, stress, proof = _bundle(1, 0, "LayerActivation")
    
    receipt = build_layer_activation_stability_receipt(matrix, energy, stress, proof)
    
    assert receipt.subsystem_type == "LAYER"
    assert receipt.subsystem_label == "LAYER_ACTIVATION"
    assert receipt.subsystem_entry_count == 1
    assert receipt.changed_entry_count == 1
    assert receipt.unchanged_entry_count == 0
    assert receipt.subsystem_stress_class == "SUBSYSTEM_STRESS_CHANGED"
    assert receipt.subsystem_stability_class == "SUBSYSTEM_STABILITY_CHANGED"
    assert receipt.replay_stable is False
    
    # Hash stability
    receipt2 = build_layer_activation_stability_receipt(matrix, energy, stress, proof)
    assert receipt.layer_activation_stability_receipt_hash == receipt2.layer_activation_stability_receipt_hash
    
    # Validation
    assert validate_layer_activation_stability_receipt(receipt) is True


def test_router_stress_receipt_basics():
    """Test basic construction and hash stability for RouterStressReceipt"""
    matrix, energy, stress, proof = _bundle(1, 0, "RouterConfig")
    
    receipt = build_router_stress_receipt(matrix, energy, stress, proof)
    
    assert receipt.subsystem_type == "ROUTER"
    assert receipt.subsystem_label == "ROUTER"
    assert receipt.subsystem_entry_count == 1
    assert receipt.changed_entry_count == 1
    assert receipt.router_stress_receipt_hash is not None
    
    # Hash stability
    receipt2 = build_router_stress_receipt(matrix, energy, stress, proof)
    assert receipt.router_stress_receipt_hash == receipt2.router_stress_receipt_hash
    
    # Validation
    assert validate_router_stress_receipt(receipt) is True


def test_subsystem_stress_classifications():
    """Test stress classification logic for different scenarios"""
    # Stable subsystem (unchanged)
    matrix1, energy1, stress1, proof1 = _bundle(0, 1, "LayerActivation")
    receipt1 = build_layer_activation_stability_receipt(matrix1, energy1, stress1, proof1)
    assert receipt1.subsystem_stress_class == "SUBSYSTEM_STRESS_STABLE"
    assert receipt1.subsystem_stability_class == "SUBSYSTEM_STABILITY_STABLE"
    assert receipt1.replay_stable is True
    
    # Changed subsystem (low impact)
    matrix2, energy2, stress2, proof2 = _bundle(1, 0, "LayerActivation")
    receipt2 = build_layer_activation_stability_receipt(matrix2, energy2, stress2, proof2)
    assert receipt2.subsystem_stress_class == "SUBSYSTEM_STRESS_CHANGED"
    assert receipt2.subsystem_stability_class == "SUBSYSTEM_STABILITY_CHANGED"
    assert receipt2.replay_stable is False
    
    # Heavy subsystem (high impact)
    matrix3, energy3, stress3, proof3 = _bundle(100, 0, "LayerActivation")
    receipt3 = build_layer_activation_stability_receipt(matrix3, energy3, stress3, proof3)
    assert receipt3.subsystem_stress_class == "SUBSYSTEM_STRESS_HEAVY"
    assert receipt3.subsystem_stability_class == "SUBSYSTEM_STABILITY_HEAVY"
    assert receipt3.replay_stable is False


def test_receipt_validation_failures():
    """Test validation failures for invalid receipts"""
    matrix, energy, stress, proof = _bundle(1, 0, "LayerActivation")
    receipt = build_layer_activation_stability_receipt(matrix, energy, stress, proof)
    
    # Bad hash format
    with pytest.raises(ValueError, match="INVALID_HASH_FORMAT"):
        validate_layer_activation_stability_receipt(replace(receipt, layer_activation_stability_receipt_hash="bad"))
    
    # Hash mismatch
    with pytest.raises(ValueError, match="HASH_MISMATCH"):
        validate_layer_activation_stability_receipt(replace(receipt, layer_activation_stability_receipt_hash="0" * 64))
    
    # Invalid subsystem type
    with pytest.raises(ValueError, match="INVALID_SUBSYSTEM_TYPE"):
        validate_layer_activation_stability_receipt(replace(receipt, subsystem_type="INVALID"))
    
    # Count mismatch
    with pytest.raises(ValueError, match="SUBSYSTEM_COUNT_MISMATCH"):
        validate_layer_activation_stability_receipt(replace(receipt, subsystem_entry_count=999))


def test_receipt_immutability():
    """Test that receipts are frozen and immutable"""
    matrix, energy, stress, proof = _bundle(1, 0, "LayerActivation")
    receipt = build_layer_activation_stability_receipt(matrix, energy, stress, proof)
    
    with pytest.raises(FrozenInstanceError):
        receipt.subsystem_entry_count = 999


def test_canonical_serialization():
    """Test canonical JSON and bytes serialization"""
    matrix, energy, stress, proof = _bundle(1, 0, "LayerActivation")
    receipt = build_layer_activation_stability_receipt(matrix, energy, stress, proof)
    
    # to_dict should produce consistent output
    dict1 = receipt.to_dict()
    dict2 = receipt.to_dict()
    assert dict1 == dict2
    
    # Canonical JSON should be deterministic
    json1 = receipt.to_canonical_json()
    json2 = receipt.to_canonical_json()
    assert json1 == json2
    
    # Canonical bytes should be deterministic
    bytes1 = receipt.to_canonical_bytes()
    bytes2 = receipt.to_canonical_bytes()
    assert bytes1 == bytes2
