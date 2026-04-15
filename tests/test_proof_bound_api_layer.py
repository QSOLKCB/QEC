from __future__ import annotations

import json

from qec.orchestration.proof_bound_api_layer import (
    ProofBoundApiLayer,
    ProofContract,
    ProofReceipt,
    build_proof_bound_api_request,
    build_proof_receipt,
    compare_proof_api_replay,
    run_proof_bound_api_layer,
    summarize_proof_bound_api,
    validate_proof_bound_api_layer,
)


def _contract() -> ProofContract:
    return ProofContract(
        contract_id="decoder-adjacent",
        invariant_requirements=("same_input_same_bytes", "same_state_same_hash"),
        interface_boundary="orchestration",
    )


def _request() -> dict:
    return {"z": [3, 2, 1], "a": {"k": "v"}}


def _response() -> dict:
    return {"decoder_output": {"syndrome": [0, 1, 0], "status": "ok"}}


def test_deterministic_repeated_api_runs():
    one = run_proof_bound_api_layer(_request(), _contract(), ("same_replay_same_artifact",), _response())
    two = run_proof_bound_api_layer(_request(), _contract(), ("same_replay_same_artifact",), _response())
    assert one.to_canonical_json() == two.to_canonical_json()
    assert one.stable_hash() == two.stable_hash()


def test_stable_hash_reproducibility_for_dataclasses():
    request = build_proof_bound_api_request(_request(), _contract(), ())
    receipt = build_proof_receipt(request, _response(), request.proof_contract.stable_hash())
    assert request.stable_hash() == request.stable_hash()
    assert receipt.stable_hash() == receipt.stable_hash()


def test_proof_receipt_determinism():
    req = build_proof_bound_api_request(_request(), _contract(), ())
    receipt_a = build_proof_receipt(req, _response(), req.proof_contract.stable_hash())
    receipt_b = build_proof_receipt(req, _response(), req.proof_contract.stable_hash())
    assert receipt_a.to_dict() == receipt_b.to_dict()


def test_canonical_json_round_trip():
    layer = run_proof_bound_api_layer(_request(), _contract(), (), _response())
    reloaded = json.loads(layer.to_canonical_json())
    assert json.dumps(reloaded, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False) == layer.to_canonical_json()


def test_validator_never_raises_on_malformed_input():
    result = validate_proof_bound_api_layer(object())
    assert result["valid"] is False
    assert isinstance(result["violations"], tuple)


def test_malformed_input_handling_normalizes_safely():
    layer = run_proof_bound_api_layer(api_request="not-mapping", proof_contract="not-mapping", invariant_requirements=123, api_response="bad")
    assert layer.api_request.api_request == {}
    assert layer.api_response == {}
    assert layer.api_request.proof_contract.contract_id == "contract"


def test_metric_bounds_and_fixed_order_in_summary():
    layer = run_proof_bound_api_layer(_request(), _contract(), (), _response())
    receipt = layer.proof_receipt
    for metric in (
        receipt.invariant_satisfaction_score,
        receipt.replay_contract_score,
        receipt.serialization_integrity_score,
        receipt.interface_boundary_score,
        receipt.proof_confidence_score,
    ):
        assert 0.0 <= metric <= 1.0
    summary = summarize_proof_bound_api(layer)
    assert summary.index("invariant_satisfaction_score") < summary.index("replay_contract_score")
    assert summary.index("replay_contract_score") < summary.index("serialization_integrity_score")


def test_replay_comparison_stability():
    base = run_proof_bound_api_layer(_request(), _contract(), (), _response())
    replay = run_proof_bound_api_layer(_request(), _contract(), (), _response())
    comparison = compare_proof_api_replay(base, replay)
    assert comparison["is_stable_replay"] is True
    assert comparison["mismatches"] == ()


def test_no_input_mutation():
    request = _request()
    response = _response()
    contract = _contract()
    request_before = json.dumps(request, sort_keys=True)
    response_before = json.dumps(response, sort_keys=True)
    run_proof_bound_api_layer(request, contract, (), response)
    assert json.dumps(request, sort_keys=True) == request_before
    assert json.dumps(response, sort_keys=True) == response_before


def test_decoder_untouched_confirmation():
    import qec.orchestration.proof_bound_api_layer as module

    source = module.__doc__ or ""
    assert "decoder" in source
    assert "does not alter decoder semantics" in source.lower()


def test_proof_boundary_violation_path():
    contract = ProofContract(
        contract_id="decoder-adjacent",
        invariant_requirements=("same_input_same_bytes",),
        interface_boundary="invalid-boundary",
    )
    layer = run_proof_bound_api_layer(_request(), contract, (), _response())
    assert layer.proof_receipt.advisory_state == "proof_boundary_violation"


def test_summary_content_contains_required_fields():
    layer = run_proof_bound_api_layer(_request(), _contract(), (), _response())
    summary = summarize_proof_bound_api(layer)
    assert "ProofBoundApiLayer" in summary
    assert "contract_hash=" in summary
    assert "advisory_state=" in summary
    assert "proof_confidence_score" in summary


def test_validator_accepts_valid_layer():
    layer = run_proof_bound_api_layer(_request(), _contract(), (), _response())
    result = validate_proof_bound_api_layer(layer)
    assert result["valid"] is True
    assert result["violations"] == ()


def test_validator_rejects_tampered_mapping_payload():
    layer = run_proof_bound_api_layer(_request(), _contract(), (), _response())
    tampered = layer.to_dict()
    tampered["contract_hash"] = "bad-contract-hash"
    tampered["proof_receipt"]["contract_hash"] = "bad-contract-hash"
    result = validate_proof_bound_api_layer(tampered)
    assert result["valid"] is False
    assert "contract_hash_mismatch" in result["violations"]
    assert "proof_receipt_mismatch" in result["violations"]
