from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import pytest

from qec.analysis.loop_termination_contract import (
    LoopTerminationContract,
    build_loop_termination_contract,
    get_allowed_loop_modes,
    get_allowed_loop_termination_policies,
    validate_loop_termination_contract,
    validate_loop_termination_contract_matches_parameters,
)


def _hash() -> str:
    return "a" * 64


def _base(**kwargs: object) -> LoopTerminationContract:
    params = kwargs.pop("termination_parameters", {})
    return build_loop_termination_contract(
        source_artifact_type=kwargs.pop("source_artifact_type", "SubstrateStateReceipt"),
        source_artifact_hash=kwargs.pop("source_artifact_hash", _hash()),
        loop_label=kwargs.pop("loop_label", "PROOF_LOOP"),
        max_depth=kwargs.pop("max_depth", 8),
        input_receipt_hash_field=kwargs.pop("input_receipt_hash_field", "input_hash"),
        output_receipt_hash_field=kwargs.pop("output_receipt_hash_field", "output_hash"),
        termination_policy=kwargs.pop("termination_policy", "MAX_DEPTH_ONLY"),
        termination_parameters=params,
        **kwargs,
    )


def test_basics_and_hash_stability():
    c1 = _base()
    c2 = _base()
    assert c1.loop_termination_contract_hash == c2.loop_termination_contract_hash
    assert get_allowed_loop_modes() == frozenset({"BOUNDED_RECEIPT_HASH_LOOP"})
    for policy, params in [
        ("MAX_DEPTH_ONLY", {}),
        ("FIXED_POINT_HASH", {"stable_hash_field": "state_hash"}),
        ("TARGET_HASH_REACHED", {"target_hash": _hash()}),
        ("STATUS_FIELD_MATCH", {"status_field": "status", "terminal_statuses": ["DONE"]}),
        ("BOUNDED_DIVERGENCE_COUNT", {"max_divergence_count": 2}),
    ]:
        assert _base(termination_policy=policy, termination_parameters=params)

    with pytest.raises(ValueError, match="INVALID_HASH_FORMAT"):
        _base(source_artifact_hash="abc")
    with pytest.raises(ValueError, match="INVALID_HASH_FORMAT"):
        _base(source_artifact_hash="A" * 64)
    with pytest.raises(ValueError, match="INVALID_SOURCE_ARTIFACT_TYPE"):
        _base(source_artifact_type="")
    with pytest.raises(ValueError, match="INVALID_LOOP_LABEL"):
        _base(loop_label="bad")
    with pytest.raises(ValueError, match="LOOP_DEPTH_OUT_OF_BOUNDS"):
        _base(max_depth=True)
    with pytest.raises(ValueError, match="LOOP_DEPTH_OUT_OF_BOUNDS"):
        _base(max_depth=0)
    with pytest.raises(ValueError, match="LOOP_DEPTH_OUT_OF_BOUNDS"):
        _base(max_depth=10001)
    with pytest.raises(ValueError, match="INVALID_RECEIPT_HASH_FIELD"):
        _base(input_receipt_hash_field="9bad")
    with pytest.raises(ValueError, match="INVALID_RECEIPT_HASH_FIELD"):
        _base(output_receipt_hash_field="9bad")
    with pytest.raises(ValueError, match="RECEIPT_HASH_FIELD_MISMATCH"):
        _base(input_receipt_hash_field="same", output_receipt_hash_field="same")
    with pytest.raises(ValueError, match="INVALID_TERMINATION_POLICY"):
        _base(termination_policy="NOPE")

    with pytest.raises(FrozenInstanceError):
        c1.max_depth = 9  # type: ignore[misc]
    assert c1.to_canonical_json() == c2.to_canonical_json()
    assert c1.to_canonical_bytes() == c2.to_canonical_bytes()


def test_hash_field_mismatch_and_mode_validation():
    c = _base()
    with pytest.raises(ValueError, match="INVALID_HASH_FORMAT"):
        validate_loop_termination_contract(replace(c, termination_parameters_hash="BAD"))
    with pytest.raises(ValueError, match="HASH_MISMATCH"):
        validate_loop_termination_contract(replace(c, termination_parameters_hash="b" * 64))
    with pytest.raises(ValueError, match="INVALID_HASH_FORMAT"):
        validate_loop_termination_contract(replace(c, loop_termination_contract_hash="BAD"))
    with pytest.raises(ValueError, match="HASH_MISMATCH"):
        validate_loop_termination_contract(replace(c, loop_termination_contract_hash="b" * 64))
    with pytest.raises(ValueError, match="INVALID_LOOP_MODE"):
        validate_loop_termination_contract(replace(c, loop_mode="OTHER"))


def test_policy_semantics_and_json_safety():
    assert _base(termination_parameters={})
    with pytest.raises(ValueError, match="INVALID_TERMINATION_PARAMETERS"):
        _base(termination_parameters={"x": 1})
    assert _base(termination_policy="FIXED_POINT_HASH", termination_parameters={"stable_hash_field": "f"})
    with pytest.raises(ValueError, match="INVALID_TERMINATION_PARAMETERS"):
        _base(termination_policy="FIXED_POINT_HASH", termination_parameters={})
    assert _base(termination_policy="TARGET_HASH_REACHED", termination_parameters={"target_hash": _hash()})
    for bad in ["A" * 64, "a" * 63, "z" * 64]:
        with pytest.raises(ValueError, match="INVALID_TERMINATION_PARAMETERS"):
            _base(termination_policy="TARGET_HASH_REACHED", termination_parameters={"target_hash": bad})
    assert _base(termination_policy="STATUS_FIELD_MATCH", termination_parameters={"status_field": "status", "terminal_statuses": ["DONE", "OK"]})
    bad_status = [
        {"terminal_statuses": ["DONE"]},
        {"status_field": "status", "terminal_statuses": []},
        {"status_field": "status", "terminal_statuses": ["DONE", "DONE"]},
        {"status_field": "status", "terminal_statuses": ["Z", "A"]},
        {"status_field": "status", "terminal_statuses": ["bad"]},
    ]
    for params in bad_status:
        with pytest.raises(ValueError, match="INVALID_TERMINATION_PARAMETERS"):
            _base(termination_policy="STATUS_FIELD_MATCH", termination_parameters=params)
    assert _base(termination_policy="BOUNDED_DIVERGENCE_COUNT", termination_parameters={"max_divergence_count": 0})
    for bad in [True, -1, 10001]:
        with pytest.raises(ValueError, match="INVALID_TERMINATION_PARAMETERS"):
            _base(termination_policy="BOUNDED_DIVERGENCE_COUNT", termination_parameters={"max_divergence_count": bad})
    for bad_params in [("x",), {"v": 1.0}, {"v": b"x"}, {"v": {1}}, {"v": object()}]:
        with pytest.raises(ValueError, match="INVALID_TERMINATION_PARAMETERS"):
            _base(termination_parameters=bad_params)


def test_canonical_parameter_validation_and_complete_validator():
    c = _base(termination_policy="STATUS_FIELD_MATCH", termination_parameters={"status_field": "status", "terminal_statuses": ["DONE", "OK"]})
    with pytest.raises(ValueError, match="INVALID_TERMINATION_PARAMETERS"):
        replace(c, canonical_termination_parameters='{"terminal_statuses":["DONE","OK"],"status_field":"status"}')
    with pytest.raises(ValueError, match="INVALID_TERMINATION_PARAMETERS"):
        replace(c, canonical_termination_parameters="{")
    with pytest.raises(ValueError, match="TERMINATION_PARAMETER_TOO_LARGE"):
        replace(c, canonical_termination_parameters="x" * 9000)
    with pytest.raises(ValueError, match="INVALID_TERMINATION_PARAMETERS"):
        _base(termination_parameters={"x": float("nan")})

    assert validate_loop_termination_contract_matches_parameters(c, {"status_field": "status", "terminal_statuses": ["DONE", "OK"]})
    with pytest.raises(ValueError, match="HASH_MISMATCH"):
        validate_loop_termination_contract_matches_parameters(c, {"status_field": "status", "terminal_statuses": ["DONE"]})

    with pytest.raises(ValueError, match="INVALID_TERMINATION_PARAMETERS"):
        replace(c, loop_termination_contract_hash=_base(termination_policy="STATUS_FIELD_MATCH", termination_parameters={"status_field": "status", "terminal_statuses": ["DONE", "OK"]}).loop_termination_contract_hash, canonical_termination_parameters='{"status_field":"status","terminal_statuses":["OK","DONE"]}')


def test_boundary_and_scope_scan():
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        validate_loop_termination_contract(object())
    with pytest.raises(TypeError):
        LoopTerminationContract(**{"bad": 1})  # type: ignore[arg-type]

    blocked = [
        "LoopIterationRecord", "RecursiveProofReceipt", "LoopTerminationProof", "OuroboricConvergenceReceipt",
        "RouterLoopReceipt", "ReadoutLoopReceipt", "MarkovLoopStabilityReceipt", "RealityLoopProofReceipt",
        "GlobalTruthReceipt", "recursion", "recurse", "recursive_execution", "infinite_loop", "while True",
        "gameplay", "render", "step_world", "execute_action", "run_game", "importlib", "__import__(",
        "subprocess", "exec(", "eval(", "random", "time.time", "datetime.now", "probability", "probabilistic",
        "neural", "learned_policy", "global_truth",
    ]
    text = open("src/qec/analysis/loop_termination_contract.py", encoding="utf-8").read()
    for token in blocked:
        assert token not in text

    assert get_allowed_loop_termination_policies() == frozenset({
        "MAX_DEPTH_ONLY", "FIXED_POINT_HASH", "TARGET_HASH_REACHED", "STATUS_FIELD_MATCH", "BOUNDED_DIVERGENCE_COUNT"
    })
