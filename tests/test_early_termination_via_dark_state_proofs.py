from __future__ import annotations

import inspect

import pytest

from qec.analysis import early_termination_via_dark_state_proofs as et
from qec.analysis.deterministic_gnn_decoder_kernel import build_deterministic_gnn_decoder_kernel


def _kernel_result(*, confidence_boost: float = 0.0, converged: bool | None = None, delta: float | None = None):
    config = {
        "num_rounds": 4,
        "self_weight": 0.2,
        "neighbor_weight": 0.3,
        "syndrome_weight": 0.4,
        "hardware_weight": 0.05,
        "residual_weight": 0.05,
        "damping_factor": 0.2,
        "score_round_digits": 8,
        "top_k": 3,
        "convergence_epsilon": 1e-9,
        "normalization_policy": "clamp_0_1",
    }
    nodes = [
        {"node_id": "n0", "syndrome": 1.0, "parity": 0.95, "defect": 0.0, "hardware_sideband": {"latency": 0.0}},
        {"node_id": "n1", "syndrome": 0.1, "parity": 0.1, "defect": 0.0, "hardware_sideband": {"latency": 0.0}},
        {"node_id": "n2", "syndrome": 0.05, "parity": 0.05, "defect": 0.0, "hardware_sideband": {"latency": 0.0}},
    ]
    edges = [
        {"edge_id": "e0", "source_node_id": "n0", "target_node_id": "n1", "coupling_weight": 0.4, "edge_sideband": {}},
        {"edge_id": "e1", "source_node_id": "n1", "target_node_id": "n2", "coupling_weight": 0.2, "edge_sideband": {}},
    ]
    result = build_deterministic_gnn_decoder_kernel(config=config, nodes=nodes, edges=edges)
    payload = result.to_dict()
    if confidence_boost:
        payload["proposals"][0]["confidence"] = min(1.0, float(payload["proposals"][0]["confidence"]) + confidence_boost)
    if converged is not None:
        payload["converged"] = converged
    if delta is not None:
        payload["convergence_delta"] = delta
    return payload


def _config(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "minimum_top_proposal_confidence": 0.45,
        "minimum_top_proposal_score": 0.45,
        "maximum_convergence_delta": 0.6,
        "minimum_dark_state_score": 0.6,
        "minimum_dark_state_coverage": 0.5,
        "minimum_proof_consistency_score": 0.6,
        "require_convergence": True,
        "require_nonempty_proposals": True,
        "allow_termination_without_dark_state_proof": False,
        "decision_round_digits": 8,
        "normalization_policy": "clamp_0_1",
    }
    base.update(overrides)
    return base


def _proof(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "dark_state_score": 0.95,
        "dark_state_coverage": 0.92,
        "proof_consistency_score": 0.93,
        "stability_score": 0.9,
        "coverage_score": 0.9,
    }
    payload.update(overrides)
    return payload


def test_determinism_same_input_same_bytes_hash_and_lineage() -> None:
    kernel = _kernel_result(confidence_boost=0.4, converged=True, delta=0.01)
    a = et.build_early_termination_analysis_result(config=_config(), kernel_result=kernel, dark_state_inputs=_proof())
    b = et.build_early_termination_analysis_result(config=_config(), kernel_result=kernel, dark_state_inputs=_proof())

    assert a.to_canonical_bytes() == b.to_canonical_bytes()
    assert a.stable_hash() == b.stable_hash()
    assert a.receipt.stable_hash() == b.receipt.stable_hash()
    assert a.receipt.termination_replay_identity == b.receipt.termination_replay_identity


def test_terminates_early_with_strong_proposal_convergence_and_proof() -> None:
    result = et.build_early_termination_analysis_result(
        config=_config(),
        kernel_result=_kernel_result(confidence_boost=0.4, converged=True, delta=0.01),
        dark_state_inputs=_proof(),
    )
    assert result.decision.decision_label == "terminate_early"
    assert result.decision.terminate_early is True


def test_continues_when_top_proposal_confidence_too_low() -> None:
    result = et.build_early_termination_analysis_result(
        config=_config(minimum_top_proposal_confidence=0.95),
        kernel_result=_kernel_result(confidence_boost=0.0, converged=True, delta=0.01),
        dark_state_inputs=_proof(),
    )
    assert result.decision.decision_label == "continue_iteration"
    assert result.decision.terminate_early is False


def test_continues_when_convergence_required_but_absent() -> None:
    result = et.build_early_termination_analysis_result(
        config=_config(require_convergence=True, maximum_convergence_delta=0.01),
        kernel_result=_kernel_result(confidence_boost=0.4, converged=False, delta=0.2),
        dark_state_inputs=_proof(),
    )
    assert result.decision.decision_label == "continue_iteration"


def test_insufficient_proof_when_required_but_missing() -> None:
    result = et.build_early_termination_analysis_result(
        config=_config(allow_termination_without_dark_state_proof=False),
        kernel_result=_kernel_result(confidence_boost=0.4, converged=True, delta=0.01),
        dark_state_inputs=None,
    )
    assert result.decision.decision_label == "insufficient_proof"


def test_ambiguous_state_when_signals_conflict() -> None:
    result = et.build_early_termination_analysis_result(
        config=_config(minimum_dark_state_score=0.95, minimum_proof_consistency_score=0.95),
        kernel_result=_kernel_result(confidence_boost=0.4, converged=True, delta=0.01),
        dark_state_inputs=_proof(dark_state_score=0.5, proof_consistency_score=0.5),
    )
    assert result.decision.decision_label == "ambiguous_state"


def test_validation_rejects_bool_numeric_nan_and_bad_ranges() -> None:
    with pytest.raises(ValueError, match="must not be a bool"):
        et.build_early_termination_analysis_result(
            config=_config(minimum_top_proposal_confidence=True),
            kernel_result=_kernel_result(),
            dark_state_inputs=_proof(),
        )
    with pytest.raises(ValueError, match="must be finite"):
        et.build_early_termination_analysis_result(
            config=_config(maximum_convergence_delta=float("nan")),
            kernel_result=_kernel_result(),
            dark_state_inputs=_proof(),
        )
    with pytest.raises(ValueError, match=">= 0.0"):
        et.build_early_termination_analysis_result(
            config=_config(minimum_dark_state_score=-0.01),
            kernel_result=_kernel_result(),
            dark_state_inputs=_proof(),
        )


def test_validation_rejects_malformed_kernel_and_dark_state_inputs() -> None:
    bad_kernel = _kernel_result()
    del bad_kernel["proposals"]
    with pytest.raises(ValueError, match="missing keys"):
        et.build_early_termination_analysis_result(config=_config(), kernel_result=bad_kernel, dark_state_inputs=_proof())

    bad_kernel2 = _kernel_result()
    bad_kernel2["proposals"][0]["confidence"] = 1.5
    with pytest.raises(ValueError, match="<= 1.0"):
        et.build_early_termination_analysis_result(config=_config(), kernel_result=bad_kernel2, dark_state_inputs=_proof())

    with pytest.raises(ValueError, match="must be mapping-like"):
        et.build_early_termination_analysis_result(config=_config(), kernel_result=_kernel_result(), dark_state_inputs=3)


def test_receipt_integrity_top_proposal_hash_and_hash_changes() -> None:
    baseline = et.build_early_termination_analysis_result(
        config=_config(),
        kernel_result=_kernel_result(confidence_boost=0.4, converged=True, delta=0.01),
        dark_state_inputs=_proof(),
    )
    changed = et.build_early_termination_analysis_result(
        config=_config(),
        kernel_result=_kernel_result(confidence_boost=0.4, converged=True, delta=0.01),
        dark_state_inputs=_proof(dark_state_score=0.8),
    )

    assert baseline.receipt.receipt_hash == baseline.receipt.stable_hash()
    assert baseline.receipt.termination_decision_hash == baseline.decision.decision_hash
    assert baseline.receipt.kernel_result_hash == baseline.kernel_result["result_hash"]
    expected_top_hash = et._sha256_hex(baseline.top_proposal)
    assert baseline.receipt.top_proposal_hash == expected_top_hash
    assert baseline.stable_hash() != changed.stable_hash()
    assert baseline.receipt.termination_replay_identity != changed.receipt.termination_replay_identity


def test_guardrail_analysis_module_has_no_decoder_core_dependency() -> None:
    source = inspect.getsource(et)
    assert "qec.decoder" not in source
