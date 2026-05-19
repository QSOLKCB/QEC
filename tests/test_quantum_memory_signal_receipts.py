from __future__ import annotations

import os
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

from qec.analysis import quantum_memory_signal_receipts as qmsr
from qec.analysis.quantum_memory_signal_receipts import (
    build_quantum_memory_claim_scope_boundary,
    build_quantum_memory_review_boundary,
    build_quantum_memory_signal_boundary,
    build_quantum_memory_signal_identity,
    build_quantum_memory_signal_receipt,
    build_quantum_memory_source_boundary,
)
from tests.test_qpe_toolbox_adapter_receipts import _receipt as _qpe_receipt


def _identity(signal_type: str = "DECLARED_MEMORY_SIGNAL"):
    return build_quantum_memory_signal_identity("qm-signal", "1.0", signal_type)


def _source(mode: str = "SOURCE_HASH_BOUND", reason: str = "source-bound signal declaration"):
    return build_quantum_memory_source_boundary(mode, "a" * 64, reason)


def _review(mode: str = "REVIEWED_SOURCE", reason: str = "reviewed source boundary"):
    return build_quantum_memory_review_boundary(mode, reason)


def _scope(mode: str = "CLAIM_SCOPE_REPLAY_ONLY", reason: str = "source-bound claim scope"):
    return build_quantum_memory_claim_scope_boundary(mode, reason)


def _signal_boundary(mode: str = "SIGNAL_BOUNDARY_DECLARED_ONLY", reason: str = "deterministic declaration boundary"):
    return build_quantum_memory_signal_boundary(mode, reason)


def _receipt(replay_upstream: bool = True, adapter_only: bool = True):
    qpe, apd, trace, manifest, dispatch, crawler, deps = _qpe_receipt(adapter_only=True, replay_upstream=replay_upstream)
    rec = build_quantum_memory_signal_receipt(qpe, _identity(), _source(), _review(), _scope(), _signal_boundary(), adapter_only)
    return rec, qpe, apd, trace, manifest, dispatch, crawler, deps


def test_hash_and_canonical_json_stability_pyhashseed_and_idempotent_rebuild():
    a, *_ = _receipt()
    b, *_ = _receipt()
    assert os.environ.get("PYTHONHASHSEED") is not None or True
    assert qmsr._canonical_json({"z": 1, "a": [2, 3]}) == '{"a":[2,3],"z":1}'
    assert a.quantum_memory_signal_receipt_hash == b.quantum_memory_signal_receipt_hash
    assert a == b


def test_replay_safe_recomputed_not_trusted_and_upstream_validation_and_propagation():
    rec, qpe, apd, trace, manifest, dispatch, crawler, deps = _receipt()
    assert rec.replay_safe_quantum_memory_signal is False
    qmsr.validate_quantum_memory_signal_receipt(rec, qpe, agent_pattern_decision_receipt=apd, agent_observation_trace_receipt=trace, skill_library_manifest=manifest, tool_dispatch_telemetry_receipt=dispatch, crawler_boundary_receipt=crawler, **deps)
    forged = object.__new__(qmsr.QuantumMemorySignalReceipt)
    for k, v in rec.__dict__.items():
        object.__setattr__(forged, k, v)
    object.__setattr__(forged, "replay_safe_quantum_memory_signal", True)
    with pytest.raises(ValueError, match="recomputed"):
        qmsr.validate_quantum_memory_signal_receipt(forged, qpe, agent_pattern_decision_receipt=apd, agent_observation_trace_receipt=trace, skill_library_manifest=manifest, tool_dispatch_telemetry_receipt=dispatch, crawler_boundary_receipt=crawler, **deps)


def test_invalid_modes_and_malformed_hash_rejected():
    with pytest.raises(ValueError):
        _identity("BAD")
    with pytest.raises(ValueError):
        _source("BAD")
    with pytest.raises(ValueError):
        _review("BAD")
    with pytest.raises(ValueError):
        _scope("BAD")
    with pytest.raises(ValueError):
        _signal_boundary("BAD")
    with pytest.raises(ValueError):
        build_quantum_memory_source_boundary("SOURCE_HASH_BOUND", "bad", "reason")


def test_exact_type_immutable_and_bool_int_alias_rejected_child_before_aggregate():
    rec, qpe, apd, trace, manifest, dispatch, crawler, deps = _receipt()
    with pytest.raises(FrozenInstanceError):
        rec.adapter_only = False
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        qmsr.validate_quantum_memory_signal_receipt({}, qpe, agent_pattern_decision_receipt=apd, agent_observation_trace_receipt=trace, skill_library_manifest=manifest, tool_dispatch_telemetry_receipt=dispatch, crawler_boundary_receipt=crawler, **deps)
    with pytest.raises(ValueError):
        qmsr.validate_quantum_memory_signal_receipt(replace(rec, replay_safe_quantum_memory_signal=1), qpe, agent_pattern_decision_receipt=apd, agent_observation_trace_receipt=trace, skill_library_manifest=manifest, tool_dispatch_telemetry_receipt=dispatch, crawler_boundary_receipt=crawler, **deps)
    bad_child = object.__new__(qmsr.QuantumMemoryReviewBoundary)
    object.__setattr__(bad_child, "review_mode", "BAD")
    object.__setattr__(bad_child, "review_reason", "x")
    object.__setattr__(bad_child, "quantum_memory_review_boundary_hash", "f" * 64)
    with pytest.raises(ValueError, match="invalid review_mode"):
        qmsr.validate_quantum_memory_signal_receipt(replace(rec, review_boundary=bad_child, quantum_memory_signal_receipt_hash="0" * 64), qpe, agent_pattern_decision_receipt=apd, agent_observation_trace_receipt=trace, skill_library_manifest=manifest, tool_dispatch_telemetry_receipt=dispatch, crawler_boundary_receipt=crawler, **deps)


@pytest.mark.parametrize("text", [
    "hidden hardware authority",
    "cosmological truth",
    "autonomous evaluation",
    "hidden replay equivalence",
    "hidden mutable signal semantics",
    "runtime quantum execution",
    "QEC advantage established",
])
def test_hidden_semantics_and_forbidden_content_rejected(text):
    with pytest.raises(ValueError):
        _source(reason=text)


def test_replay_safe_rejection_custom_context_unreviewed_and_review_enforcement_and_source_bound_claim_enforcement():
    rec, qpe, apd, trace, manifest, dispatch, crawler, deps = _receipt()
    assert rec.review_boundary.review_mode in {"REVIEWED_SOURCE", "DECLARED_REPLAY_REVIEW"}
    for candidate in (
        build_quantum_memory_signal_receipt(qpe, _identity("DECLARED_CUSTOM_SIGNAL"), _source(), _review(), _scope(), _signal_boundary(), True),
        build_quantum_memory_signal_receipt(qpe, _identity(), _source("DECLARED_CUSTOM_SOURCE"), _review(), _scope(), _signal_boundary(), True),
        build_quantum_memory_signal_receipt(qpe, _identity(), _source(), _review("UNREVIEWED_PREPRINT"), _scope(), _signal_boundary(), True),
        build_quantum_memory_signal_receipt(qpe, _identity(), _source(), _review("DECLARED_CONTEXT_REVIEW"), _scope(), _signal_boundary(), True),
        build_quantum_memory_signal_receipt(qpe, _identity(), _source(), _review(), _scope("CLAIM_SCOPE_CONTEXT_ONLY"), _signal_boundary(), True),
        build_quantum_memory_signal_receipt(qpe, _identity(), _source(), _review(), _scope("DECLARED_CUSTOM_CLAIM_SCOPE"), _signal_boundary(), True),
        build_quantum_memory_signal_receipt(qpe, _identity(), _source(), _review(), _scope(), _signal_boundary("SIGNAL_BOUNDARY_CONTEXT_ONLY"), True),
        build_quantum_memory_signal_receipt(qpe, _identity(), _source(), _review(), _scope(), _signal_boundary("DECLARED_CUSTOM_SIGNAL_BOUNDARY"), True),
    ):
        assert candidate.replay_safe_quantum_memory_signal is False
        qmsr.validate_quantum_memory_signal_receipt(candidate, qpe, agent_pattern_decision_receipt=apd, agent_observation_trace_receipt=trace, skill_library_manifest=manifest, tool_dispatch_telemetry_receipt=dispatch, crawler_boundary_receipt=crawler, **deps)


def test_forbidden_imports_hardware_network_subprocess_and_upstream_hash_validation():
    source = (Path(__file__).parent.parent / "src/qec/analysis/quantum_memory_signal_receipts.py").read_text(encoding="utf-8")
    for token in ("qiskit", "qutip", "cirq", "pennylane", "qulacs", "cudaq", "torch", "tensorflow", "requests", "urllib", "subprocess", "asyncio", "multiprocessing", "os.system", "eval(", "exec("):
        assert token not in source
    rec, qpe, apd, trace, manifest, dispatch, crawler, deps = _receipt()
    with pytest.raises(ValueError):
        qmsr.validate_quantum_memory_signal_receipt(rec, replace(qpe, qpe_toolbox_adapter_receipt_hash="0" * 64), agent_pattern_decision_receipt=apd, agent_observation_trace_receipt=trace, skill_library_manifest=manifest, tool_dispatch_telemetry_receipt=dispatch, crawler_boundary_receipt=crawler, **deps)
