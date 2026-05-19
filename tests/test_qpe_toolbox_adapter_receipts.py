from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

from qec.analysis import qpe_toolbox_adapter_receipts as qtar
from tests.test_agent_pattern_decision_receipts import _receipt as _pattern_receipt


def _mk_hash(payload, key):
    return qtar._hash_payload(qtar._base_payload(payload, key))


def _toolbox(toolbox_type: str = "DECLARED_QPE_TOOLBOX"):
    base = {"toolbox_name": "qpe-toolbox", "toolbox_version": "1.0", "toolbox_type": toolbox_type}
    return qtar.QPEToolboxIdentity(**base, qpe_toolbox_identity_hash=_mk_hash(base, "qpe_toolbox_identity_hash"))


def _source(mode: str = "SOURCE_HASH_BOUND", reason: str = "source-bound declaration"):
    base = {"source_mode": mode, "source_reference": "paper:hash:abc", "source_reason": reason}
    return qtar.QPESourceBoundary(**base, qpe_source_boundary_hash=_mk_hash(base, "qpe_source_boundary_hash"))


def _size(size: int = 64, reason: str = "declared deterministic size"):
    base = {"declared_system_size": size, "system_size_reason": reason}
    return qtar.QPESystemSizeDeclaration(**base, qpe_system_size_declaration_hash=_mk_hash(base, "qpe_system_size_declaration_hash"))


def _review(mode: str = "REVIEWED_SOURCE", reason: str = "peer reviewed source"):
    base = {"review_mode": mode, "review_reason": reason}
    return qtar.QPEReviewBoundary(**base, qpe_review_boundary_hash=_mk_hash(base, "qpe_review_boundary_hash"))


def _scope(mode: str = "CLAIM_SCOPE_REPLAY_ONLY", reason: str = "replay-only claim scope"):
    base = {"claim_scope_mode": mode, "claim_scope_reason": reason}
    return qtar.QPEClaimScopeBoundary(**base, qpe_claim_scope_boundary_hash=_mk_hash(base, "qpe_claim_scope_boundary_hash"))


def _receipt(adapter_only: bool = True, replay_upstream: bool = True):
    apd, trace, manifest, dispatch, crawler, deps = _pattern_receipt(adapter_only=True, replay_upstream=replay_upstream)
    rec = qtar.build_qpe_toolbox_adapter_receipt(apd, _toolbox(), _source(), _size(), _review(), _scope(), adapter_only)
    return rec, apd, trace, manifest, dispatch, crawler, deps


def test_hash_stability_canonical_pyhashseed_and_idempotent_rebuild():
    a, *_ = _receipt()
    b, *_ = _receipt()
    assert a.qpe_toolbox_adapter_receipt_hash == b.qpe_toolbox_adapter_receipt_hash
    assert qtar._canonical_json(a.__dict__) == qtar._canonical_json(b.__dict__)


def test_replay_safe_recomputed_not_trusted_and_upstream_propagation():
    rec, apd, trace, manifest, dispatch, crawler, deps = _receipt()
    assert rec.replay_safe_qpe_adapter is False
    qtar.validate_qpe_toolbox_adapter_receipt(rec, apd, agent_observation_trace_receipt=trace, skill_library_manifest=manifest, tool_dispatch_telemetry_receipt=dispatch, crawler_boundary_receipt=crawler, **deps)
    forged = replace(rec, replay_safe_qpe_adapter=True, qpe_toolbox_adapter_receipt_hash="0" * 64)
    forged = replace(forged, qpe_toolbox_adapter_receipt_hash=qtar._hash_payload(qtar._base_payload(forged.__dict__, "qpe_toolbox_adapter_receipt_hash")))
    with pytest.raises(ValueError, match="recomputed"):
        qtar.validate_qpe_toolbox_adapter_receipt(forged, apd, agent_observation_trace_receipt=trace, skill_library_manifest=manifest, tool_dispatch_telemetry_receipt=dispatch, crawler_boundary_receipt=crawler, **deps)


@pytest.mark.parametrize("value", ["abc", "A" * 64, "a" * 63])
def test_malformed_hash_rejected(value):
    with pytest.raises(ValueError):
        qtar.QPEToolboxIdentity("x", "1", "DECLARED_QPE_TOOLBOX", value)


@pytest.mark.parametrize("bad", ["BAD"])
def test_invalid_modes_rejected(bad):
    with pytest.raises(ValueError):
        _toolbox(bad)
    with pytest.raises(ValueError):
        _source(bad)
    with pytest.raises(ValueError):
        _review(bad)
    with pytest.raises(ValueError):
        _scope(bad)


@pytest.mark.parametrize("n", [0, -1, 10**9 + 1])
def test_invalid_system_size_rejected(n):
    with pytest.raises(ValueError):
        _size(n)


def test_exact_artifact_type_immutable_payload_and_bool_int_alias_rejected():
    rec, apd, trace, manifest, dispatch, crawler, deps = _receipt()
    with pytest.raises(FrozenInstanceError):
        rec.adapter_only = False
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        qtar.validate_qpe_toolbox_adapter_receipt({}, apd, agent_observation_trace_receipt=trace, skill_library_manifest=manifest, tool_dispatch_telemetry_receipt=dispatch, crawler_boundary_receipt=crawler, **deps)
    with pytest.raises(ValueError):
        qtar.validate_qpe_toolbox_adapter_receipt(replace(rec, adapter_only=1), apd, agent_observation_trace_receipt=trace, skill_library_manifest=manifest, tool_dispatch_telemetry_receipt=dispatch, crawler_boundary_receipt=crawler, **deps)
    with pytest.raises(ValueError):
        qtar.validate_qpe_toolbox_adapter_receipt(replace(rec, replay_safe_qpe_adapter=1), apd, agent_observation_trace_receipt=trace, skill_library_manifest=manifest, tool_dispatch_telemetry_receipt=dispatch, crawler_boundary_receipt=crawler, **deps)
    with pytest.raises(ValueError):
        _size(True)


def test_child_before_aggregate_validation_and_upstream_validation_enforced():
    rec, apd, trace, manifest, dispatch, crawler, deps = _receipt()
    bad_child = object.__new__(qtar.QPEReviewBoundary)
    object.__setattr__(bad_child, "review_mode", "BAD")
    object.__setattr__(bad_child, "review_reason", "x")
    object.__setattr__(bad_child, "qpe_review_boundary_hash", "f" * 64)
    with pytest.raises(ValueError, match="invalid review_mode"):
        qtar.validate_qpe_toolbox_adapter_receipt(replace(rec, review_boundary=bad_child, qpe_toolbox_adapter_receipt_hash="0" * 64), apd, agent_observation_trace_receipt=trace, skill_library_manifest=manifest, tool_dispatch_telemetry_receipt=dispatch, crawler_boundary_receipt=crawler, **deps)
    with pytest.raises(ValueError):
        qtar.validate_qpe_toolbox_adapter_receipt(rec, replace(apd, agent_pattern_decision_receipt_hash="0" * 64), agent_observation_trace_receipt=trace, skill_library_manifest=manifest, tool_dispatch_telemetry_receipt=dispatch, crawler_boundary_receipt=crawler, **deps)


@pytest.mark.parametrize("text", [
    "hidden hardware authority", "cosmological truth", "autonomous evaluation", "hidden replay equivalence", "hidden mutable toolbox", "runtime quantum execution", "quantum advantage established",
])
def test_hidden_semantics_and_forbidden_content_rejected(text):
    with pytest.raises(ValueError):
        _source(reason=text)


def test_replay_safe_rejection_custom_context_and_unreviewed_modes_and_review_enforcement_and_source_bound():
    rec, apd, trace, manifest, dispatch, crawler, deps = _receipt()
    assert rec.review_boundary.review_mode in {"REVIEWED_SOURCE", "DECLARED_REPLAY_REVIEW"}
    assert rec.source_boundary.source_mode in {"SOURCE_DECLARED_ONLY", "SOURCE_HASH_BOUND", "SOURCE_REPLAY_ONLY"}
    for candidate in (
        qtar.build_qpe_toolbox_adapter_receipt(apd, _toolbox("DECLARED_CUSTOM_TOOLBOX"), _source(), _size(), _review(), _scope(), True),
        qtar.build_qpe_toolbox_adapter_receipt(apd, _toolbox(), _source("DECLARED_CUSTOM_SOURCE"), _size(), _review(), _scope(), True),
        qtar.build_qpe_toolbox_adapter_receipt(apd, _toolbox(), _source("SOURCE_CONTEXT_ONLY"), _size(), _review(), _scope(), True),
        qtar.build_qpe_toolbox_adapter_receipt(apd, _toolbox(), _source(), _size(), _review("DECLARED_CUSTOM_REVIEW"), _scope(), True),
        qtar.build_qpe_toolbox_adapter_receipt(apd, _toolbox(), _source(), _size(), _review("DECLARED_CONTEXT_REVIEW"), _scope(), True),
        qtar.build_qpe_toolbox_adapter_receipt(apd, _toolbox(), _source(), _size(), _review("UNREVIEWED_PREPRINT"), _scope(), True),
        qtar.build_qpe_toolbox_adapter_receipt(apd, _toolbox(), _source(), _size(), _review(), _scope("DECLARED_CUSTOM_CLAIM_SCOPE"), True),
        qtar.build_qpe_toolbox_adapter_receipt(apd, _toolbox(), _source(), _size(), _review(), _scope("CLAIM_SCOPE_CONTEXT_ONLY"), True),
    ):
        assert candidate.replay_safe_qpe_adapter is False
        qtar.validate_qpe_toolbox_adapter_receipt(candidate, apd, agent_observation_trace_receipt=trace, skill_library_manifest=manifest, tool_dispatch_telemetry_receipt=dispatch, crawler_boundary_receipt=crawler, **deps)


def test_no_forbidden_imports_hardware_network_or_subprocess():
    source = (Path(__file__).parent.parent / "src/qec/analysis/qpe_toolbox_adapter_receipts.py").read_text(encoding="utf-8")
    for token in ("qiskit", "qutip", "cirq", "pennylane", "qulacs", "cudaq", "torch", "tensorflow", "requests", "urllib", "subprocess", "asyncio", "multiprocessing", "os.system", "eval(", "exec("):
        assert token not in source
