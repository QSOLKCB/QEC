from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

from qec.analysis import qpe_toolbox_adapter_receipts as qtar
from qec.analysis.qpe_toolbox_adapter_receipts import (
    build_qpe_claim_scope_boundary,
    build_qpe_review_boundary,
    build_qpe_source_boundary,
    build_qpe_system_size_declaration,
    build_qpe_toolbox_adapter_receipt,
    build_qpe_toolbox_identity,
)
from tests.test_agent_pattern_decision_receipts import _receipt as _pattern_receipt


def _toolbox(toolbox_type: str = "DECLARED_QPE_TOOLBOX"):
    return build_qpe_toolbox_identity("qpe-toolbox", "1.0", toolbox_type)


def _source(mode: str = "SOURCE_HASH_BOUND", reason: str = "source-bound declaration"):
    return build_qpe_source_boundary(mode, "a" * 64, reason)


def _size(size: int = 64, reason: str = "declared deterministic size"):
    return build_qpe_system_size_declaration(size, reason)


def _review(mode: str = "REVIEWED_SOURCE", reason: str = "peer reviewed source"):
    return build_qpe_review_boundary(mode, reason)


def _scope(mode: str = "CLAIM_SCOPE_REPLAY_ONLY", reason: str = "replay-only claim scope"):
    return build_qpe_claim_scope_boundary(mode, reason)


def _receipt(adapter_only: bool = True, replay_upstream: bool = True):
    apd, trace, manifest, dispatch, crawler, deps = _pattern_receipt(adapter_only=True, replay_upstream=replay_upstream)
    rec = build_qpe_toolbox_adapter_receipt(apd, _toolbox(), _source(), _size(), _review(), _scope(), adapter_only)
    return rec, apd, trace, manifest, dispatch, crawler, deps


def test_hash_stability_canonical_pyhashseed_and_idempotent_rebuild():
    a, *_ = _receipt()
    b, *_ = _receipt()
    assert a.qpe_toolbox_adapter_receipt_hash == b.qpe_toolbox_adapter_receipt_hash
    assert a == b


def test_replay_safe_recomputed_not_trusted_and_upstream_propagation():
    rec, apd, trace, manifest, dispatch, crawler, deps = _receipt()
    assert rec.replay_safe_qpe_adapter is False
    qtar.validate_qpe_toolbox_adapter_receipt(rec, apd, agent_observation_trace_receipt=trace, skill_library_manifest=manifest, tool_dispatch_telemetry_receipt=dispatch, crawler_boundary_receipt=crawler, **deps)
    forged = object.__new__(qtar.QPEToolboxAdapterReceipt)
    object.__setattr__(forged, "schema_version", rec.schema_version)
    object.__setattr__(forged, "agent_pattern_decision_receipt_hash", rec.agent_pattern_decision_receipt_hash)
    object.__setattr__(forged, "toolbox_identity", rec.toolbox_identity)
    object.__setattr__(forged, "source_boundary", rec.source_boundary)
    object.__setattr__(forged, "system_size_declaration", rec.system_size_declaration)
    object.__setattr__(forged, "review_boundary", rec.review_boundary)
    object.__setattr__(forged, "claim_scope_boundary", rec.claim_scope_boundary)
    object.__setattr__(forged, "replay_safe_qpe_adapter", True)
    object.__setattr__(forged, "adapter_only", rec.adapter_only)
    object.__setattr__(forged, "qpe_toolbox_adapter_receipt_hash", rec.qpe_toolbox_adapter_receipt_hash)
    with pytest.raises(ValueError, match="recomputed"):
        qtar.validate_qpe_toolbox_adapter_receipt(forged, apd, agent_observation_trace_receipt=trace, skill_library_manifest=manifest, tool_dispatch_telemetry_receipt=dispatch, crawler_boundary_receipt=crawler, **deps)


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
        qtar.validate_qpe_toolbox_adapter_receipt(replace(rec, adapter_only=False), apd, agent_observation_trace_receipt=trace, skill_library_manifest=manifest, tool_dispatch_telemetry_receipt=dispatch, crawler_boundary_receipt=crawler, **deps)
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
    "hidden hardware authority", "hardware authority claim", "cosmological truth", "autonomous evaluation", "hidden replay equivalence", "hidden mutable toolbox", "runtime quantum execution", "quantum advantage established", "quantum advantage proven", "QEC advantage proven",
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


def test_hash_bound_source_requires_real_hash_and_unreviewed_reason_requires_mode():
    with pytest.raises(ValueError):
        build_qpe_source_boundary("SOURCE_HASH_BOUND", "paper:hash:abc", "source")
    with pytest.raises(ValueError):
        build_qpe_review_boundary("REVIEWED_SOURCE", "this is an unreviewed preprint")
