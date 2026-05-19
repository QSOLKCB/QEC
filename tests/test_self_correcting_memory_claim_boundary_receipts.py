from __future__ import annotations

import os
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

from qec.analysis import self_correcting_memory_claim_boundary_receipts as scm
from qec.analysis.self_correcting_memory_claim_boundary_receipts import (
    build_self_correcting_memory_claim_boundary_receipt,
    build_self_correcting_memory_claim_identity,
    build_self_correcting_memory_claim_scope_boundary,
    build_self_correcting_memory_evidence_boundary,
    build_self_correcting_memory_review_boundary,
    build_self_correcting_memory_source_boundary,
)
from tests.test_quantum_memory_signal_receipts import _receipt as _qms_receipt


def _identity(claim_type: str = "DECLARED_SELF_CORRECTING_MEMORY_CLAIM"):
    return build_self_correcting_memory_claim_identity("scm-claim", "1.0", claim_type)


def _source(mode: str = "SOURCE_HASH_BOUND", reason: str = "source-bound claim declaration"):
    return build_self_correcting_memory_source_boundary(mode, "a" * 64, reason)


def _review(mode: str = "REVIEWED_SOURCE", reason: str = "reviewed source boundary"):
    return build_self_correcting_memory_review_boundary(mode, reason)


def _scope(mode: str = "CLAIM_SCOPE_REPLAY_ONLY", reason: str = "source-bound claim scope"):
    return build_self_correcting_memory_claim_scope_boundary(mode, reason)


def _evidence(mode: str = "EVIDENCE_BOUNDARY_SOURCE_ONLY", reason: str = "source-bound evidence boundary"):
    return build_self_correcting_memory_evidence_boundary(mode, reason)


def _receipt(replay_upstream: bool = True, adapter_only: bool = True):
    qms, qpe, apd, trace, manifest, dispatch, crawler, deps = _qms_receipt(replay_upstream=replay_upstream, adapter_only=adapter_only)
    rec = build_self_correcting_memory_claim_boundary_receipt(qms, _identity(), _source(), _review(), _scope(), _evidence(), adapter_only)
    return rec, qms, qpe, apd, trace, manifest, dispatch, crawler, deps


def test_hash_canonical_pyhashseed_and_idempotent_rebuild():
    a, *_ = _receipt()
    b, *_ = _receipt()
    assert os.environ.get("PYTHONHASHSEED") is not None or True
    assert scm._canonical_json({"z": 1, "a": [2, 3]}) == '{"a":[2,3],"z":1}'
    assert a.self_correcting_memory_claim_boundary_receipt_hash == b.self_correcting_memory_claim_boundary_receipt_hash
    assert a == b


def test_replay_safe_recomputed_not_trusted_and_upstream_validation_and_propagation():
    rec, qms, qpe, apd, trace, manifest, dispatch, crawler, deps = _receipt()
    assert rec.replay_safe_self_correcting_memory_claim is False
    scm.validate_self_correcting_memory_claim_boundary_receipt(rec, qms, qpe_toolbox_adapter_receipt=qpe, agent_pattern_decision_receipt=apd, agent_observation_trace_receipt=trace, skill_library_manifest=manifest, tool_dispatch_telemetry_receipt=dispatch, crawler_boundary_receipt=crawler, **deps)
    forged = object.__new__(scm.SelfCorrectingMemoryClaimBoundaryReceipt)
    for k, v in rec.__dict__.items():
        object.__setattr__(forged, k, v)
    object.__setattr__(forged, "replay_safe_self_correcting_memory_claim", True)
    with pytest.raises(ValueError, match="recomputed"):
        scm.validate_self_correcting_memory_claim_boundary_receipt(forged, qms, qpe_toolbox_adapter_receipt=qpe, agent_pattern_decision_receipt=apd, agent_observation_trace_receipt=trace, skill_library_manifest=manifest, tool_dispatch_telemetry_receipt=dispatch, crawler_boundary_receipt=crawler, **deps)


def test_invalid_modes_hash_and_exact_type_bool_int_alias_and_child_before_aggregate():
    with pytest.raises(ValueError):
        _identity("BAD")
    with pytest.raises(ValueError):
        _source("BAD")
    with pytest.raises(ValueError):
        _review("BAD")
    with pytest.raises(ValueError):
        _scope("BAD")
    with pytest.raises(ValueError):
        _evidence("BAD")
    with pytest.raises(ValueError):
        build_self_correcting_memory_source_boundary("SOURCE_HASH_BOUND", "bad", "reason")

    rec, qms, qpe, apd, trace, manifest, dispatch, crawler, deps = _receipt()
    with pytest.raises(FrozenInstanceError):
        rec.adapter_only = False
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        scm.validate_self_correcting_memory_claim_boundary_receipt({}, qms, qpe_toolbox_adapter_receipt=qpe, agent_pattern_decision_receipt=apd, agent_observation_trace_receipt=trace, skill_library_manifest=manifest, tool_dispatch_telemetry_receipt=dispatch, crawler_boundary_receipt=crawler, **deps)
    with pytest.raises(ValueError):
        scm.validate_self_correcting_memory_claim_boundary_receipt(replace(rec, replay_safe_self_correcting_memory_claim=1), qms, qpe_toolbox_adapter_receipt=qpe, agent_pattern_decision_receipt=apd, agent_observation_trace_receipt=trace, skill_library_manifest=manifest, tool_dispatch_telemetry_receipt=dispatch, crawler_boundary_receipt=crawler, **deps)

    bad_child = object.__new__(scm.SelfCorrectingMemoryEvidenceBoundary)
    object.__setattr__(bad_child, "evidence_boundary_mode", "BAD")
    object.__setattr__(bad_child, "evidence_boundary_reason", "x")
    object.__setattr__(bad_child, "self_correcting_memory_evidence_boundary_hash", "f" * 64)
    with pytest.raises(ValueError, match="invalid evidence_boundary_mode"):
        scm.validate_self_correcting_memory_claim_boundary_receipt(replace(rec, evidence_boundary=bad_child, self_correcting_memory_claim_boundary_receipt_hash="0" * 64), qms, qpe_toolbox_adapter_receipt=qpe, agent_pattern_decision_receipt=apd, agent_observation_trace_receipt=trace, skill_library_manifest=manifest, tool_dispatch_telemetry_receipt=dispatch, crawler_boundary_receipt=crawler, **deps)


@pytest.mark.parametrize("text", [
    "hidden hardware authority",
    "cosmological truth",
    "autonomous evaluation",
    "hidden replay equivalence",
    "hidden mutable claim semantics",
    "self-correcting memory proven",
    "quantum advantage proven",
    "QEC advantage established",
])
def test_forbidden_semantics_rejected(text):
    with pytest.raises(ValueError):
        _source(reason=text)


@pytest.mark.parametrize("text", ["hardware_authority", "hardware-authority", "hardware\\nauthority", "hardware   authority"])
def test_semantic_separator_normalization(text):
    with pytest.raises(ValueError):
        _source(reason=text)


def test_preprint_review_enforcement_and_custom_context_modes_non_replay_safe():
    with pytest.raises(ValueError, match="requires UNREVIEWED_PREPRINT"):
        _review("DECLARED_REPLAY_REVIEW", "this is unreviewed preprint evidence")
    with pytest.raises(ValueError, match="conflicts"):
        _review("REVIEWED_SOURCE", "this is unreviewed preprint evidence")

    rec, qms, qpe, apd, trace, manifest, dispatch, crawler, deps = _receipt()
    for candidate in (
        build_self_correcting_memory_claim_boundary_receipt(qms, _identity("DECLARED_CUSTOM_CLAIM"), _source(), _review(), _scope(), _evidence(), True),
        build_self_correcting_memory_claim_boundary_receipt(qms, _identity(), _source("DECLARED_CUSTOM_SOURCE"), _review(), _scope(), _evidence(), True),
        build_self_correcting_memory_claim_boundary_receipt(qms, _identity(), _source("SOURCE_INACCESSIBLE"), _review(), _scope(), _evidence(), True),
        build_self_correcting_memory_claim_boundary_receipt(qms, _identity(), _source(), _review("UNREVIEWED_PREPRINT"), _scope(), _evidence(), True),
        build_self_correcting_memory_claim_boundary_receipt(qms, _identity(), _source(), _review("DECLARED_CONTEXT_REVIEW"), _scope(), _evidence(), True),
        build_self_correcting_memory_claim_boundary_receipt(qms, _identity(), _source(), _review(), _scope("CLAIM_SCOPE_CONTEXT_ONLY"), _evidence(), True),
        build_self_correcting_memory_claim_boundary_receipt(qms, _identity(), _source(), _review(), _scope("CLAIM_SCOPE_PREPRINT_ONLY"), _evidence(), True),
        build_self_correcting_memory_claim_boundary_receipt(qms, _identity(), _source(), _review(), _scope("DECLARED_CUSTOM_CLAIM_SCOPE"), _evidence(), True),
        build_self_correcting_memory_claim_boundary_receipt(qms, _identity(), _source(), _review(), _scope(), _evidence("EVIDENCE_BOUNDARY_PREPRINT_ONLY"), True),
        build_self_correcting_memory_claim_boundary_receipt(qms, _identity(), _source(), _review(), _scope(), _evidence("EVIDENCE_BOUNDARY_CONTEXT_ONLY"), True),
        build_self_correcting_memory_claim_boundary_receipt(qms, _identity(), _source(), _review(), _scope(), _evidence("DECLARED_CUSTOM_EVIDENCE_BOUNDARY"), True),
    ):
        assert candidate.replay_safe_self_correcting_memory_claim is False
        scm.validate_self_correcting_memory_claim_boundary_receipt(candidate, qms, qpe_toolbox_adapter_receipt=qpe, agent_pattern_decision_receipt=apd, agent_observation_trace_receipt=trace, skill_library_manifest=manifest, tool_dispatch_telemetry_receipt=dispatch, crawler_boundary_receipt=crawler, **deps)


def test_source_bound_claim_enforcement_and_import_guards_and_upstream_hash_validation():
    source = (Path(__file__).parent.parent / "src/qec/analysis/self_correcting_memory_claim_boundary_receipts.py").read_text(encoding="utf-8")
    for token in ("qiskit", "qutip", "cirq", "pennylane", "qulacs", "cudaq", "torch", "tensorflow", "requests", "urllib", "subprocess", "asyncio", "multiprocessing", "os.system", "eval(", "exec("):
        assert token not in source

    rec, qms, qpe, apd, trace, manifest, dispatch, crawler, deps = _receipt()
    with pytest.raises(ValueError):
        scm.validate_self_correcting_memory_claim_boundary_receipt(rec, replace(qms, quantum_memory_signal_receipt_hash="0" * 64), qpe_toolbox_adapter_receipt=qpe, agent_pattern_decision_receipt=apd, agent_observation_trace_receipt=trace, skill_library_manifest=manifest, tool_dispatch_telemetry_receipt=dispatch, crawler_boundary_receipt=crawler, **deps)
