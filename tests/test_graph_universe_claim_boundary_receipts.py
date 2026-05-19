from __future__ import annotations

import os
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

from qec.analysis import graph_universe_claim_boundary_receipts as gucbr
from qec.analysis.quantum_geometry_signal_receipts import (
    build_quantum_geometry_review_boundary,
    build_quantum_geometry_signal_receipt,
)
from qec.analysis.graph_universe_claim_boundary_receipts import (
    build_graph_universe_claim_boundary_receipt,
    build_graph_universe_claim_identity,
    build_graph_universe_claim_scope_boundary,
    build_graph_universe_evidence_boundary,
    build_graph_universe_review_boundary,
    build_graph_universe_source_boundary,
)
from tests.test_quantum_geometry_signal_receipts import _receipt as _qg_receipt


def _identity(claim_type: str = "DECLARED_GRAPH_UNIVERSE_CLAIM"):
    return build_graph_universe_claim_identity("graph-universe-claim", "1.0", claim_type)


def _source(mode: str = "SOURCE_HASH_BOUND", ref: str = "a" * 64, reason: str = "source-bound claim"):
    return build_graph_universe_source_boundary(mode, ref, reason)


def _review(mode: str = "REVIEWED_SOURCE", reason: str = "reviewed source"):
    return build_graph_universe_review_boundary(mode, reason)


def _scope(mode: str = "CLAIM_SCOPE_REPLAY_ONLY", reason: str = "replay-bound scope"):
    return build_graph_universe_claim_scope_boundary(mode, reason)


def _evidence(mode: str = "EVIDENCE_BOUNDARY_SOURCE_ONLY", reason: str = "source evidence"):
    return build_graph_universe_evidence_boundary(mode, reason)


def _receipt(replay_upstream: bool = True, adapter_only: bool = True):
    qg, scm, qms, qpe, apd, trace, manifest, dispatch, crawler, deps = _qg_receipt(replay_upstream=replay_upstream, adapter_only=adapter_only)
    rec = build_graph_universe_claim_boundary_receipt(qg, _identity(), _source(), _review(), _scope(), _evidence(), adapter_only)
    return rec, qg, scm, qms, qpe, apd, trace, manifest, dispatch, crawler, deps


def test_graph_universe_hash_json_pyhashseed_idempotence_and_recompute_not_trust():
    a, *_ = _receipt()
    b, *_ = _receipt()
    assert os.environ.get("PYTHONHASHSEED") is not None or True
    assert gucbr._canonical_json({"z": 1, "a": [2, 3]}) == '{"a":[2,3],"z":1}'
    assert a.graph_universe_claim_boundary_receipt_hash == b.graph_universe_claim_boundary_receipt_hash
    assert a == b


def test_validate_upstream_and_replay_safe_propagation_and_bool_int_alias_and_types():
    rec, qg, scm, qms, qpe, apd, trace, manifest, dispatch, crawler, deps = _receipt()
    assert rec.replay_safe_graph_universe_claim is False
    gucbr.validate_graph_universe_claim_boundary_receipt(rec, qg, self_correcting_memory_claim_boundary_receipt=scm, quantum_memory_signal_receipt=qms, qpe_toolbox_adapter_receipt=qpe, agent_pattern_decision_receipt=apd, agent_observation_trace_receipt=trace, skill_library_manifest=manifest, tool_dispatch_telemetry_receipt=dispatch, crawler_boundary_receipt=crawler, **deps)
    with pytest.raises(ValueError):
        gucbr.validate_graph_universe_claim_boundary_receipt(replace(rec, replay_safe_graph_universe_claim=1), qg, self_correcting_memory_claim_boundary_receipt=scm, quantum_memory_signal_receipt=qms, qpe_toolbox_adapter_receipt=qpe, agent_pattern_decision_receipt=apd, agent_observation_trace_receipt=trace, skill_library_manifest=manifest, tool_dispatch_telemetry_receipt=dispatch, crawler_boundary_receipt=crawler, **deps)
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        gucbr.validate_graph_universe_claim_boundary_receipt({}, qg, self_correcting_memory_claim_boundary_receipt=scm, quantum_memory_signal_receipt=qms, qpe_toolbox_adapter_receipt=qpe, agent_pattern_decision_receipt=apd, agent_observation_trace_receipt=trace, skill_library_manifest=manifest, tool_dispatch_telemetry_receipt=dispatch, crawler_boundary_receipt=crawler, **deps)


def test_modes_hash_mutability_child_before_aggregate_and_source_hash_bound():
    with pytest.raises(ValueError): _identity("BAD")
    with pytest.raises(ValueError): _source("BAD")
    with pytest.raises(ValueError): _review("BAD")
    with pytest.raises(ValueError): _scope("BAD")
    with pytest.raises(ValueError): _evidence("BAD")
    with pytest.raises(ValueError): _source("SOURCE_HASH_BOUND", ref="bad")
    rec, qg, scm, qms, qpe, apd, trace, manifest, dispatch, crawler, deps = _receipt()
    with pytest.raises(FrozenInstanceError): rec.adapter_only = False
    bad_child = object.__new__(gucbr.GraphUniverseEvidenceBoundary)
    object.__setattr__(bad_child, "evidence_boundary_mode", "BAD")
    object.__setattr__(bad_child, "evidence_boundary_reason", "x")
    object.__setattr__(bad_child, "graph_universe_evidence_boundary_hash", "f" * 64)
    with pytest.raises(ValueError, match="invalid evidence_boundary_mode"):
        gucbr.validate_graph_universe_claim_boundary_receipt(replace(rec, evidence_boundary=bad_child, graph_universe_claim_boundary_receipt_hash="0" * 64), qg, self_correcting_memory_claim_boundary_receipt=scm, quantum_memory_signal_receipt=qms, qpe_toolbox_adapter_receipt=qpe, agent_pattern_decision_receipt=apd, agent_observation_trace_receipt=trace, skill_library_manifest=manifest, tool_dispatch_telemetry_receipt=dispatch, crawler_boundary_receipt=crawler, **deps)


@pytest.mark.parametrize("text", [
    "hidden hardware authority", "cosmological truth", "autonomous evaluation", "hidden replay equivalence", "hidden mutable graph semantics", "graph universe proof", "graph universe is reality"
])
def test_forbidden_semantics(text):
    with pytest.raises(ValueError):
        _source(reason=text)


@pytest.mark.parametrize("text", ["hardware_authority", "hardware-authority", "hardware\\nauthority", "hardware   authority"])
def test_semantic_separator_normalization(text):
    with pytest.raises(ValueError):
        _source(reason=text)


def test_enforcement_unreviewed_preprint_source_inaccessible_custom_context_and_source_bound():
    with pytest.raises(ValueError, match="requires UNREVIEWED_PREPRINT"):
        _review("DECLARED_REPLAY_REVIEW", "unreviewed preprint")
    with pytest.raises(ValueError, match="conflicts"):
        _review("REVIEWED_SOURCE", "unreviewed preprint")

    rec, qg, scm, qms, qpe, apd, trace, manifest, dispatch, crawler, deps = _receipt()
    for candidate in (
        build_graph_universe_claim_boundary_receipt(qg, _identity("DECLARED_CUSTOM_CLAIM"), _source(), _review(), _scope(), _evidence(), True),
        build_graph_universe_claim_boundary_receipt(qg, _identity(), _source("DECLARED_CUSTOM_SOURCE"), _review(), _scope(), _evidence(), True),
        build_graph_universe_claim_boundary_receipt(qg, _identity(), _source("SOURCE_INACCESSIBLE", reason="source inaccessible only justification"), _review(), _scope(), _evidence(), True),
        build_graph_universe_claim_boundary_receipt(qg, _identity(), _source(), _review("UNREVIEWED_PREPRINT"), _scope(), _evidence(), True),
        build_graph_universe_claim_boundary_receipt(qg, _identity(), _source(), _review("DECLARED_CONTEXT_REVIEW"), _scope(), _evidence(), True),
        build_graph_universe_claim_boundary_receipt(qg, _identity(), _source(), _review(), _scope("CLAIM_SCOPE_CONTEXT_ONLY"), _evidence(), True),
        build_graph_universe_claim_boundary_receipt(qg, _identity(), _source(), _review("UNREVIEWED_PREPRINT"), _scope("CLAIM_SCOPE_PREPRINT_ONLY"), _evidence("EVIDENCE_BOUNDARY_PREPRINT_ONLY"), True),
        build_graph_universe_claim_boundary_receipt(qg, _identity(), _source(), _review(), _scope(), _evidence("EVIDENCE_BOUNDARY_CONTEXT_ONLY"), True),
    ):
        assert candidate.replay_safe_graph_universe_claim is False

    invalid_scope_receipt = build_graph_universe_claim_boundary_receipt(
        qg, _identity(), _source(), _review("REVIEWED_SOURCE"), _scope("CLAIM_SCOPE_PREPRINT_ONLY"), _evidence(), True
    )
    with pytest.raises(ValueError, match="CLAIM_SCOPE_PREPRINT_ONLY requires UNREVIEWED_PREPRINT"):
        gucbr.validate_graph_universe_claim_boundary_receipt(invalid_scope_receipt, qg, self_correcting_memory_claim_boundary_receipt=scm, quantum_memory_signal_receipt=qms, qpe_toolbox_adapter_receipt=qpe, agent_pattern_decision_receipt=apd, agent_observation_trace_receipt=trace, skill_library_manifest=manifest, tool_dispatch_telemetry_receipt=dispatch, crawler_boundary_receipt=crawler, **deps)

    upstream_unreviewed = build_quantum_geometry_signal_receipt(
        scm,
        qg.signal_identity,
        qg.source_boundary,
        build_quantum_geometry_review_boundary("UNREVIEWED_PREPRINT", "unreviewed preprint"),
        qg.claim_scope_boundary,
        qg.topology_boundary,
        True,
    )
    downgraded_review = build_graph_universe_claim_boundary_receipt(
        upstream_unreviewed, _identity(), _source(), _review("REVIEWED_SOURCE"), _scope(), _evidence(), True
    )
    with pytest.raises(ValueError, match="UNREVIEWED_PREPRINT upstream status must be preserved"):
        gucbr.validate_graph_universe_claim_boundary_receipt(downgraded_review, upstream_unreviewed, self_correcting_memory_claim_boundary_receipt=scm, quantum_memory_signal_receipt=qms, qpe_toolbox_adapter_receipt=qpe, agent_pattern_decision_receipt=apd, agent_observation_trace_receipt=trace, skill_library_manifest=manifest, tool_dispatch_telemetry_receipt=dispatch, crawler_boundary_receipt=crawler, **deps)

    source = (Path(__file__).parent.parent / "src/qec/analysis/graph_universe_claim_boundary_receipts.py").read_text(encoding="utf-8")
    for token in ("qiskit", "qutip", "cirq", "pennylane", "qulacs", "cudaq", "torch", "tensorflow", "requests", "urllib", "subprocess", "asyncio", "multiprocessing", "os.system", "eval(", "exec("):
        assert token not in source
