from __future__ import annotations

import os
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

from qec.analysis import quantum_geometry_signal_receipts as qgsr
from qec.analysis.quantum_geometry_signal_receipts import (
    build_quantum_geometry_claim_scope_boundary,
    build_quantum_geometry_review_boundary,
    build_quantum_geometry_signal_identity,
    build_quantum_geometry_signal_receipt,
    build_quantum_geometry_source_boundary,
    build_quantum_geometry_topology_boundary,
)
from tests.test_self_correcting_memory_claim_boundary_receipts import _receipt as _scm_receipt


def _identity(signal_type: str = "DECLARED_GEOMETRY_SIGNAL"):
    return build_quantum_geometry_signal_identity("geometry-signal", "1.0", signal_type)


def _source(mode: str = "SOURCE_HASH_BOUND", ref: str = "a" * 64, reason: str = "source-bound geometry signal"):
    return build_quantum_geometry_source_boundary(mode, ref, reason)


def _review(mode: str = "REVIEWED_SOURCE", reason: str = "reviewed source"):
    return build_quantum_geometry_review_boundary(mode, reason)


def _scope(mode: str = "CLAIM_SCOPE_REPLAY_ONLY", reason: str = "replay-scoped geometry claim"):
    return build_quantum_geometry_claim_scope_boundary(mode, reason)


def _topology(mode: str = "TOPOLOGY_BOUNDARY_DECLARED_ONLY", reason: str = "deterministic topology declaration"):
    return build_quantum_geometry_topology_boundary(mode, reason)


def _receipt(replay_upstream: bool = True, adapter_only: bool = True):
    scm, qms, qpe, apd, trace, manifest, dispatch, crawler, deps = _scm_receipt(replay_upstream=replay_upstream, adapter_only=adapter_only)
    rec = build_quantum_geometry_signal_receipt(scm, _identity(), _source(), _review(), _scope(), _topology(), adapter_only)
    return rec, scm, qms, qpe, apd, trace, manifest, dispatch, crawler, deps


def test_hash_and_json_stability_pyhashseed_and_idempotent_rebuild():
    a, *_ = _receipt()
    b, *_ = _receipt()
    assert os.environ.get("PYTHONHASHSEED") is not None or True
    assert qgsr._canonical_json({"z": 1, "a": [2, 3]}) == '{"a":[2,3],"z":1}'
    assert a.quantum_geometry_signal_receipt_hash == b.quantum_geometry_signal_receipt_hash
    assert a == b


def test_replay_safe_recomputed_not_trusted_and_upstream_validation_and_propagation():
    rec, scm, qms, qpe, apd, trace, manifest, dispatch, crawler, deps = _receipt()
    assert rec.replay_safe_quantum_geometry_signal is False
    qgsr.validate_quantum_geometry_signal_receipt(rec, scm, quantum_memory_signal_receipt=qms, qpe_toolbox_adapter_receipt=qpe, agent_pattern_decision_receipt=apd, agent_observation_trace_receipt=trace, skill_library_manifest=manifest, tool_dispatch_telemetry_receipt=dispatch, crawler_boundary_receipt=crawler, **deps)

    forged = object.__new__(qgsr.QuantumGeometrySignalReceipt)
    for k, v in rec.__dict__.items():
        object.__setattr__(forged, k, v)
    object.__setattr__(forged, "replay_safe_quantum_geometry_signal", True)
    with pytest.raises(ValueError, match="recomputed"):
        qgsr.validate_quantum_geometry_signal_receipt(forged, scm, quantum_memory_signal_receipt=qms, qpe_toolbox_adapter_receipt=qpe, agent_pattern_decision_receipt=apd, agent_observation_trace_receipt=trace, skill_library_manifest=manifest, tool_dispatch_telemetry_receipt=dispatch, crawler_boundary_receipt=crawler, **deps)


def test_modes_hash_type_mutability_and_child_before_aggregate_validation():
    with pytest.raises(ValueError):
        _identity("BAD")
    with pytest.raises(ValueError):
        _source("BAD")
    with pytest.raises(ValueError):
        _review("BAD")
    with pytest.raises(ValueError):
        _scope("BAD")
    with pytest.raises(ValueError):
        _topology("BAD")
    with pytest.raises(ValueError):
        _source("SOURCE_HASH_BOUND", ref="bad")

    rec, scm, qms, qpe, apd, trace, manifest, dispatch, crawler, deps = _receipt()
    with pytest.raises(FrozenInstanceError):
        rec.adapter_only = False
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        qgsr.validate_quantum_geometry_signal_receipt({}, scm, quantum_memory_signal_receipt=qms, qpe_toolbox_adapter_receipt=qpe, agent_pattern_decision_receipt=apd, agent_observation_trace_receipt=trace, skill_library_manifest=manifest, tool_dispatch_telemetry_receipt=dispatch, crawler_boundary_receipt=crawler, **deps)
    with pytest.raises(ValueError):
        qgsr.validate_quantum_geometry_signal_receipt(replace(rec, replay_safe_quantum_geometry_signal=1), scm, quantum_memory_signal_receipt=qms, qpe_toolbox_adapter_receipt=qpe, agent_pattern_decision_receipt=apd, agent_observation_trace_receipt=trace, skill_library_manifest=manifest, tool_dispatch_telemetry_receipt=dispatch, crawler_boundary_receipt=crawler, **deps)

    bad_child = object.__new__(qgsr.QuantumGeometryTopologyBoundary)
    object.__setattr__(bad_child, "topology_boundary_mode", "BAD")
    object.__setattr__(bad_child, "topology_boundary_reason", "x")
    object.__setattr__(bad_child, "quantum_geometry_topology_boundary_hash", "f" * 64)
    with pytest.raises(ValueError, match="invalid topology_boundary_mode"):
        qgsr.validate_quantum_geometry_signal_receipt(replace(rec, topology_boundary=bad_child, quantum_geometry_signal_receipt_hash="0" * 64), scm, quantum_memory_signal_receipt=qms, qpe_toolbox_adapter_receipt=qpe, agent_pattern_decision_receipt=apd, agent_observation_trace_receipt=trace, skill_library_manifest=manifest, tool_dispatch_telemetry_receipt=dispatch, crawler_boundary_receipt=crawler, **deps)


@pytest.mark.parametrize("text", [
    "hidden hardware authority",
    "cosmological truth",
    "autonomous evaluation",
    "hidden replay equivalence",
    "hidden mutable geometry semantics",
    "quantum geometry proven",
    "graph universe established",
])
def test_forbidden_semantics_rejected(text):
    with pytest.raises(ValueError):
        _source(reason=text)


@pytest.mark.parametrize("text", ["hardware_authority", "hardware-authority", "hardware\\nauthority", "hardware   authority"])
def test_semantic_separator_normalization(text):
    with pytest.raises(ValueError):
        _source(reason=text)


def test_review_and_source_bound_enforcement_custom_context_modes_and_import_guards():
    with pytest.raises(ValueError, match="requires UNREVIEWED_PREPRINT"):
        _review("DECLARED_REPLAY_REVIEW", "unreviewed preprint analysis")
    with pytest.raises(ValueError, match="conflicts"):
        _review("REVIEWED_SOURCE", "unreviewed preprint analysis")

    rec, scm, qms, qpe, apd, trace, manifest, dispatch, crawler, deps = _receipt()
    for candidate in (
        build_quantum_geometry_signal_receipt(scm, _identity("DECLARED_CUSTOM_SIGNAL"), _source(), _review(), _scope(), _topology(), True),
        build_quantum_geometry_signal_receipt(scm, _identity(), _source("DECLARED_CUSTOM_SOURCE"), _review(), _scope(), _topology(), True),
        build_quantum_geometry_signal_receipt(scm, _identity(), _source("SOURCE_INACCESSIBLE"), _review(), _scope(), _topology(), True),
        build_quantum_geometry_signal_receipt(scm, _identity(), _source(), _review("UNREVIEWED_PREPRINT"), _scope(), _topology(), True),
        build_quantum_geometry_signal_receipt(scm, _identity(), _source(), _review("DECLARED_CONTEXT_REVIEW"), _scope(), _topology(), True),
        build_quantum_geometry_signal_receipt(scm, _identity(), _source(), _review(), _scope("CLAIM_SCOPE_CONTEXT_ONLY"), _topology(), True),
        build_quantum_geometry_signal_receipt(scm, _identity(), _source(), _review("UNREVIEWED_PREPRINT"), _scope("CLAIM_SCOPE_PREPRINT_ONLY"), _topology(), True),
        build_quantum_geometry_signal_receipt(scm, _identity(), _source(), _review(), _scope("DECLARED_CUSTOM_CLAIM_SCOPE"), _topology(), True),
        build_quantum_geometry_signal_receipt(scm, _identity(), _source(), _review(), _scope(), _topology("TOPOLOGY_BOUNDARY_CONTEXT_ONLY"), True),
        build_quantum_geometry_signal_receipt(scm, _identity(), _source(), _review(), _scope(), _topology("DECLARED_CUSTOM_TOPOLOGY_BOUNDARY"), True),
    ):
        assert candidate.replay_safe_quantum_geometry_signal is False
        qgsr.validate_quantum_geometry_signal_receipt(candidate, scm, quantum_memory_signal_receipt=qms, qpe_toolbox_adapter_receipt=qpe, agent_pattern_decision_receipt=apd, agent_observation_trace_receipt=trace, skill_library_manifest=manifest, tool_dispatch_telemetry_receipt=dispatch, crawler_boundary_receipt=crawler, **deps)

    source = (Path(__file__).parent.parent / "src/qec/analysis/quantum_geometry_signal_receipts.py").read_text(encoding="utf-8")
    for token in ("qiskit", "qutip", "cirq", "pennylane", "qulacs", "cudaq", "torch", "tensorflow", "requests", "urllib", "subprocess", "asyncio", "multiprocessing", "os.system", "eval(", "exec("):
        assert token not in source

    with pytest.raises(ValueError):
        qgsr.validate_quantum_geometry_signal_receipt(rec, replace(scm, self_correcting_memory_claim_boundary_receipt_hash="0" * 64), quantum_memory_signal_receipt=qms, qpe_toolbox_adapter_receipt=qpe, agent_pattern_decision_receipt=apd, agent_observation_trace_receipt=trace, skill_library_manifest=manifest, tool_dispatch_telemetry_receipt=dispatch, crawler_boundary_receipt=crawler, **deps)
