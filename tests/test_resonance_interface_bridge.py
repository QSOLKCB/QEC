from __future__ import annotations

import hashlib
import json

import pytest

from qec.analysis import resonance_interface_bridge as rib
from qec.analysis.e8_topology_projection_experiment import run_e8_topology_projection_experiment
from qec.analysis.fractal_field_invariant_mapper import map_fractal_field_invariants
from qec.analysis.phase_coherence_audit_layer import run_phase_coherence_audit
from qec.analysis.resonance_interface_bridge import build_resonance_interface_bridge
from qec.analysis.resonance_lock_diagnostic_kernel import run_resonance_lock_diagnostic


def _rehash_replay_identity(payload: dict[str, object]) -> dict[str, object]:
    data = dict(payload)
    data.pop("replay_identity", None)
    canonical = json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
    payload["replay_identity"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return payload


def _make_sources() -> tuple[object, object, object, object]:
    states = ("a", "a", "a", "b", "b", "a", "a", "a")
    phase_sequence = (0.0, 0.01, 0.02, 0.5, 0.51, 0.02, 0.01, 0.0)
    resonance = run_resonance_lock_diagnostic(state_sequence=states, drift_sequence=(0.0,) * len(states))
    phase = run_phase_coherence_audit(
        state_sequence=states,
        phase_sequence=phase_sequence,
        resonance_receipt=resonance,
    )
    topology = run_e8_topology_projection_experiment(
        state_sequence=states,
        phase_sequence=phase_sequence,
        resonance_receipt=resonance,
        phase_audit_receipt=phase,
    )
    fractal = map_fractal_field_invariants(states)
    return resonance, phase, topology, fractal


def test_determinism_same_input_same_bytes_hash() -> None:
    resonance, phase, topology, fractal = _make_sources()
    a = build_resonance_interface_bridge(
        resonance_receipt=resonance,
        phase_audit_receipt=phase,
        topology_projection_receipt=topology,
        fractal_invariant_receipt=fractal,
    )
    b = build_resonance_interface_bridge(
        resonance_receipt=resonance,
        phase_audit_receipt=phase,
        topology_projection_receipt=topology,
        fractal_invariant_receipt=fractal,
    )
    assert a.to_canonical_bytes() == b.to_canonical_bytes()
    assert a.stable_hash() == b.stable_hash()


def test_rejects_no_sources() -> None:
    with pytest.raises(ValueError, match="at least one source"):
        build_resonance_interface_bridge()


def test_source_kind_version_validation() -> None:
    resonance, phase, topology, fractal = _make_sources()

    bad_resonance = resonance.to_dict()
    bad_resonance["release_version"] = "v138.5.1"
    _rehash_replay_identity(bad_resonance)
    with pytest.raises(ValueError, match="release_version"):
        build_resonance_interface_bridge(resonance_receipt=bad_resonance)

    bad_phase = phase.to_dict()
    bad_phase["audit_kind"] = "other"
    _rehash_replay_identity(bad_phase)
    with pytest.raises(ValueError, match="audit_kind"):
        build_resonance_interface_bridge(phase_audit_receipt=bad_phase)

    bad_topology = topology.to_dict()
    bad_topology["experiment_kind"] = "other"
    _rehash_replay_identity(bad_topology)
    with pytest.raises(ValueError, match="experiment_kind"):
        build_resonance_interface_bridge(topology_projection_receipt=bad_topology)

    bad_fractal = fractal.to_dict()
    bad_fractal["experiment_kind"] = "other"
    with pytest.raises(ValueError, match="experiment_kind"):
        build_resonance_interface_bridge(fractal_invariant_receipt=bad_fractal)


def test_malformed_structure_rejection_and_tamper_regressions() -> None:
    resonance, _, topology, fractal = _make_sources()

    tampered = resonance.to_dict()
    tampered["resonance_classification"] = "tampered"
    with pytest.raises(ValueError, match="hash mismatch"):
        build_resonance_interface_bridge(resonance_receipt=tampered)

    malformed = topology.to_dict()
    malformed["ordered_coordinates"] = {"index": 1}
    _rehash_replay_identity(malformed)
    with pytest.raises(ValueError, match="must be a tuple"):
        build_resonance_interface_bridge(topology_projection_receipt=malformed)

    malformed_fractal = fractal.to_dict()
    malformed_fractal["ordered_scale_profiles"] = ({"scale_size": 2},)
    with pytest.raises(ValueError, match="missing field"):
        build_resonance_interface_bridge(fractal_invariant_receipt=malformed_fractal)


def test_trajectory_length_mismatch_rejected() -> None:
    resonance, phase, _, _ = _make_sources()
    short_states = ("a", "a", "b", "b")
    topology_short = run_e8_topology_projection_experiment(state_sequence=short_states)

    with pytest.raises(ValueError, match="trajectory_length"):
        build_resonance_interface_bridge(
            resonance_receipt=resonance,
            phase_audit_receipt=phase,
            topology_projection_receipt=topology_short,
        )


def test_source_ordering_determinism() -> None:
    resonance, phase, topology, fractal = _make_sources()
    receipt = build_resonance_interface_bridge(
        phase_audit_receipt=phase,
        fractal_invariant_receipt=fractal,
        resonance_receipt=resonance,
        topology_projection_receipt=topology,
    )
    assert tuple(item.source_name for item in receipt.ordered_source_summaries) == (
        "resonance",
        "phase",
        "topology",
        "fractal",
    )


def test_metric_bounds_and_presence_flags() -> None:
    resonance, phase, topology, fractal = _make_sources()
    receipt = build_resonance_interface_bridge(
        resonance_receipt=resonance,
        phase_audit_receipt=phase,
        topology_projection_receipt=topology,
        fractal_invariant_receipt=fractal,
    )
    for value in receipt.bounded_metrics.values():
        assert 0.0 <= float(value) <= 1.0
    assert receipt.source_presence_flags == {
        "resonance": True,
        "phase": True,
        "topology": True,
        "fractal": True,
    }


def test_strong_partial_weak_and_conflicted_classifications() -> None:
    resonance, phase, topology, fractal = _make_sources()
    strong = build_resonance_interface_bridge(
        resonance_receipt=resonance,
        phase_audit_receipt=phase,
        topology_projection_receipt=topology,
        fractal_invariant_receipt=fractal,
    )
    assert strong.interface_classification in {
        "strongly_unified_interface",
        "partially_unified_interface",
    }

    partial = build_resonance_interface_bridge(
        resonance_receipt=resonance,
        phase_audit_receipt=phase,
    )
    assert partial.interface_classification in {
        "partially_unified_interface",
        "weakly_supported_interface",
    }

    weak = build_resonance_interface_bridge(fractal_invariant_receipt=fractal)
    assert weak.interface_classification == "weakly_supported_interface"

    conflict_phase = run_phase_coherence_audit(
        state_sequence=("a", "a", "a", "b", "b", "a", "a", "a"),
        phase_sequence=(0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0),
        resonance_receipt=resonance,
    )
    conflicted = build_resonance_interface_bridge(
        resonance_receipt=resonance,
        phase_audit_receipt=conflict_phase,
    )
    assert conflicted.interface_classification in {"conflicted_interface", "weakly_supported_interface"}


def test_missing_sources_reduce_completeness_not_invalidity() -> None:
    resonance, _, _, _ = _make_sources()
    receipt = build_resonance_interface_bridge(resonance_receipt=resonance)
    assert receipt.bounded_metrics["interface_completeness_score"] == 0.25
    assert receipt.interface_classification == "weakly_supported_interface"


def test_immutability_enforcement() -> None:
    resonance, phase, _, _ = _make_sources()
    receipt = build_resonance_interface_bridge(
        resonance_receipt=resonance,
        phase_audit_receipt=phase,
    )
    with pytest.raises(TypeError):
        receipt.bounded_metrics["bounded_interface_confidence"] = 0.0  # type: ignore[index]
    with pytest.raises(TypeError):
        receipt.structure_summary["resonance_classification"] = "x"  # type: ignore[index]
    if receipt.structure_summary["strongest_lock"] is not None:
        with pytest.raises(TypeError):
            receipt.structure_summary["strongest_lock"]["lock_strength"] = 0.0  # type: ignore[index]


def test_strongest_lock_selected_by_lock_strength() -> None:
    resonance, _, _, _ = _make_sources()
    modified = resonance.to_dict()
    modified["ordered_lock_spans"] = (
        {"start_index": 0, "end_index": 2, "lock_strength": 0.25},
        {"start_index": 3, "end_index": 6, "lock_strength": 0.9},
    )
    _rehash_replay_identity(modified)
    receipt = build_resonance_interface_bridge(resonance_receipt=modified)
    assert receipt.structure_summary["strongest_lock"] == {
        "end_index": 6,
        "lock_strength": 0.9,
        "start_index": 3,
    }


def test_conflict_interpretation_reachable_with_contradictory_sources() -> None:
    resonance, phase, _, _ = _make_sources()
    weak_resonance = resonance.to_dict()
    weak_resonance["resonance_classification"] = "unknown_resonance_state"
    _rehash_replay_identity(weak_resonance)
    weak_phase = phase.to_dict()
    weak_phase["coherence_classification"] = "unknown_phase_state"
    weak_phase["resonance_source_identity"] = weak_resonance["replay_identity"]
    _rehash_replay_identity(weak_phase)

    receipt = build_resonance_interface_bridge(
        resonance_receipt=weak_resonance,
        phase_audit_receipt=weak_phase,
    )
    assert receipt.agreement_summary["source_agreement_interpretation"] == "cross_source_conflict_detected"
    assert receipt.interface_classification == "conflicted_interface"


def test_replay_hash_stability_and_call_order_independence() -> None:
    resonance, phase, topology, fractal = _make_sources()
    r1 = build_resonance_interface_bridge(
        resonance_receipt=resonance,
        phase_audit_receipt=phase,
        topology_projection_receipt=topology,
        fractal_invariant_receipt=fractal,
    )
    r2 = build_resonance_interface_bridge(
        fractal_invariant_receipt=fractal,
        topology_projection_receipt=topology,
        phase_audit_receipt=phase,
        resonance_receipt=resonance,
    )
    assert r1.to_canonical_json() == r2.to_canonical_json()
    assert r1.stable_hash() == r2.stable_hash()


def test_elimination_readiness_normalization_called_once_per_source(monkeypatch: pytest.MonkeyPatch) -> None:
    resonance, phase, topology, fractal = _make_sources()
    counts = {"res": 0, "phase": 0, "top": 0, "frac": 0, "agree": 0}

    orig_r = rib._normalize_resonance_source
    orig_p = rib._normalize_phase_audit_source
    orig_t = rib._normalize_topology_source
    orig_f = rib._normalize_fractal_source
    orig_a = rib._precompute_interface_agreement

    def wrap_r(x):
        counts["res"] += 1
        return orig_r(x)

    def wrap_p(x):
        counts["phase"] += 1
        return orig_p(x)

    def wrap_t(x):
        counts["top"] += 1
        return orig_t(x)

    def wrap_f(x):
        counts["frac"] += 1
        return orig_f(x)

    def wrap_a(x):
        counts["agree"] += 1
        return orig_a(x)

    monkeypatch.setattr(rib, "_normalize_resonance_source", wrap_r)
    monkeypatch.setattr(rib, "_normalize_phase_audit_source", wrap_p)
    monkeypatch.setattr(rib, "_normalize_topology_source", wrap_t)
    monkeypatch.setattr(rib, "_normalize_fractal_source", wrap_f)
    monkeypatch.setattr(rib, "_precompute_interface_agreement", wrap_a)

    receipt = build_resonance_interface_bridge(
        resonance_receipt=resonance,
        phase_audit_receipt=phase,
        topology_projection_receipt=topology,
        fractal_invariant_receipt=fractal,
    )
    assert receipt.stable_hash()
    assert counts == {"res": 1, "phase": 1, "top": 1, "frac": 1, "agree": 1}



def test_resonance_classification_validation_rejected_when_missing_empty_or_non_string() -> None:
    resonance, _, _, _ = _make_sources()

    missing = resonance.to_dict()
    missing.pop("resonance_classification", None)
    _rehash_replay_identity(missing)
    with pytest.raises(ValueError, match="resonance_classification must be a non-empty string"):
        build_resonance_interface_bridge(resonance_receipt=missing)

    empty = resonance.to_dict()
    empty["resonance_classification"] = ""
    _rehash_replay_identity(empty)
    with pytest.raises(ValueError, match="resonance_classification must be a non-empty string"):
        build_resonance_interface_bridge(resonance_receipt=empty)

    non_string = resonance.to_dict()
    non_string["resonance_classification"] = 123
    _rehash_replay_identity(non_string)
    with pytest.raises(ValueError, match="resonance_classification must be a non-empty string"):
        build_resonance_interface_bridge(resonance_receipt=non_string)

def test_invalid_metric_range_rejected() -> None:
    resonance, _, _, _ = _make_sources()
    bad = resonance.to_dict()
    bad["bounded_metrics"]["lock_strength_score"] = 2.0
    _rehash_replay_identity(bad)
    with pytest.raises(ValueError, match=r"must be in \[0,1\]"):
        build_resonance_interface_bridge(resonance_receipt=bad)
