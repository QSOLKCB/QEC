from __future__ import annotations

import hashlib
import math
import json

import pytest

from qec.analysis.e8_topology_projection_experiment import (
    E8TopologyProjectionPolicy,
    run_e8_topology_projection_experiment,
)
from qec.analysis.phase_coherence_audit_layer import run_phase_coherence_audit
from qec.analysis.resonance_lock_diagnostic_kernel import run_resonance_lock_diagnostic


def _rehash_receipt(payload: dict[str, object]) -> dict[str, object]:
    data = dict(payload)
    data.pop("replay_identity", None)
    canonical = json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
    payload["replay_identity"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return payload


def test_determinism_same_input_same_bytes_and_hash() -> None:
    kwargs = {
        "state_sequence": ("A", "A", "A", "B", "B", "A"),
        "phase_sequence": (0.01, 0.02, 0.02, 0.5, 0.52, 0.02),
    }
    a = run_e8_topology_projection_experiment(**kwargs)
    b = run_e8_topology_projection_experiment(**kwargs)
    assert a.to_canonical_bytes() == b.to_canonical_bytes()
    assert a.stable_hash() == b.stable_hash()


def test_material_input_change_changes_receipt_or_hash() -> None:
    a = run_e8_topology_projection_experiment(state_sequence=(1, 1, 2, 2, 1), phase_sequence=(0.0, 0.0, 0.1, 0.1, 0.0))
    b = run_e8_topology_projection_experiment(state_sequence=(1, 2, 3, 4, 5), phase_sequence=(0.0, 0.9, -0.9, 0.9, -0.9))
    assert a.to_canonical_bytes() != b.to_canonical_bytes()
    assert a.stable_hash() != b.stable_hash()


def test_validation_errors() -> None:
    with pytest.raises(ValueError, match="state_sequence must be non-empty"):
        run_e8_topology_projection_experiment(state_sequence=())

    with pytest.raises(ValueError, match="phase_sequence length"):
        run_e8_topology_projection_experiment(state_sequence=("x", "y"), phase_sequence=(0.1,))

    with pytest.raises(ValueError, match="must be finite"):
        run_e8_topology_projection_experiment(state_sequence=("x", "y"), phase_sequence=(0.1, math.inf))

    with pytest.raises(ValueError, match="policy concentrated_threshold"):
        run_e8_topology_projection_experiment(
            state_sequence=("x", "x"),
            policy=E8TopologyProjectionPolicy(concentrated_threshold=1.2),
        )


def test_malformed_and_wrong_version_sources_rejected() -> None:
    with pytest.raises(ValueError, match="missing field 'replay_identity'"):
        run_e8_topology_projection_experiment(
            state_sequence=("a", "a"),
            resonance_receipt={"release_version": "v138.5.0", "diagnostic_kind": "resonance_lock_diagnostic_kernel"},
        )

    resonance = run_resonance_lock_diagnostic(state_sequence=("a", "a"), drift_sequence=(0.0, 0.0)).to_dict()
    resonance["release_version"] = "v138.4.9"
    _rehash_receipt(resonance)
    with pytest.raises(ValueError, match="release_version"):
        run_e8_topology_projection_experiment(state_sequence=("a", "a"), resonance_receipt=resonance)

    phase = run_phase_coherence_audit(state_sequence=("a", "a"), phase_sequence=(0.0, 0.0)).to_dict()
    phase["audit_kind"] = "not_phase"
    _rehash_receipt(phase)
    with pytest.raises(ValueError, match="audit_kind"):
        run_e8_topology_projection_experiment(state_sequence=("a", "a"), phase_sequence=(0.0, 0.0), phase_audit_receipt=phase)


def test_projection_coordinates_shape_and_bounds() -> None:
    receipt = run_e8_topology_projection_experiment(
        state_sequence=("a", "a", "b", "b", "a", "a"),
        phase_sequence=(0.0, 0.01, 0.02, 0.03, 0.02, 0.01),
    )
    assert len(receipt.ordered_coordinates) == 8
    assert tuple(c.index for c in receipt.ordered_coordinates) == (1, 2, 3, 4, 5, 6, 7, 8)
    for coordinate in receipt.ordered_coordinates:
        assert 0.0 <= coordinate.value <= 1.0


def test_dominant_coordinate_tie_break_is_deterministic() -> None:
    receipt = run_e8_topology_projection_experiment(state_sequence=(1, 2))
    # c8 saturates for this sparse input and remains deterministic.
    max_value = max(c.value for c in receipt.ordered_coordinates)
    first_max = next(c.label for c in receipt.ordered_coordinates if c.value == max_value)
    assert first_max == "c8_balance_axis"


def test_recurrence_and_phase_inputs_strengthen_expected_axes() -> None:
    recurrence_heavy = run_e8_topology_projection_experiment(
        state_sequence=("x", "x", "x", "x", "y", "x", "x"),
        resonance_receipt=run_resonance_lock_diagnostic(
            state_sequence=("x", "x", "x", "x", "y", "x", "x"),
            drift_sequence=(0.0, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0),
        ),
    )
    c1 = recurrence_heavy.ordered_coordinates[0].value
    c2 = recurrence_heavy.ordered_coordinates[1].value
    assert c1 >= 0.6
    assert c2 >= 0.4

    phase_stable = run_e8_topology_projection_experiment(
        state_sequence=(0, 1, 2, 3, 4, 5),
        phase_sequence=(0.0, 0.01, 0.01, 0.02, 0.02, 0.03),
    )
    assert phase_stable.ordered_coordinates[3].value >= 0.9


def test_dispersed_input_yields_cloud_like_classification() -> None:
    dispersed = run_e8_topology_projection_experiment(
        state_sequence=(0, 1, 2, 3, 4, 5, 6, 7),
        phase_sequence=(0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0),
    )
    assert dispersed.topology_classification == "dispersed_topology_cloud"


def test_source_agreement_behavior() -> None:
    states = ("a", "a", "a", "b", "b", "b")
    resonance = run_resonance_lock_diagnostic(state_sequence=states, drift_sequence=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0))

    phase_supportive = run_phase_coherence_audit(
        state_sequence=states,
        phase_sequence=(0.0, 0.01, 0.01, 0.02, 0.02, 0.03),
        resonance_receipt=resonance,
    )
    phase_conflicting = run_phase_coherence_audit(
        state_sequence=states,
        phase_sequence=(0.0, 1.0, 0.0, 1.0, 0.0, 1.0),
        resonance_receipt=resonance,
    )

    resonance_only = run_e8_topology_projection_experiment(state_sequence=states, resonance_receipt=resonance)
    phase_only = run_e8_topology_projection_experiment(state_sequence=states, phase_audit_receipt=phase_supportive)
    joint = run_e8_topology_projection_experiment(
        state_sequence=states,
        resonance_receipt=resonance,
        phase_audit_receipt=phase_supportive,
    )
    conflict = run_e8_topology_projection_experiment(
        state_sequence=states,
        resonance_receipt=resonance,
        phase_audit_receipt=phase_conflicting,
    )

    assert resonance_only.resonance_source_identity is not None
    assert phase_only.phase_audit_source_identity is not None
    assert joint.bounded_metrics["cross_source_stability_score"] > conflict.bounded_metrics["cross_source_stability_score"]


def test_classification_behavior_for_concentrated_and_balanced() -> None:
    concentrated = run_e8_topology_projection_experiment(
        state_sequence=("z", "z", "z", "z", "z", "z", "a"),
        resonance_receipt=run_resonance_lock_diagnostic(
            state_sequence=("z", "z", "z", "z", "z", "z", "a"),
            drift_sequence=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2),
        ),
    )
    assert concentrated.topology_classification in {"concentrated_topology_cluster", "lock_dominant_projection"}

    balanced = run_e8_topology_projection_experiment(
        state_sequence=("a", "a", "b", "b", "a", "b"),
        phase_sequence=("x", "x", "y", "y", "x", "x"),
        policy=E8TopologyProjectionPolicy(concentrated_threshold=1.0, balanced_threshold=0.5, dispersed_threshold=0.99),
    )
    assert balanced.topology_classification == "balanced_topology_field"


def test_bounded_metrics_and_invariants_and_immutability() -> None:
    receipt = run_e8_topology_projection_experiment(
        state_sequence=(1, 1, 2, 2, 1, 1),
        phase_sequence=(0.0, 0.1, 0.0, 0.1, 0.0, 0.1),
    )
    for value in receipt.bounded_metrics.values():
        assert 0.0 <= float(value) <= 1.0
    assert receipt.advisory_only is True
    assert receipt.decoder_core_modified is False

    with pytest.raises(TypeError):
        receipt.bounded_metrics["bounded_projection_confidence"] = 0.0  # type: ignore[index]
    with pytest.raises(TypeError):
        receipt.feature_profile.feature_values["trajectory_recurrence_score"] = 0.0  # type: ignore[index]


def test_canonical_json_hash_stability_and_source_binding_regression() -> None:
    states = ("a", "a", "b", "b")
    resonance_a = run_resonance_lock_diagnostic(state_sequence=states, drift_sequence=(0.0, 0.0, 0.0, 0.0))
    resonance_b = run_resonance_lock_diagnostic(state_sequence=states, drift_sequence=(0.8, 0.8, 0.8, 0.8))

    receipt_a = run_e8_topology_projection_experiment(state_sequence=states, resonance_receipt=resonance_a)
    receipt_b = run_e8_topology_projection_experiment(state_sequence=states, resonance_receipt=resonance_a)
    receipt_c = run_e8_topology_projection_experiment(state_sequence=states, resonance_receipt=resonance_b)

    assert receipt_a.to_canonical_json() == receipt_b.to_canonical_json()
    assert receipt_a.stable_hash() == receipt_b.stable_hash()
    assert receipt_a.stable_hash() != receipt_c.stable_hash()

    tampered = resonance_a.to_dict()
    tampered["bounded_metrics"]["lock_strength_score"] = 0.123
    with pytest.raises(ValueError, match="hash mismatch"):
        run_e8_topology_projection_experiment(state_sequence=states, resonance_receipt=tampered)
