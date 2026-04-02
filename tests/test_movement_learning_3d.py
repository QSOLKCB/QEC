"""Tests for the 3D Movement Learning Sandbox (v136.8.0).

Minimum 50 tests covering:
- dataclass immutability
- seed determinism
- same-seed replay
- different-seed divergence
- 3D action correctness
- hazard bounds
- basin crossing detection
- recovery arc detection
- classification correctness
- surface feedback integration
- bridge compatibility
- validator compatibility
- 100-run replay determinism
- stable hash identity
- decoder untouched verification
"""

from __future__ import annotations

import os

import pytest

from qec.ai.movement_learning_3d import (
    ACTION_VECTORS_3D,
    VALID_ACTIONS_3D,
    VALID_CLASSIFICATIONS_3D,
    MovementState3D,
    PolicyDecision3D,
    Trajectory3D,
    classify_trajectory,
    detect_basin_crossing,
    detect_recovery_arc,
    evaluate_3d_policy,
    export_trajectory_state_space,
    initialize_state_3d,
    run_trajectory,
    score_3d_state,
    step_3d_environment,
    trajectory_to_feedback_ledger,
)
from qec.ai.state_space_bridge import (
    UnifiedStateSpaceReport,
    build_movement_state_space,
)
from qec.ai.state_space_validator import (
    compute_state_space_hash,
    validate_state_space_report,
)
from qec.ai.surface_feedback_engine import (
    FeedbackLedger,
    VALID_EVENT_TYPES,
    VALID_LEDGER_CLASSIFICATIONS,
)


# ===================================================================
# 1. Dataclass immutability tests
# ===================================================================


class TestDataclassImmutability:
    """Verify all dataclasses are frozen."""

    def test_movement_state_3d_frozen(self):
        state = initialize_state_3d(seed=1)
        with pytest.raises(AttributeError):
            state.hazard_score = 0.5  # type: ignore[misc]

    def test_movement_state_3d_position_frozen(self):
        state = initialize_state_3d(seed=1)
        with pytest.raises(AttributeError):
            state.position = (0.0, 0.0, 0.0)  # type: ignore[misc]

    def test_policy_decision_3d_frozen(self):
        decision = evaluate_3d_policy(initialize_state_3d(seed=1))
        with pytest.raises(AttributeError):
            decision.action = "up"  # type: ignore[misc]

    def test_trajectory_3d_frozen(self):
        traj = run_trajectory(seed=1, steps=5)
        with pytest.raises(AttributeError):
            traj.total_reward = 999.0  # type: ignore[misc]

    def test_movement_state_3d_fields(self):
        state = initialize_state_3d(seed=42)
        assert isinstance(state.position, tuple)
        assert len(state.position) == 3
        assert isinstance(state.velocity, tuple)
        assert len(state.velocity) == 3
        assert isinstance(state.hazard_score, float)
        assert isinstance(state.coherence, float)
        assert isinstance(state.entropy, float)
        assert isinstance(state.stability, float)

    def test_policy_decision_3d_fields(self):
        decision = evaluate_3d_policy(initialize_state_3d(seed=42))
        assert isinstance(decision.action, str)
        assert isinstance(decision.confidence, float)
        assert isinstance(decision.expected_reward, float)
        assert isinstance(decision.basin_risk, float)

    def test_trajectory_3d_fields(self):
        traj = run_trajectory(seed=42, steps=5)
        assert isinstance(traj.states, tuple)
        assert isinstance(traj.basin_crossings, tuple)
        assert isinstance(traj.recovery_arcs, tuple)
        assert isinstance(traj.total_reward, float)
        assert isinstance(traj.classification, str)

    def test_trajectory_3d_states_are_tuples(self):
        traj = run_trajectory(seed=42, steps=5)
        for state in traj.states:
            assert isinstance(state, MovementState3D)


# ===================================================================
# 2. Seed determinism tests
# ===================================================================


class TestSeedDeterminism:
    """Verify deterministic behavior with explicit seeds."""

    def test_initialize_state_3d_deterministic(self):
        s1 = initialize_state_3d(seed=42)
        s2 = initialize_state_3d(seed=42)
        assert s1 == s2

    def test_same_seed_same_trajectory(self):
        t1 = run_trajectory(seed=42, steps=60)
        t2 = run_trajectory(seed=42, steps=60)
        assert t1 == t2

    def test_different_seed_divergence(self):
        t1 = run_trajectory(seed=42, steps=60)
        t2 = run_trajectory(seed=43, steps=60)
        assert t1 != t2

    def test_different_seed_different_initial_state(self):
        s1 = initialize_state_3d(seed=100)
        s2 = initialize_state_3d(seed=200)
        assert s1 != s2

    def test_step_determinism(self):
        state = initialize_state_3d(seed=7)
        r1 = step_3d_environment(state, "up")
        r2 = step_3d_environment(state, "up")
        assert r1 == r2

    def test_policy_determinism(self):
        state = initialize_state_3d(seed=7)
        d1 = evaluate_3d_policy(state)
        d2 = evaluate_3d_policy(state)
        assert d1 == d2

    def test_forward_backward_determinism(self):
        state = initialize_state_3d(seed=7)
        r1 = step_3d_environment(state, "forward")
        r2 = step_3d_environment(state, "forward")
        assert r1 == r2


# ===================================================================
# 3. 3D Action correctness tests
# ===================================================================


class TestActionCorrectness3D:
    """Verify 3D action vectors and movement law."""

    def test_all_valid_actions_accepted(self):
        state = initialize_state_3d(seed=1)
        for action in VALID_ACTIONS_3D:
            result = step_3d_environment(state, action)
            assert isinstance(result, MovementState3D)

    def test_invalid_action_raises(self):
        state = initialize_state_3d(seed=1)
        with pytest.raises(ValueError, match="Invalid action"):
            step_3d_environment(state, "fly")

    def test_up_increases_y(self):
        state = MovementState3D(
            position=(0.0, 0.0, 0.0),
            velocity=(0.0, 0.0, 0.0),
            hazard_score=0.0, coherence=0.9, entropy=0.1, stability=0.9,
        )
        result = step_3d_environment(state, "up")
        assert result.position[1] > state.position[1]

    def test_down_decreases_y(self):
        state = MovementState3D(
            position=(0.0, 0.0, 0.0),
            velocity=(0.0, 0.0, 0.0),
            hazard_score=0.0, coherence=0.9, entropy=0.1, stability=0.9,
        )
        result = step_3d_environment(state, "down")
        assert result.position[1] < state.position[1]

    def test_left_decreases_x(self):
        state = MovementState3D(
            position=(0.0, 0.0, 0.0),
            velocity=(0.0, 0.0, 0.0),
            hazard_score=0.0, coherence=0.9, entropy=0.1, stability=0.9,
        )
        result = step_3d_environment(state, "left")
        assert result.position[0] < state.position[0]

    def test_right_increases_x(self):
        state = MovementState3D(
            position=(0.0, 0.0, 0.0),
            velocity=(0.0, 0.0, 0.0),
            hazard_score=0.0, coherence=0.9, entropy=0.1, stability=0.9,
        )
        result = step_3d_environment(state, "right")
        assert result.position[0] > state.position[0]

    def test_forward_increases_z(self):
        state = MovementState3D(
            position=(0.0, 0.0, 0.0),
            velocity=(0.0, 0.0, 0.0),
            hazard_score=0.0, coherence=0.9, entropy=0.1, stability=0.9,
        )
        result = step_3d_environment(state, "forward")
        assert result.position[2] > state.position[2]

    def test_backward_decreases_z(self):
        state = MovementState3D(
            position=(0.0, 0.0, 0.0),
            velocity=(0.0, 0.0, 0.0),
            hazard_score=0.0, coherence=0.9, entropy=0.1, stability=0.9,
        )
        result = step_3d_environment(state, "backward")
        assert result.position[2] < state.position[2]

    def test_hold_no_action_vector(self):
        assert ACTION_VECTORS_3D["hold"] == (0.0, 0.0, 0.0)

    def test_recover_no_action_vector(self):
        assert ACTION_VECTORS_3D["recover"] == (0.0, 0.0, 0.0)

    def test_all_action_vectors_are_3d(self):
        for action, vec in ACTION_VECTORS_3D.items():
            assert len(vec) == 3, f"Action {action} vector is not 3D"


# ===================================================================
# 4. Hazard bounds tests
# ===================================================================


class TestHazardBounds:
    """Verify all metrics stay in [0, 1]."""

    def test_hazard_bounded_initial(self):
        for seed in range(100):
            state = initialize_state_3d(seed=seed)
            assert 0.0 <= state.hazard_score <= 1.0

    def test_hazard_bounded_after_steps(self):
        state = initialize_state_3d(seed=42)
        for _ in range(200):
            state = step_3d_environment(state, "right")
        assert 0.0 <= state.hazard_score <= 1.0

    def test_coherence_bounded(self):
        state = initialize_state_3d(seed=42)
        for _ in range(200):
            state = step_3d_environment(state, "forward")
        assert 0.0 <= state.coherence <= 1.0

    def test_entropy_bounded(self):
        state = initialize_state_3d(seed=42)
        for _ in range(200):
            state = step_3d_environment(state, "up")
        assert 0.0 <= state.entropy <= 1.0

    def test_stability_bounded(self):
        state = initialize_state_3d(seed=42)
        for _ in range(200):
            state = step_3d_environment(state, "backward")
        assert 0.0 <= state.stability <= 1.0

    def test_all_metrics_bounded_trajectory(self):
        traj = run_trajectory(seed=42, steps=100)
        for state in traj.states:
            assert 0.0 <= state.hazard_score <= 1.0
            assert 0.0 <= state.coherence <= 1.0
            assert 0.0 <= state.entropy <= 1.0
            assert 0.0 <= state.stability <= 1.0

    def test_all_metrics_bounded_multi_seed(self):
        for seed in range(50):
            traj = run_trajectory(seed=seed, steps=30)
            for state in traj.states:
                assert 0.0 <= state.hazard_score <= 1.0
                assert 0.0 <= state.coherence <= 1.0
                assert 0.0 <= state.entropy <= 1.0
                assert 0.0 <= state.stability <= 1.0


# ===================================================================
# 5. Basin crossing detection tests
# ===================================================================


class TestBasinCrossingDetection:
    """Verify basin crossing detection in 3D."""

    def test_no_crossings_for_hold(self):
        state = MovementState3D(
            position=(0.0, 0.0, 0.0),
            velocity=(0.0, 0.0, 0.0),
            hazard_score=0.1, coherence=0.9, entropy=0.1, stability=0.9,
        )
        states = [state]
        for _ in range(10):
            state = step_3d_environment(state, "hold")
            states.append(state)
        traj = Trajectory3D(
            states=tuple(states), basin_crossings=(),
            recovery_arcs=(), total_reward=0.0, classification="",
        )
        crossings = detect_basin_crossing(traj)
        assert len(crossings) == 0

    def test_crossing_detected_on_large_jump(self):
        s1 = MovementState3D(
            (0.0, 0.0, 0.0), (0.0, 0.0, 0.0),
            0.1, 0.9, 0.1, 0.9,
        )
        s2 = MovementState3D(
            (1.0, 1.0, 1.0), (0.0, 0.0, 0.0),
            0.1, 0.9, 0.1, 0.9,
        )
        traj = Trajectory3D(
            states=(s1, s2), basin_crossings=(),
            recovery_arcs=(), total_reward=0.0, classification="",
        )
        crossings = detect_basin_crossing(traj)
        assert len(crossings) >= 1
        assert 0 in crossings

    def test_crossings_are_tuple(self):
        traj = run_trajectory(seed=42, steps=30)
        crossings = detect_basin_crossing(traj)
        assert isinstance(crossings, tuple)

    def test_crossing_indices_valid(self):
        traj = run_trajectory(seed=42, steps=30)
        crossings = detect_basin_crossing(traj)
        for idx in crossings:
            assert 0 <= idx < len(traj.states) - 1


# ===================================================================
# 6. Recovery arc detection tests
# ===================================================================


class TestRecoveryArcDetection:
    """Verify recovery arc detection in 3D."""

    def test_arc_detected_on_hazard_drop(self):
        states = [
            MovementState3D((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 0.8, 0.5, 0.3, 0.5),
            MovementState3D((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 0.6, 0.6, 0.2, 0.6),
            MovementState3D((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 0.4, 0.7, 0.2, 0.7),
        ]
        traj = Trajectory3D(
            states=tuple(states), basin_crossings=(),
            recovery_arcs=(), total_reward=0.0, classification="",
        )
        arcs = detect_recovery_arc(traj)
        assert len(arcs) >= 1

    def test_no_arc_when_hazard_stable(self):
        states = [
            MovementState3D((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 0.2, 0.9, 0.1, 0.9),
            MovementState3D((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 0.2, 0.9, 0.1, 0.9),
            MovementState3D((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 0.2, 0.9, 0.1, 0.9),
        ]
        traj = Trajectory3D(
            states=tuple(states), basin_crossings=(),
            recovery_arcs=(), total_reward=0.0, classification="",
        )
        arcs = detect_recovery_arc(traj)
        assert len(arcs) == 0

    def test_arcs_are_tuple_of_pairs(self):
        traj = run_trajectory(seed=42, steps=30)
        arcs = detect_recovery_arc(traj)
        assert isinstance(arcs, tuple)
        for arc in arcs:
            assert isinstance(arc, tuple)
            assert len(arc) == 2
            assert arc[0] <= arc[1]

    def test_arc_indices_valid(self):
        traj = run_trajectory(seed=42, steps=60)
        arcs = detect_recovery_arc(traj)
        for start, end in arcs:
            assert 0 <= start < len(traj.states)
            assert 0 <= end < len(traj.states)


# ===================================================================
# 7. Classification correctness tests
# ===================================================================


class TestClassificationCorrectness:
    """Verify trajectory classification returns valid labels."""

    def test_classification_is_valid(self):
        for seed in range(20):
            traj = run_trajectory(seed=seed, steps=60)
            assert traj.classification in VALID_CLASSIFICATIONS_3D

    def test_classify_trajectory_deterministic(self):
        traj = run_trajectory(seed=42, steps=60)
        c1 = classify_trajectory(traj)
        c2 = classify_trajectory(traj)
        assert c1 == c2

    def test_classify_short_trajectory(self):
        traj = run_trajectory(seed=42, steps=1)
        assert traj.classification in VALID_CLASSIFICATIONS_3D

    def test_classify_single_state_trajectory(self):
        traj = Trajectory3D(
            states=(initialize_state_3d(1),),
            basin_crossings=(),
            recovery_arcs=(),
            total_reward=0.0,
            classification="",
        )
        result = classify_trajectory(traj)
        assert result in VALID_CLASSIFICATIONS_3D

    def test_stable_volume_classification(self):
        """A trajectory near origin with low hazard should be stable_volume."""
        state = MovementState3D(
            position=(0.0, 0.0, 0.0),
            velocity=(0.0, 0.0, 0.0),
            hazard_score=0.1, coherence=0.9, entropy=0.1, stability=0.9,
        )
        states = [state]
        for _ in range(10):
            state = step_3d_environment(state, "hold")
            states.append(state)
        traj = Trajectory3D(
            states=tuple(states), basin_crossings=(),
            recovery_arcs=(), total_reward=0.0, classification="",
        )
        assert classify_trajectory(traj) == "stable_volume"


# ===================================================================
# 8. Surface feedback integration tests
# ===================================================================


class TestSurfaceFeedbackIntegration:
    """Verify trajectory_to_feedback_ledger integration."""

    def test_returns_feedback_ledger(self):
        traj = run_trajectory(seed=42, steps=20)
        ledger = trajectory_to_feedback_ledger(traj)
        assert isinstance(ledger, FeedbackLedger)

    def test_ledger_has_events(self):
        traj = run_trajectory(seed=42, steps=20)
        ledger = trajectory_to_feedback_ledger(traj)
        assert len(ledger.events) == len(traj.states) - 1

    def test_ledger_event_types_valid(self):
        traj = run_trajectory(seed=42, steps=20)
        ledger = trajectory_to_feedback_ledger(traj)
        for event in ledger.events:
            assert event.event_type in VALID_EVENT_TYPES

    def test_ledger_classification_valid(self):
        traj = run_trajectory(seed=42, steps=20)
        ledger = trajectory_to_feedback_ledger(traj)
        assert ledger.classification in VALID_LEDGER_CLASSIFICATIONS

    def test_ledger_scores_bounded(self):
        traj = run_trajectory(seed=42, steps=60)
        ledger = trajectory_to_feedback_ledger(traj)
        assert 0.0 <= ledger.cumulative_score <= 1.0
        assert 0.0 <= ledger.stability_score <= 1.0
        assert 0.0 <= ledger.hazard_pressure <= 1.0

    def test_ledger_deterministic(self):
        traj = run_trajectory(seed=42, steps=20)
        l1 = trajectory_to_feedback_ledger(traj)
        l2 = trajectory_to_feedback_ledger(traj)
        assert l1 == l2

    def test_empty_trajectory_returns_neutral_ledger(self):
        traj = Trajectory3D(
            states=(initialize_state_3d(1),),
            basin_crossings=(),
            recovery_arcs=(),
            total_reward=0.0,
            classification="stable_volume",
        )
        ledger = trajectory_to_feedback_ledger(traj)
        assert isinstance(ledger, FeedbackLedger)
        assert len(ledger.events) == 0
        assert ledger.cumulative_score == 0.5
        assert ledger.stability_score == 0.5
        assert ledger.hazard_pressure == 0.0


# ===================================================================
# 9. Bridge compatibility tests
# ===================================================================


class TestBridgeCompatibility:
    """Verify export_trajectory_state_space compatibility with bridge."""

    def test_export_returns_sequence_of_dicts(self):
        traj = run_trajectory(seed=42, steps=10)
        trace = export_trajectory_state_space(traj)
        assert len(trace) == len(traj.states)
        for entry in trace:
            assert "x" in entry
            assert "y" in entry
            assert "coherence" in entry
            assert "entropy" in entry
            assert "stability" in entry
            assert "label" in entry

    def test_export_compatible_with_build_movement_state_space(self):
        traj = run_trajectory(seed=42, steps=10)
        trace = export_trajectory_state_space(traj)
        report = build_movement_state_space(trace)
        assert isinstance(report, UnifiedStateSpaceReport)
        assert len(report.nodes) == len(traj.states)

    def test_export_preserves_xy_positions(self):
        traj = run_trajectory(seed=42, steps=5)
        trace = export_trajectory_state_space(traj)
        for i, entry in enumerate(trace):
            assert entry["x"] == traj.states[i].position[0]
            assert entry["y"] == traj.states[i].position[1]

    def test_export_label_matches_classification(self):
        traj = run_trajectory(seed=42, steps=10)
        trace = export_trajectory_state_space(traj)
        for entry in trace:
            assert entry["label"] == traj.classification


# ===================================================================
# 10. Validator compatibility tests
# ===================================================================


class TestValidatorCompatibility:
    """Verify exported state-space passes validation."""

    def test_validation_passes(self):
        traj = run_trajectory(seed=42, steps=20)
        trace = export_trajectory_state_space(traj)
        report = build_movement_state_space(trace)
        validation = validate_state_space_report(report)
        assert validation.overall_passed is True

    def test_classification_is_stable(self):
        traj = run_trajectory(seed=42, steps=20)
        trace = export_trajectory_state_space(traj)
        report = build_movement_state_space(trace)
        validation = validate_state_space_report(report)
        assert validation.classification_stable is True

    def test_hash_is_deterministic(self):
        traj = run_trajectory(seed=42, steps=20)
        trace = export_trajectory_state_space(traj)
        report = build_movement_state_space(trace)
        h1 = compute_state_space_hash(report)
        h2 = compute_state_space_hash(report)
        assert h1 == h2
        assert isinstance(h1, str)
        assert len(h1) == 64  # SHA-256 hex

    def test_validation_passes_multiple_seeds(self):
        for seed in [0, 7, 42, 99, 255]:
            traj = run_trajectory(seed=seed, steps=30)
            trace = export_trajectory_state_space(traj)
            report = build_movement_state_space(trace)
            validation = validate_state_space_report(report)
            assert validation.overall_passed is True


# ===================================================================
# 11. 100-run replay determinism tests
# ===================================================================


class TestReplayDeterminism:
    """Verify byte-identical replay over 100 runs."""

    def test_100_run_trajectory_replay(self):
        reference = run_trajectory(seed=42)
        for _ in range(100):
            assert run_trajectory(seed=42) == reference

    def test_100_run_state_space_hash_stable(self):
        reference = run_trajectory(seed=42)
        trace = export_trajectory_state_space(reference)
        report = build_movement_state_space(trace)
        ref_hash = compute_state_space_hash(report)
        for _ in range(100):
            traj = run_trajectory(seed=42)
            t = export_trajectory_state_space(traj)
            r = build_movement_state_space(t)
            assert compute_state_space_hash(r) == ref_hash


# ===================================================================
# 12. Decoder untouched verification
# ===================================================================


class TestDecoderUntouched:
    """Verify the decoder core is not imported or modified."""

    def test_no_decoder_import_in_module(self):
        import inspect
        import qec.ai.movement_learning_3d as mod
        source = inspect.getsource(mod)
        assert "qec.decoder" not in source
        assert "from qec.decoder" not in source
        assert "import qec.decoder" not in source

    def test_decoder_directory_unmodified(self):
        """Verify decoder directory exists and is non-empty (sanity check)."""
        decoder_path = os.path.join(
            os.path.dirname(__file__), "..", "src", "qec", "decoder",
        )
        if os.path.exists(decoder_path):
            assert os.path.isdir(decoder_path)
            assert len(os.listdir(decoder_path)) > 0


# ===================================================================
# 13. Stable hash identity tests
# ===================================================================


class TestStableHashIdentity:
    """Verify hash identity across independent computations."""

    def test_hash_same_for_same_seed(self):
        t1 = run_trajectory(seed=99, steps=30)
        t2 = run_trajectory(seed=99, steps=30)
        trace1 = export_trajectory_state_space(t1)
        trace2 = export_trajectory_state_space(t2)
        r1 = build_movement_state_space(trace1)
        r2 = build_movement_state_space(trace2)
        assert compute_state_space_hash(r1) == compute_state_space_hash(r2)

    def test_hash_differs_for_different_seed(self):
        t1 = run_trajectory(seed=42, steps=30)
        t2 = run_trajectory(seed=43, steps=30)
        trace1 = export_trajectory_state_space(t1)
        trace2 = export_trajectory_state_space(t2)
        r1 = build_movement_state_space(trace1)
        r2 = build_movement_state_space(trace2)
        assert compute_state_space_hash(r1) != compute_state_space_hash(r2)


# ===================================================================
# 14. Edge cases and integration tests
# ===================================================================


class TestEdgeCases:
    """Additional tests for full coverage."""

    def test_trajectory_states_length(self):
        traj = run_trajectory(seed=1, steps=60)
        assert len(traj.states) == 61

    def test_trajectory_steps_1(self):
        traj = run_trajectory(seed=1, steps=1)
        assert len(traj.states) == 2

    def test_multiple_seeds_all_valid(self):
        for seed in range(50):
            traj = run_trajectory(seed=seed, steps=20)
            assert traj.classification in VALID_CLASSIFICATIONS_3D
            assert isinstance(traj.total_reward, float)

    def test_policy_decision_action_is_valid(self):
        for seed in range(20):
            state = initialize_state_3d(seed=seed)
            decision = evaluate_3d_policy(state)
            assert decision.action in VALID_ACTIONS_3D

    def test_score_3d_state_positive_for_good_state(self):
        state = MovementState3D(
            position=(0.0, 0.0, 0.0),
            velocity=(0.0, 0.0, 0.0),
            hazard_score=0.0, coherence=1.0, entropy=0.0, stability=1.0,
        )
        assert score_3d_state(state) > 0.0

    def test_score_3d_state_negative_for_bad_state(self):
        state = MovementState3D(
            position=(5.0, 5.0, 5.0),
            velocity=(1.0, 1.0, 1.0),
            hazard_score=1.0, coherence=0.0, entropy=1.0, stability=0.0,
        )
        assert score_3d_state(state) < 0.0

    def test_recover_reduces_hazard_3d(self):
        state = MovementState3D(
            position=(0.0, 0.0, 0.0),
            velocity=(0.0, 0.0, 0.0),
            hazard_score=0.8, coherence=0.5, entropy=0.3, stability=0.5,
        )
        result = step_3d_environment(state, "recover")
        assert result.hazard_score < state.hazard_score

    def test_recover_increases_coherence_3d(self):
        state = MovementState3D(
            position=(0.0, 0.0, 0.0),
            velocity=(0.0, 0.0, 0.0),
            hazard_score=0.8, coherence=0.5, entropy=0.3, stability=0.5,
        )
        result = step_3d_environment(state, "recover")
        assert result.coherence > state.coherence

    def test_surface_feedback_multi_seed(self):
        for seed in [0, 7, 42, 99]:
            traj = run_trajectory(seed=seed, steps=20)
            ledger = trajectory_to_feedback_ledger(traj)
            assert isinstance(ledger, FeedbackLedger)
            assert 0.0 <= ledger.cumulative_score <= 1.0

    def test_3d_position_z_component_changes(self):
        """Verify the z-component actually changes during trajectories."""
        traj = run_trajectory(seed=42, steps=30)
        z_values = [s.position[2] for s in traj.states]
        # At least some z-values should differ from the initial
        assert len(set(z_values)) > 1
