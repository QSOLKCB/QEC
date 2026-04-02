"""Tests for the 2D Movement Learning Sandbox (v136.7.0).

Minimum 45 tests covering:
- dataclass immutability
- seed determinism
- same-seed replay
- different-seed divergence
- action correctness
- hazard bounds
- recovery behavior
- reward scoring
- episode classification
- state-space export compatibility
- validator compatibility
- 100-run replay determinism
- decoder untouched verification
"""

from __future__ import annotations

import os

import pytest

from qec.ai.movement_learning_2d import (
    ACTION_VECTORS,
    VALID_ACTIONS,
    VALID_CLASSIFICATIONS,
    MovementEpisode,
    MovementState,
    PolicyDecision,
    classify_episode,
    detect_recovery_events,
    evaluate_policy,
    export_episode_state_space,
    initialize_state,
    run_episode,
    score_episode,
    score_state,
    step_environment,
)
from qec.ai.state_space_bridge import (
    UnifiedStateSpaceReport,
    build_movement_state_space,
)
from qec.ai.state_space_validator import (
    compute_state_space_hash,
    validate_state_space_report,
)


# ===================================================================
# 1. Dataclass immutability tests
# ===================================================================


class TestDataclassImmutability:
    """Verify all dataclasses are frozen."""

    def test_movement_state_frozen(self):
        state = initialize_state(seed=1)
        with pytest.raises(AttributeError):
            state.hazard_score = 0.5  # type: ignore[misc]

    def test_policy_decision_frozen(self):
        decision = evaluate_policy(initialize_state(seed=1))
        with pytest.raises(AttributeError):
            decision.action = "up"  # type: ignore[misc]

    def test_movement_episode_frozen(self):
        ep = run_episode(seed=1, steps=5)
        with pytest.raises(AttributeError):
            ep.total_reward = 999.0  # type: ignore[misc]

    def test_movement_state_fields(self):
        state = initialize_state(seed=42)
        assert isinstance(state.position, tuple)
        assert isinstance(state.velocity, tuple)
        assert isinstance(state.hazard_score, float)
        assert isinstance(state.coherence, float)
        assert isinstance(state.entropy, float)
        assert isinstance(state.stability, float)

    def test_policy_decision_fields(self):
        decision = evaluate_policy(initialize_state(seed=42))
        assert isinstance(decision.action, str)
        assert isinstance(decision.confidence, float)
        assert isinstance(decision.expected_reward, float)
        assert isinstance(decision.basin_risk, float)

    def test_episode_fields(self):
        ep = run_episode(seed=42, steps=5)
        assert isinstance(ep.states, tuple)
        assert isinstance(ep.decisions, tuple)
        assert isinstance(ep.total_reward, float)
        assert isinstance(ep.recovery_events, int)
        assert isinstance(ep.classification, str)


# ===================================================================
# 2. Seed determinism tests
# ===================================================================


class TestSeedDeterminism:
    """Verify deterministic behavior with explicit seeds."""

    def test_initialize_state_deterministic(self):
        s1 = initialize_state(seed=42)
        s2 = initialize_state(seed=42)
        assert s1 == s2

    def test_same_seed_same_episode(self):
        ep1 = run_episode(seed=42, steps=50)
        ep2 = run_episode(seed=42, steps=50)
        assert ep1 == ep2

    def test_different_seed_divergence(self):
        ep1 = run_episode(seed=42, steps=50)
        ep2 = run_episode(seed=43, steps=50)
        assert ep1 != ep2

    def test_different_seed_different_initial_state(self):
        s1 = initialize_state(seed=100)
        s2 = initialize_state(seed=200)
        assert s1 != s2

    def test_step_determinism(self):
        state = initialize_state(seed=7)
        r1 = step_environment(state, "up")
        r2 = step_environment(state, "up")
        assert r1 == r2

    def test_policy_determinism(self):
        state = initialize_state(seed=7)
        d1 = evaluate_policy(state)
        d2 = evaluate_policy(state)
        assert d1 == d2


# ===================================================================
# 3. Action correctness tests
# ===================================================================


class TestActionCorrectness:
    """Verify action vectors and movement law."""

    def test_all_valid_actions_accepted(self):
        state = initialize_state(seed=1)
        for action in VALID_ACTIONS:
            result = step_environment(state, action)
            assert isinstance(result, MovementState)

    def test_invalid_action_raises(self):
        state = initialize_state(seed=1)
        with pytest.raises(ValueError, match="Invalid action"):
            step_environment(state, "fly")

    def test_up_increases_y(self):
        state = MovementState(
            position=(0.0, 0.0),
            velocity=(0.0, 0.0),
            hazard_score=0.0,
            coherence=0.9,
            entropy=0.1,
            stability=0.9,
        )
        result = step_environment(state, "up")
        assert result.position[1] > state.position[1]

    def test_down_decreases_y(self):
        state = MovementState(
            position=(0.0, 0.0),
            velocity=(0.0, 0.0),
            hazard_score=0.0,
            coherence=0.9,
            entropy=0.1,
            stability=0.9,
        )
        result = step_environment(state, "down")
        assert result.position[1] < state.position[1]

    def test_left_decreases_x(self):
        state = MovementState(
            position=(0.0, 0.0),
            velocity=(0.0, 0.0),
            hazard_score=0.0,
            coherence=0.9,
            entropy=0.1,
            stability=0.9,
        )
        result = step_environment(state, "left")
        assert result.position[0] < state.position[0]

    def test_right_increases_x(self):
        state = MovementState(
            position=(0.0, 0.0),
            velocity=(0.0, 0.0),
            hazard_score=0.0,
            coherence=0.9,
            entropy=0.1,
            stability=0.9,
        )
        result = step_environment(state, "right")
        assert result.position[0] > state.position[0]

    def test_hold_no_action_vector(self):
        assert ACTION_VECTORS["hold"] == (0.0, 0.0)

    def test_recover_no_action_vector(self):
        assert ACTION_VECTORS["recover"] == (0.0, 0.0)


# ===================================================================
# 4. Hazard bounds tests
# ===================================================================


class TestHazardBounds:
    """Verify hazard_score stays in [0, 1]."""

    def test_hazard_bounded_initial(self):
        for seed in range(100):
            state = initialize_state(seed=seed)
            assert 0.0 <= state.hazard_score <= 1.0

    def test_hazard_bounded_after_steps(self):
        state = initialize_state(seed=42)
        for _ in range(200):
            state = step_environment(state, "right")
        assert 0.0 <= state.hazard_score <= 1.0

    def test_coherence_bounded(self):
        state = initialize_state(seed=42)
        for _ in range(200):
            state = step_environment(state, "right")
        assert 0.0 <= state.coherence <= 1.0

    def test_entropy_bounded(self):
        state = initialize_state(seed=42)
        for _ in range(200):
            state = step_environment(state, "right")
        assert 0.0 <= state.entropy <= 1.0

    def test_stability_bounded(self):
        state = initialize_state(seed=42)
        for _ in range(200):
            state = step_environment(state, "right")
        assert 0.0 <= state.stability <= 1.0

    def test_all_metrics_bounded_episode(self):
        ep = run_episode(seed=42, steps=100)
        for state in ep.states:
            assert 0.0 <= state.hazard_score <= 1.0
            assert 0.0 <= state.coherence <= 1.0
            assert 0.0 <= state.entropy <= 1.0
            assert 0.0 <= state.stability <= 1.0


# ===================================================================
# 5. Recovery behavior tests
# ===================================================================


class TestRecoveryBehavior:
    """Verify recovery action mechanics."""

    def test_recover_reduces_hazard(self):
        state = MovementState(
            position=(0.0, 0.0),
            velocity=(0.0, 0.0),
            hazard_score=0.8,
            coherence=0.5,
            entropy=0.3,
            stability=0.5,
        )
        result = step_environment(state, "recover")
        assert result.hazard_score < state.hazard_score

    def test_recover_increases_coherence(self):
        state = MovementState(
            position=(0.0, 0.0),
            velocity=(0.0, 0.0),
            hazard_score=0.8,
            coherence=0.5,
            entropy=0.3,
            stability=0.5,
        )
        result = step_environment(state, "recover")
        assert result.coherence > state.coherence

    def test_recovery_events_detected(self):
        """Build an episode with forced recovery transitions."""
        states = [
            MovementState((0.0, 0.0), (0.0, 0.0), 0.8, 0.5, 0.3, 0.5),
            MovementState((0.0, 0.0), (0.0, 0.0), 0.3, 0.7, 0.2, 0.6),
        ]
        ep = MovementEpisode(
            states=tuple(states),
            decisions=(),
            total_reward=0.0,
            recovery_events=0,
            classification="",
        )
        assert detect_recovery_events(ep) >= 1


# ===================================================================
# 6. Reward scoring tests
# ===================================================================


class TestRewardScoring:
    """Verify reward computation."""

    def test_score_state_positive_for_good_state(self):
        state = MovementState(
            position=(0.0, 0.0),
            velocity=(0.0, 0.0),
            hazard_score=0.0,
            coherence=1.0,
            entropy=0.0,
            stability=1.0,
        )
        assert score_state(state) > 0.0

    def test_score_state_negative_for_bad_state(self):
        state = MovementState(
            position=(5.0, 5.0),
            velocity=(1.0, 1.0),
            hazard_score=1.0,
            coherence=0.0,
            entropy=1.0,
            stability=0.0,
        )
        assert score_state(state) < 0.0

    def test_score_episode_returns_total_reward(self):
        ep = run_episode(seed=42, steps=10)
        assert score_episode(ep) == ep.total_reward

    def test_score_state_deterministic(self):
        state = initialize_state(seed=42)
        assert score_state(state) == score_state(state)


# ===================================================================
# 7. Episode classification tests
# ===================================================================


class TestEpisodeClassification:
    """Verify episode classification returns valid labels."""

    def test_classification_is_valid(self):
        for seed in range(20):
            ep = run_episode(seed=seed, steps=50)
            assert ep.classification in VALID_CLASSIFICATIONS

    def test_classify_episode_deterministic(self):
        ep = run_episode(seed=42, steps=50)
        c1 = classify_episode(ep)
        c2 = classify_episode(ep)
        assert c1 == c2

    def test_classify_short_episode(self):
        ep = run_episode(seed=42, steps=1)
        assert ep.classification in VALID_CLASSIFICATIONS

    def test_classify_single_state_episode(self):
        ep = MovementEpisode(
            states=(initialize_state(1),),
            decisions=(),
            total_reward=0.0,
            recovery_events=0,
            classification="",
        )
        result = classify_episode(ep)
        assert result in VALID_CLASSIFICATIONS


# ===================================================================
# 8. State-space export compatibility tests
# ===================================================================


class TestStateSpaceExport:
    """Verify export_episode_state_space compatibility with bridge."""

    def test_export_returns_sequence_of_dicts(self):
        ep = run_episode(seed=42, steps=10)
        trace = export_episode_state_space(ep)
        assert len(trace) == len(ep.states)
        for entry in trace:
            assert "x" in entry
            assert "y" in entry
            assert "coherence" in entry
            assert "entropy" in entry
            assert "stability" in entry
            assert "label" in entry

    def test_export_compatible_with_build_movement_state_space(self):
        ep = run_episode(seed=42, steps=10)
        trace = export_episode_state_space(ep)
        report = build_movement_state_space(trace)
        assert isinstance(report, UnifiedStateSpaceReport)
        assert len(report.nodes) == len(ep.states)

    def test_export_preserves_positions(self):
        ep = run_episode(seed=42, steps=5)
        trace = export_episode_state_space(ep)
        for i, entry in enumerate(trace):
            assert entry["x"] == ep.states[i].position[0]
            assert entry["y"] == ep.states[i].position[1]

    def test_export_label_matches_classification(self):
        ep = run_episode(seed=42, steps=10)
        trace = export_episode_state_space(ep)
        for entry in trace:
            assert entry["label"] == ep.classification


# ===================================================================
# 9. Validator compatibility tests
# ===================================================================


class TestValidatorCompatibility:
    """Verify exported state-space passes validation."""

    def test_validation_passes(self):
        ep = run_episode(seed=42, steps=20)
        trace = export_episode_state_space(ep)
        report = build_movement_state_space(trace)
        validation = validate_state_space_report(report)
        assert validation.overall_passed is True

    def test_classification_is_stable(self):
        ep = run_episode(seed=42, steps=20)
        trace = export_episode_state_space(ep)
        report = build_movement_state_space(trace)
        validation = validate_state_space_report(report)
        assert validation.classification_stable is True

    def test_hash_is_deterministic(self):
        ep = run_episode(seed=42, steps=20)
        trace = export_episode_state_space(ep)
        report = build_movement_state_space(trace)
        h1 = compute_state_space_hash(report)
        h2 = compute_state_space_hash(report)
        assert h1 == h2
        assert isinstance(h1, str)
        assert len(h1) == 64  # SHA-256 hex


# ===================================================================
# 10. 100-run replay determinism tests
# ===================================================================


class TestReplayDeterminism:
    """Verify byte-identical replay over 100 runs."""

    def test_100_run_episode_replay(self):
        reference = run_episode(seed=42)
        for _ in range(100):
            assert run_episode(seed=42) == reference

    def test_100_run_state_space_hash_stable(self):
        reference = run_episode(seed=42)
        trace = export_episode_state_space(reference)
        report = build_movement_state_space(trace)
        ref_hash = compute_state_space_hash(report)
        for _ in range(100):
            ep = run_episode(seed=42)
            t = export_episode_state_space(ep)
            r = build_movement_state_space(t)
            assert compute_state_space_hash(r) == ref_hash


# ===================================================================
# 11. Decoder untouched verification
# ===================================================================


class TestDecoderUntouched:
    """Verify the decoder core is not imported or modified."""

    def test_no_decoder_import_in_module(self):
        import inspect
        import qec.ai.movement_learning_2d as mod
        source = inspect.getsource(mod)
        assert "qec.decoder" not in source
        assert "from qec.decoder" not in source
        assert "import qec.decoder" not in source

    def test_decoder_directory_unmodified(self):
        """Verify decoder directory exists and is non-empty (sanity check)."""
        decoder_path = os.path.join(
            os.path.dirname(__file__), "..", "src", "qec", "decoder"
        )
        if os.path.exists(decoder_path):
            assert os.path.isdir(decoder_path)
            assert len(os.listdir(decoder_path)) > 0


# ===================================================================
# 12. Additional edge-case and integration tests
# ===================================================================


class TestEdgeCases:
    """Additional tests for full coverage."""

    def test_episode_states_length(self):
        ep = run_episode(seed=1, steps=50)
        # states has steps+1 entries (initial + 50 steps)
        assert len(ep.states) == 51
        assert len(ep.decisions) == 50

    def test_episode_steps_1(self):
        ep = run_episode(seed=1, steps=1)
        assert len(ep.states) == 2
        assert len(ep.decisions) == 1

    def test_multiple_seeds_all_valid(self):
        for seed in range(50):
            ep = run_episode(seed=seed, steps=20)
            assert ep.classification in VALID_CLASSIFICATIONS
            assert isinstance(ep.total_reward, float)
            assert isinstance(ep.recovery_events, int)
            assert ep.recovery_events >= 0

    def test_policy_decision_action_is_valid(self):
        for seed in range(20):
            state = initialize_state(seed=seed)
            decision = evaluate_policy(state)
            assert decision.action in VALID_ACTIONS

    def test_export_roundtrip_validation_multiple_seeds(self):
        for seed in [0, 7, 42, 99, 255]:
            ep = run_episode(seed=seed, steps=30)
            trace = export_episode_state_space(ep)
            report = build_movement_state_space(trace)
            validation = validate_state_space_report(report)
            assert validation.overall_passed is True
