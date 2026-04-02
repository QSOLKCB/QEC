"""
Tests for the State-Space Invariant Validation + Replay Audit Layer (v136.6.4).

Minimum 40 tests covering:
 1. dataclass immutability
 2. hash stability (100-replay)
 3. transition invariants
 4. attractor validity
 5. recovery path validity
 6. classification validation
 7. replay audit (100x)
 8. hash replay identity
 9. merged report validation
10. decoder untouched verification
11. distance invariants
12. canonical serialization determinism
"""

from __future__ import annotations

import ast
import hashlib
import json
import os

import pytest

from qec.ai.state_space_bridge import (
    StateSpaceNode,
    StateSpaceTransition,
    UnifiedStateSpaceReport,
    _compute_transitions,
    build_movement_state_space,
    build_qec_state_space,
    merge_state_spaces,
)
from qec.ai.state_space_validator import (
    VALID_CLASSIFICATIONS,
    InvariantCheckResult,
    ReplayAuditResult,
    StateSpaceValidationReport,
    compute_state_space_hash,
    run_replay_audit,
    validate_attractor_invariants,
    validate_classification_stability,
    validate_recovery_path_invariants,
    validate_state_space_report,
    validate_transition_consistency,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _node(
    nid: str = "n0",
    x: float = 0.0,
    y: float = 0.0,
    coherence: float = 0.9,
    entropy: float = 0.1,
    stability: float = 0.8,
    label: str = "test",
) -> StateSpaceNode:
    return StateSpaceNode(
        node_id=nid, x=x, y=y, coherence=coherence,
        entropy=entropy, stability=stability, topology_label=label,
    )


def _make_report(
    nodes=None,
    transitions=None,
    attractor_nodes=None,
    recovery_paths=None,
    classification="stable_basin",
) -> UnifiedStateSpaceReport:
    """Build a report with auto-computed transitions if not provided."""
    if nodes is None:
        nodes = ()
    if transitions is None:
        transitions = _compute_transitions(nodes)
    if attractor_nodes is None:
        attractor_nodes = ()
    if recovery_paths is None:
        recovery_paths = ()
    return UnifiedStateSpaceReport(
        nodes=nodes,
        transitions=transitions,
        attractor_nodes=attractor_nodes,
        recovery_paths=recovery_paths,
        classification=classification,
    )


def _sample_qec_metrics():
    return [
        {"stability": 0.9, "entropy": 0.1, "convergence": 0.95,
         "syndrome_consistency": 0.88},
        {"stability": 0.7, "entropy": 0.3, "convergence": 0.6,
         "syndrome_consistency": 0.7},
        {"stability": 0.85, "entropy": 0.15, "convergence": 0.9,
         "syndrome_consistency": 0.85},
    ]


def _sample_movement_trace():
    return [
        {"x": 0.1, "y": 0.2, "coherence": 0.8, "entropy": 0.2, "stability": 0.7},
        {"x": 0.3, "y": 0.4, "coherence": 0.6, "entropy": 0.4, "stability": 0.5},
        {"x": 0.15, "y": 0.25, "coherence": 0.85, "entropy": 0.15, "stability": 0.75},
    ]


# ===================================================================
# 1. Dataclass immutability
# ===================================================================


class TestDataclassImmutability:
    def test_invariant_check_result_frozen(self):
        r = InvariantCheckResult("test", True, 1.0, "ok")
        with pytest.raises(AttributeError):
            r.passed = False  # type: ignore[misc]

    def test_replay_audit_result_frozen(self):
        r = ReplayAuditResult(100, 100, True, "abc")
        with pytest.raises(AttributeError):
            r.deterministic = False  # type: ignore[misc]

    def test_state_space_validation_report_frozen(self):
        r = StateSpaceValidationReport(
            invariant_results=(),
            replay_audit=ReplayAuditResult(0, 0, True, ""),
            overall_passed=True,
            classification_stable=True,
        )
        with pytest.raises(AttributeError):
            r.overall_passed = False  # type: ignore[misc]

    def test_invariant_check_result_fields(self):
        r = InvariantCheckResult("name", True, 0.5, "detail")
        assert r.invariant_name == "name"
        assert r.passed is True
        assert r.score == 0.5
        assert r.details == "detail"

    def test_replay_audit_result_fields(self):
        r = ReplayAuditResult(50, 50, True, "hash_val")
        assert r.replay_runs == 50
        assert r.identical_runs == 50
        assert r.deterministic is True
        assert r.state_hash == "hash_val"


# ===================================================================
# 2. Hash stability
# ===================================================================


class TestHashStability:
    def test_hash_is_sha256_hex(self):
        report = _make_report()
        h = compute_state_space_hash(report)
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)

    def test_hash_deterministic_simple(self):
        report = _make_report()
        h1 = compute_state_space_hash(report)
        h2 = compute_state_space_hash(report)
        assert h1 == h2

    def test_hash_100_replay(self):
        """Same report hashed 100 times must produce identical hash."""
        nodes = (
            _node("a", 0.5, 0.5, 0.9),
            _node("b", 0.51, 0.51, 0.9),
        )
        report = _make_report(nodes=nodes, attractor_nodes=("a", "b"))
        ref_hash = compute_state_space_hash(report)
        for _ in range(100):
            assert compute_state_space_hash(report) == ref_hash

    def test_different_reports_different_hashes(self):
        r1 = _make_report(nodes=(_node("a"),))
        r2 = _make_report(nodes=(_node("b"),))
        assert compute_state_space_hash(r1) != compute_state_space_hash(r2)

    def test_hash_changes_with_classification(self):
        nodes = (_node("a"),)
        r1 = _make_report(nodes=nodes, classification="stable_basin")
        r2 = _make_report(nodes=nodes, classification="chaotic")
        assert compute_state_space_hash(r1) != compute_state_space_hash(r2)

    def test_hash_changes_with_attractor(self):
        nodes = (_node("a"), _node("b", x=0.01, y=0.01))
        r1 = _make_report(nodes=nodes, attractor_nodes=())
        r2 = _make_report(nodes=nodes, attractor_nodes=("a",))
        assert compute_state_space_hash(r1) != compute_state_space_hash(r2)


# ===================================================================
# 3. Transition invariants
# ===================================================================


class TestTransitionInvariants:
    def test_empty_report_passes(self):
        result = validate_transition_consistency(_make_report())
        assert result.passed is True
        assert result.score == 1.0

    def test_single_node_passes(self):
        result = validate_transition_consistency(
            _make_report(nodes=(_node("a"),))
        )
        assert result.passed is True

    def test_correct_count_passes(self):
        nodes = (_node("a"), _node("b", x=0.1), _node("c", x=0.2))
        result = validate_transition_consistency(_make_report(nodes=nodes))
        assert result.passed is True
        assert result.invariant_name == "transition_consistency"

    def test_wrong_count_fails(self):
        nodes = (_node("a"), _node("b", x=0.1))
        # Force wrong transition count by providing empty transitions
        report = _make_report(nodes=nodes, transitions=())
        result = validate_transition_consistency(report)
        assert result.passed is False
        assert result.score == 0.0

    def test_extra_transitions_fail(self):
        nodes = (_node("a"),)
        fake_t = (StateSpaceTransition("a", "b", 0.1, 0.1, 0.14, "stable", False),)
        report = _make_report(nodes=nodes, transitions=fake_t)
        result = validate_transition_consistency(report)
        assert result.passed is False


# ===================================================================
# 4. Attractor validity
# ===================================================================


class TestAttractorValidity:
    def test_valid_attractors_pass(self):
        nodes = (_node("a"), _node("b", x=0.01))
        report = _make_report(nodes=nodes, attractor_nodes=("a", "b"))
        result = validate_attractor_invariants(report)
        assert result.passed is True

    def test_empty_attractors_pass(self):
        result = validate_attractor_invariants(_make_report())
        assert result.passed is True

    def test_missing_attractor_fails(self):
        nodes = (_node("a"),)
        report = _make_report(nodes=nodes, attractor_nodes=("a", "ghost"))
        result = validate_attractor_invariants(report)
        assert result.passed is False
        assert "ghost" in result.details

    def test_attractor_name(self):
        result = validate_attractor_invariants(_make_report())
        assert result.invariant_name == "attractor_validity"


# ===================================================================
# 5. Recovery path validity
# ===================================================================


class TestRecoveryPathValidity:
    def test_valid_recovery_passes(self):
        nodes = (_node("a"), _node("b", x=0.1), _node("c", x=0.2))
        report = _make_report(
            nodes=nodes,
            recovery_paths=(("a", "b", "c"),),
        )
        result = validate_recovery_path_invariants(report)
        assert result.passed is True

    def test_empty_recovery_passes(self):
        result = validate_recovery_path_invariants(_make_report())
        assert result.passed is True

    def test_missing_recovery_node_fails(self):
        nodes = (_node("a"), _node("b", x=0.1))
        report = _make_report(
            nodes=nodes,
            recovery_paths=(("a", "b", "phantom"),),
        )
        result = validate_recovery_path_invariants(report)
        assert result.passed is False
        assert "phantom" in result.details

    def test_recovery_invariant_name(self):
        result = validate_recovery_path_invariants(_make_report())
        assert result.invariant_name == "recovery_path_validity"


# ===================================================================
# 6. Classification validation
# ===================================================================


class TestClassificationValidation:
    def test_valid_classification_passes(self):
        report = build_qec_state_space(_sample_qec_metrics())
        result = validate_classification_stability(report)
        assert result.passed is True

    def test_invalid_classification_fails(self):
        report = _make_report(classification="nonsense")
        result = validate_classification_stability(report)
        assert result.passed is False
        assert result.score == 0.0
        assert "Invalid" in result.details

    def test_inconsistent_classification_fails(self):
        """Classification doesn't match recomputed value."""
        nodes = (
            _node("a", 0.5, 0.5, 0.9),
            _node("b", 0.51, 0.51, 0.9),
        )
        # These nodes would classify as stable_basin, but we set chaotic
        report = _make_report(nodes=nodes, classification="chaotic")
        result = validate_classification_stability(report)
        assert result.passed is False
        assert result.score == 0.5  # valid but inconsistent

    def test_all_valid_classifications(self):
        for c in VALID_CLASSIFICATIONS:
            assert c in (
                "stable_basin", "collapse_recovery", "drifting",
                "multi_attractor", "chaotic",
            )

    def test_classification_invariant_name(self):
        result = validate_classification_stability(_make_report())
        assert result.invariant_name == "classification_stability"


# ===================================================================
# 7. Replay audit (100x)
# ===================================================================


class TestReplayAudit:
    def test_qec_replay_100(self):
        metrics = _sample_qec_metrics()
        result = run_replay_audit(build_qec_state_space, metrics, runs=100)
        assert result.replay_runs == 100
        assert result.identical_runs == 100
        assert result.deterministic is True
        assert len(result.state_hash) == 64

    def test_movement_replay_100(self):
        trace = _sample_movement_trace()
        result = run_replay_audit(build_movement_state_space, trace, runs=100)
        assert result.deterministic is True
        assert result.identical_runs == 100

    def test_replay_custom_runs(self):
        metrics = _sample_qec_metrics()
        result = run_replay_audit(build_qec_state_space, metrics, runs=10)
        assert result.replay_runs == 10
        assert result.identical_runs == 10

    def test_replay_hash_matches_direct_hash(self):
        metrics = _sample_qec_metrics()
        report = build_qec_state_space(metrics)
        direct_hash = compute_state_space_hash(report)
        audit = run_replay_audit(build_qec_state_space, metrics, runs=5)
        assert audit.state_hash == direct_hash


# ===================================================================
# 8. Hash replay identity
# ===================================================================


class TestHashReplayIdentity:
    def test_qec_hash_100_replay(self):
        report = build_qec_state_space(_sample_qec_metrics())
        ref_hash = compute_state_space_hash(report)
        for _ in range(100):
            assert compute_state_space_hash(report) == ref_hash

    def test_movement_hash_100_replay(self):
        report = build_movement_state_space(_sample_movement_trace())
        ref_hash = compute_state_space_hash(report)
        for _ in range(100):
            assert compute_state_space_hash(report) == ref_hash

    def test_empty_report_hash_stable(self):
        report = _make_report()
        ref_hash = compute_state_space_hash(report)
        for _ in range(100):
            assert compute_state_space_hash(report) == ref_hash


# ===================================================================
# 9. Merged report validation
# ===================================================================


class TestMergedReportValidation:
    def test_merged_passes_all_invariants(self):
        r1 = build_qec_state_space(_sample_qec_metrics())
        r2 = build_movement_state_space(_sample_movement_trace())
        merged = merge_state_spaces([r1, r2])
        validation = validate_state_space_report(merged)
        assert validation.overall_passed is True

    def test_merged_hash_deterministic(self):
        r1 = build_qec_state_space(_sample_qec_metrics())
        r2 = build_movement_state_space(_sample_movement_trace())
        merged = merge_state_spaces([r1, r2])
        ref_hash = compute_state_space_hash(merged)
        for _ in range(100):
            m2 = merge_state_spaces([r1, r2])
            assert compute_state_space_hash(m2) == ref_hash

    def test_merged_transition_consistency(self):
        r1 = build_qec_state_space(_sample_qec_metrics())
        r2 = build_movement_state_space(_sample_movement_trace())
        merged = merge_state_spaces([r1, r2])
        result = validate_transition_consistency(merged)
        assert result.passed is True


# ===================================================================
# 10. Decoder untouched verification
# ===================================================================


class TestDecoderUntouched:
    def test_no_decoder_import_in_validator(self):
        """Verify state_space_validator does not import from qec.decoder."""
        import qec.ai.state_space_validator as mod

        source_path = mod.__file__
        assert source_path is not None
        with open(source_path, "r") as f:
            tree = ast.parse(f.read(), filename=source_path)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                assert not node.module.startswith("qec.decoder"), (
                    f"Forbidden decoder import: from {node.module}"
                )
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    assert not alias.name.startswith("qec.decoder"), (
                        f"Forbidden decoder import: import {alias.name}"
                    )

    def test_decoder_directory_exists_untouched(self):
        """Verify decoder directory exists and has not been removed."""
        decoder_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "src", "qec", "decoder",
        )
        assert os.path.isdir(decoder_path)


# ===================================================================
# 11. Distance invariants
# ===================================================================


class TestDistanceInvariants:
    def test_all_distances_non_negative(self):
        report = build_qec_state_space(_sample_qec_metrics())
        for t in report.transitions:
            assert t.distance_2d >= 0.0

    def test_distance_check_in_validation(self):
        report = build_qec_state_space(_sample_qec_metrics())
        validation = validate_state_space_report(report)
        distance_check = next(
            c for c in validation.invariant_results
            if c.invariant_name == "distance_non_negative"
        )
        assert distance_check.passed is True


# ===================================================================
# 12. Full validation report
# ===================================================================


class TestFullValidation:
    def test_qec_report_passes(self):
        report = build_qec_state_space(_sample_qec_metrics())
        validation = validate_state_space_report(report)
        assert validation.overall_passed is True
        assert validation.classification_stable is True

    def test_movement_report_passes(self):
        report = build_movement_state_space(_sample_movement_trace())
        validation = validate_state_space_report(report)
        assert validation.overall_passed is True

    def test_empty_report_passes(self):
        report = _make_report()
        validation = validate_state_space_report(report)
        assert validation.overall_passed is True

    def test_validation_contains_all_checks(self):
        report = build_qec_state_space(_sample_qec_metrics())
        validation = validate_state_space_report(report)
        names = {c.invariant_name for c in validation.invariant_results}
        assert "transition_consistency" in names
        assert "attractor_validity" in names
        assert "recovery_path_validity" in names
        assert "classification_stability" in names
        assert "distance_non_negative" in names

    def test_validation_includes_hash(self):
        report = build_qec_state_space(_sample_qec_metrics())
        validation = validate_state_space_report(report)
        assert len(validation.replay_audit.state_hash) == 64

    def test_validation_report_frozen(self):
        report = build_qec_state_space(_sample_qec_metrics())
        validation = validate_state_space_report(report)
        with pytest.raises(AttributeError):
            validation.overall_passed = False  # type: ignore[misc]
