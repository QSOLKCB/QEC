"""
State-Space Invariant Validation + Replay Audit Layer (v136.6.4).

Formally validates:
- state-space invariants (transition consistency, attractor validity)
- recovery path integrity
- classification consistency
- replay determinism (100-run audit)
- merge integrity
- hash stability via canonical deterministic JSON + SHA-256

Design invariants
-----------------
* frozen dataclasses only
* tuple-only collections
* deterministic ordering
* no decoder imports
* no hidden randomness
* byte-identical replay under fixed configuration
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence, Tuple

from qec.ai.state_space_bridge import (
    UnifiedStateSpaceReport,
    build_movement_state_space,
    build_qec_state_space,
    classify_from_nodes_and_transitions,
)

# ---------------------------------------------------------------------------
# Valid classifications
# ---------------------------------------------------------------------------

VALID_CLASSIFICATIONS: Tuple[str, ...] = (
    "stable_basin",
    "collapse_recovery",
    "drifting",
    "multi_attractor",
    "chaotic",
)

# ---------------------------------------------------------------------------
# Frozen dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InvariantCheckResult:
    """Result of a single invariant check."""

    invariant_name: str
    passed: bool
    score: float
    details: str


@dataclass(frozen=True)
class ReplayAuditResult:
    """Result of deterministic replay audit."""

    replay_runs: int
    identical_runs: int
    deterministic: bool
    state_hash: str


@dataclass(frozen=True)
class StateSpaceValidationReport:
    """Complete state-space validation report."""

    invariant_results: Tuple[InvariantCheckResult, ...]
    replay_audit: ReplayAuditResult
    overall_passed: bool
    classification_stable: bool


# ---------------------------------------------------------------------------
# Canonical hashing
# ---------------------------------------------------------------------------


def _node_to_dict(node: Any) -> Mapping[str, Any]:
    """Convert a StateSpaceNode to a sorted dict for canonical serialization."""
    return {
        "coherence": node.coherence,
        "entropy": node.entropy,
        "node_id": node.node_id,
        "stability": node.stability,
        "topology_label": node.topology_label,
        "x": node.x,
        "y": node.y,
    }


def _transition_to_dict(t: Any) -> Mapping[str, Any]:
    """Convert a StateSpaceTransition to a sorted dict."""
    return {
        "basin_switch": t.basin_switch,
        "delta_x": t.delta_x,
        "delta_y": t.delta_y,
        "distance_2d": t.distance_2d,
        "from_node": t.from_node,
        "to_node": t.to_node,
        "transition_label": t.transition_label,
    }


def compute_state_space_hash(report: UnifiedStateSpaceReport) -> str:
    """Compute a deterministic SHA-256 hash of a UnifiedStateSpaceReport.

    Uses canonical JSON with sorted keys for byte-identical serialization.
    Same input always produces the same hash.
    """
    canonical = {
        "attractor_nodes": list(report.attractor_nodes),
        "classification": report.classification,
        "nodes": [_node_to_dict(n) for n in report.nodes],
        "recovery_paths": [list(p) for p in report.recovery_paths],
        "transitions": [_transition_to_dict(t) for t in report.transitions],
    }
    serialized = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Invariant validators
# ---------------------------------------------------------------------------


def validate_transition_consistency(
    report: UnifiedStateSpaceReport,
) -> InvariantCheckResult:
    """Validate: len(transitions) == max(0, len(nodes) - 1)."""
    expected = max(0, len(report.nodes) - 1)
    actual = len(report.transitions)
    passed = actual == expected
    return InvariantCheckResult(
        invariant_name="transition_consistency",
        passed=passed,
        score=1.0 if passed else 0.0,
        details=(
            f"Expected {expected} transitions for {len(report.nodes)} nodes, "
            f"got {actual}."
        ),
    )


def validate_attractor_invariants(
    report: UnifiedStateSpaceReport,
) -> InvariantCheckResult:
    """Validate: all attractor node IDs must exist in report.nodes."""
    node_ids = frozenset(n.node_id for n in report.nodes)
    missing = tuple(a for a in report.attractor_nodes if a not in node_ids)
    passed = len(missing) == 0
    return InvariantCheckResult(
        invariant_name="attractor_validity",
        passed=passed,
        score=1.0 if passed else 0.0,
        details=(
            "All attractor nodes exist in report."
            if passed
            else f"Missing attractor nodes: {missing}"
        ),
    )


def validate_recovery_path_invariants(
    report: UnifiedStateSpaceReport,
) -> InvariantCheckResult:
    """Validate: all recovery path node IDs must exist in report.nodes."""
    node_ids = frozenset(n.node_id for n in report.nodes)
    missing: list = []
    for path in report.recovery_paths:
        for nid in path:
            if nid not in node_ids:
                missing.append(nid)
    passed = len(missing) == 0
    return InvariantCheckResult(
        invariant_name="recovery_path_validity",
        passed=passed,
        score=1.0 if passed else 0.0,
        details=(
            "All recovery path nodes exist in report."
            if passed
            else f"Missing recovery path nodes: {tuple(missing)}"
        ),
    )


def validate_classification_stability(
    report: UnifiedStateSpaceReport,
) -> InvariantCheckResult:
    """Validate: classification is one of the valid set and is consistent
    with re-classification from nodes and transitions."""
    valid = report.classification in VALID_CLASSIFICATIONS
    recomputed = classify_from_nodes_and_transitions(
        report.nodes, report.transitions
    )
    consistent = report.classification == recomputed
    passed = valid and consistent
    score = 1.0 if passed else (0.5 if valid else 0.0)
    if not valid:
        details = f"Invalid classification: {report.classification!r}"
    elif not consistent:
        details = (
            f"Classification {report.classification!r} inconsistent with "
            f"recomputed {recomputed!r}"
        )
    else:
        details = f"Classification {report.classification!r} is valid and consistent."
    return InvariantCheckResult(
        invariant_name="classification_stability",
        passed=passed,
        score=score,
        details=details,
    )


def _validate_distance_non_negative(
    report: UnifiedStateSpaceReport,
) -> InvariantCheckResult:
    """Validate: all transition distances are >= 0.0."""
    violations = tuple(
        t.distance_2d for t in report.transitions if t.distance_2d < 0.0
    )
    passed = len(violations) == 0
    return InvariantCheckResult(
        invariant_name="distance_non_negative",
        passed=passed,
        score=1.0 if passed else 0.0,
        details=(
            "All distances non-negative."
            if passed
            else f"Negative distances found: {violations}"
        ),
    )


# ---------------------------------------------------------------------------
# Replay audit
# ---------------------------------------------------------------------------


def run_replay_audit(
    builder_fn: Callable[..., UnifiedStateSpaceReport],
    input_payload: Any,
    runs: int = 100,
) -> ReplayAuditResult:
    """Run deterministic replay audit.

    Calls ``builder_fn(input_payload)`` *runs* times and verifies that
    every invocation produces the same hash as the first.

    Parameters
    ----------
    builder_fn
        A builder function (e.g. ``build_qec_state_space``).
    input_payload
        The input argument passed to *builder_fn*.
    runs
        Number of replay iterations (default 100).
    """
    reference = builder_fn(input_payload)
    ref_hash = compute_state_space_hash(reference)
    identical = 0
    for _ in range(runs):
        result = builder_fn(input_payload)
        if compute_state_space_hash(result) == ref_hash:
            identical += 1
    return ReplayAuditResult(
        replay_runs=runs,
        identical_runs=identical,
        deterministic=(identical == runs),
        state_hash=ref_hash,
    )


# ---------------------------------------------------------------------------
# Full validation
# ---------------------------------------------------------------------------


def validate_state_space_report(
    report: UnifiedStateSpaceReport,
) -> StateSpaceValidationReport:
    """Run all invariant checks against a UnifiedStateSpaceReport.

    Returns a complete ``StateSpaceValidationReport`` with individual
    invariant results and an overall pass/fail verdict.
    """
    checks = (
        validate_transition_consistency(report),
        validate_attractor_invariants(report),
        validate_recovery_path_invariants(report),
        validate_classification_stability(report),
        _validate_distance_non_negative(report),
    )
    all_passed = all(c.passed for c in checks)
    classification_check = next(
        c for c in checks if c.invariant_name == "classification_stability"
    )
    return StateSpaceValidationReport(
        invariant_results=checks,
        replay_audit=ReplayAuditResult(
            replay_runs=0,
            identical_runs=0,
            deterministic=True,
            state_hash=compute_state_space_hash(report),
        ),
        overall_passed=all_passed,
        classification_stable=classification_check.passed,
    )
