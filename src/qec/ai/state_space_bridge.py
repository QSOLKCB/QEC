"""
Universal 2D State-Space Bridge Layer (v136.6.3).

Converts audio topology lineage, movement trajectories, and QEC controller
metrics into a shared deterministic 2D state-space.  This is the canonical
mathematical bridge for attractor, basin-switch, recovery-arc, and regime
topology analysis.

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

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence, Tuple, Union

# ---------------------------------------------------------------------------
# Thresholds (mirrored from audio_topology_lineage for consistency)
# ---------------------------------------------------------------------------
ATTRACTOR_PROXIMITY_THRESHOLD: float = 0.05
BASIN_SWITCH_DISTANCE_THRESHOLD: float = 0.25
BASIN_SWITCH_COHERENCE_THRESHOLD: float = 0.05

# ---------------------------------------------------------------------------
# Frozen dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StateSpaceNode:
    """Single point in the unified 2D state-space."""

    node_id: str
    x: float
    y: float
    coherence: float
    entropy: float
    stability: float
    topology_label: str


@dataclass(frozen=True)
class StateSpaceTransition:
    """Ordered transition between two state-space nodes."""

    from_node: str
    to_node: str
    delta_x: float
    delta_y: float
    distance_2d: float
    transition_label: str
    basin_switch: bool


@dataclass(frozen=True)
class UnifiedStateSpaceReport:
    """Complete unified state-space analysis report."""

    nodes: Tuple[StateSpaceNode, ...]
    transitions: Tuple[StateSpaceTransition, ...]
    attractor_nodes: Tuple[str, ...]
    recovery_paths: Tuple[Tuple[str, ...], ...]
    classification: str


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def _distance_2d(x1: float, y1: float, x2: float, y2: float) -> float:
    """Euclidean distance in 2D."""
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def _is_basin_switch(
    dist: float,
    coherence_delta: float,
) -> bool:
    """Determine whether a transition constitutes a basin switch."""
    return (
        dist > BASIN_SWITCH_DISTANCE_THRESHOLD
        or abs(coherence_delta) > BASIN_SWITCH_COHERENCE_THRESHOLD
    )


def _transition_label(dist: float, coherence_delta: float) -> str:
    """Classify a transition deterministically."""
    if _is_basin_switch(dist, coherence_delta):
        if coherence_delta < -BASIN_SWITCH_COHERENCE_THRESHOLD:
            return "degrading"
        if coherence_delta > BASIN_SWITCH_COHERENCE_THRESHOLD:
            return "recovering"
        return "basin_jump"
    if dist < ATTRACTOR_PROXIMITY_THRESHOLD:
        return "stable"
    return "drifting"


def _compute_transitions(
    nodes: Tuple[StateSpaceNode, ...],
) -> Tuple[StateSpaceTransition, ...]:
    """Build the ordered transition sequence for a node tuple."""
    transitions: List[StateSpaceTransition] = []
    for i in range(len(nodes) - 1):
        a, b = nodes[i], nodes[i + 1]
        dx = b.x - a.x
        dy = b.y - a.y
        dist = _distance_2d(a.x, a.y, b.x, b.y)
        c_delta = b.coherence - a.coherence
        label = _transition_label(dist, c_delta)
        basin = _is_basin_switch(dist, c_delta)
        transitions.append(
            StateSpaceTransition(
                from_node=a.node_id,
                to_node=b.node_id,
                delta_x=dx,
                delta_y=dy,
                distance_2d=dist,
                transition_label=label,
                basin_switch=basin,
            )
        )
    return tuple(transitions)


# ---------------------------------------------------------------------------
# Audio integration
# ---------------------------------------------------------------------------

# Lazy import to keep the module importable even when audio is absent.
_AudioTopologyNode = None
_AudioLineageReport = None


def _ensure_audio_imports() -> None:
    global _AudioTopologyNode, _AudioLineageReport
    if _AudioTopologyNode is None:
        from qec.audio.audio_topology_lineage import (
            AudioLineageReport,
            AudioTopologyNode,
        )

        _AudioTopologyNode = AudioTopologyNode
        _AudioLineageReport = AudioLineageReport


def _audio_node_to_ss(node: object) -> StateSpaceNode:
    """Convert an AudioTopologyNode to a StateSpaceNode."""
    # AudioTopologyNode has: filename, coherence_score, spectral_entropy,
    # harmonic_density, cluster_tightness, mapping_2d, topology_label
    n = node  # type: ignore[union-attr]
    x, y = n.mapping_2d
    return StateSpaceNode(
        node_id=n.filename,
        x=float(x),
        y=float(y),
        coherence=float(n.coherence_score),
        entropy=float(n.spectral_entropy),
        stability=float(n.cluster_tightness),
        topology_label=n.topology_label,
    )


def build_audio_state_space(
    lineage_report: object,
) -> UnifiedStateSpaceReport:
    """Build a unified state-space from an AudioLineageReport.

    Parameters
    ----------
    lineage_report
        An ``AudioLineageReport`` instance produced by
        ``build_audio_lineage()`` or ``build_lineage_from_nodes()``.
    """
    _ensure_audio_imports()
    report = lineage_report  # type: ignore[union-attr]
    nodes = tuple(_audio_node_to_ss(n) for n in report.nodes)
    transitions = _compute_transitions(nodes)
    attractors = detect_shared_attractors_from_nodes(nodes)
    recovery = detect_recovery_paths_from_nodes(nodes)
    classification = classify_from_nodes_and_transitions(nodes, transitions)
    return UnifiedStateSpaceReport(
        nodes=nodes,
        transitions=transitions,
        attractor_nodes=attractors,
        recovery_paths=recovery,
        classification=classification,
    )


# ---------------------------------------------------------------------------
# QEC metric integration
# ---------------------------------------------------------------------------


def build_qec_state_space(
    metrics_report: Sequence[Mapping[str, Any]],
) -> UnifiedStateSpaceReport:
    """Build a unified state-space from QEC controller metric dicts.

    Expected keys per dict: ``stability``, ``entropy``, ``convergence``,
    ``syndrome_consistency``.  An optional ``label`` key provides a
    topology label (defaults to ``"qec"``).

    Mapping:
        x = stability score
        y = 1.0 - entropy  (entropy inverse, clamped to [0, 1])
        coherence = convergence quality
        stability = syndrome consistency
    """
    nodes: List[StateSpaceNode] = []
    for idx, m in enumerate(metrics_report):
        stability = float(m.get("stability", 0.0))
        entropy = float(m.get("entropy", 0.0))
        convergence = float(m.get("convergence", 0.0))
        syndrome = float(m.get("syndrome_consistency", 0.0))
        label = str(m.get("label", "qec"))
        node_id = f"qec_{idx:04d}"
        nodes.append(
            StateSpaceNode(
                node_id=node_id,
                x=stability,
                y=max(0.0, min(1.0, 1.0 - entropy)),
                coherence=convergence,
                entropy=entropy,
                stability=syndrome,
                topology_label=label,
            )
        )
    nodes_t = tuple(nodes)
    transitions = _compute_transitions(nodes_t)
    attractors = detect_shared_attractors_from_nodes(nodes_t)
    recovery = detect_recovery_paths_from_nodes(nodes_t)
    classification = classify_from_nodes_and_transitions(nodes_t, transitions)
    return UnifiedStateSpaceReport(
        nodes=nodes_t,
        transitions=transitions,
        attractor_nodes=attractors,
        recovery_paths=recovery,
        classification=classification,
    )


# ---------------------------------------------------------------------------
# Movement integration
# ---------------------------------------------------------------------------


def build_movement_state_space(
    trace_report: Sequence[Mapping[str, Any]],
) -> UnifiedStateSpaceReport:
    """Build a unified state-space from movement trajectory trace dicts.

    Expected keys per dict: ``x``, ``y``, ``coherence``, ``entropy``,
    ``stability``.  An optional ``label`` key provides a topology label
    (defaults to ``"movement"``).
    """
    nodes: List[StateSpaceNode] = []
    for idx, m in enumerate(trace_report):
        label = str(m.get("label", "movement"))
        node_id = f"mov_{idx:04d}"
        nodes.append(
            StateSpaceNode(
                node_id=node_id,
                x=float(m.get("x", 0.0)),
                y=float(m.get("y", 0.0)),
                coherence=float(m.get("coherence", 0.0)),
                entropy=float(m.get("entropy", 0.0)),
                stability=float(m.get("stability", 0.0)),
                topology_label=label,
            )
        )
    nodes_t = tuple(nodes)
    transitions = _compute_transitions(nodes_t)
    attractors = detect_shared_attractors_from_nodes(nodes_t)
    recovery = detect_recovery_paths_from_nodes(nodes_t)
    classification = classify_from_nodes_and_transitions(nodes_t, transitions)
    return UnifiedStateSpaceReport(
        nodes=nodes_t,
        transitions=transitions,
        attractor_nodes=attractors,
        recovery_paths=recovery,
        classification=classification,
    )


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------


def merge_state_spaces(
    reports: Sequence[UnifiedStateSpaceReport],
) -> UnifiedStateSpaceReport:
    """Merge multiple state-space reports into one unified report.

    Nodes are concatenated in deterministic input order and transitions are
    recomputed over the merged node sequence.  Attractors and recovery paths
    are recomputed over the merged node set.
    """
    all_nodes: List[StateSpaceNode] = []
    for r in reports:
        all_nodes.extend(r.nodes)
    nodes_t = tuple(all_nodes)
    transitions_t = _compute_transitions(nodes_t)
    attractors = detect_shared_attractors_from_nodes(nodes_t)
    recovery = detect_recovery_paths_from_nodes(nodes_t)
    classification = classify_from_nodes_and_transitions(nodes_t, transitions_t)
    return UnifiedStateSpaceReport(
        nodes=nodes_t,
        transitions=transitions_t,
        attractor_nodes=attractors,
        recovery_paths=recovery,
        classification=classification,
    )


# ---------------------------------------------------------------------------
# Analysis — attractors
# ---------------------------------------------------------------------------


def detect_shared_attractors_from_nodes(
    nodes: Tuple[StateSpaceNode, ...],
) -> Tuple[str, ...]:
    """Identify attractor nodes (consecutive proximity < threshold)."""
    if len(nodes) < 2:
        return ()
    attractor_ids: List[str] = []
    seen: set = set()
    for i in range(len(nodes) - 1):
        a, b = nodes[i], nodes[i + 1]
        dist = _distance_2d(a.x, a.y, b.x, b.y)
        if dist < ATTRACTOR_PROXIMITY_THRESHOLD:
            if a.node_id not in seen:
                attractor_ids.append(a.node_id)
                seen.add(a.node_id)
            if b.node_id not in seen:
                attractor_ids.append(b.node_id)
                seen.add(b.node_id)
    return tuple(attractor_ids)


def detect_shared_attractors(
    report: UnifiedStateSpaceReport,
) -> Tuple[str, ...]:
    """Public API: detect shared attractors from a report."""
    return detect_shared_attractors_from_nodes(report.nodes)


# ---------------------------------------------------------------------------
# Analysis — recovery paths
# ---------------------------------------------------------------------------


def detect_recovery_paths_from_nodes(
    nodes: Tuple[StateSpaceNode, ...],
) -> Tuple[Tuple[str, ...], ...]:
    """Detect collapse-then-recovery arcs in the node sequence.

    A recovery path is a contiguous sub-sequence where:
    1. Coherence drops below the initial coherence (collapse phase).
    2. Coherence later rises above the initial coherence (recovery phase).
    """
    if len(nodes) < 3:
        return ()
    paths: List[Tuple[str, ...]] = []
    i = 0
    while i < len(nodes) - 2:
        initial_c = nodes[i].coherence
        # Look for a drop
        j = i + 1
        found_drop = False
        while j < len(nodes):
            if nodes[j].coherence < initial_c:
                found_drop = True
                break
            j += 1
        if not found_drop:
            i += 1
            continue
        # Look for recovery after the drop
        k = j + 1
        while k < len(nodes):
            if nodes[k].coherence > initial_c:
                # Found a recovery path: i -> j -> k
                path_ids = tuple(
                    nodes[idx].node_id for idx in range(i, k + 1)
                )
                paths.append(path_ids)
                i = k + 1
                break
            k += 1
        else:
            i = j + 1
    return tuple(paths)


def detect_recovery_paths(
    report: UnifiedStateSpaceReport,
) -> Tuple[Tuple[str, ...], ...]:
    """Public API: detect recovery paths from a report."""
    return detect_recovery_paths_from_nodes(report.nodes)


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def classify_from_nodes_and_transitions(
    nodes: Tuple[StateSpaceNode, ...],
    transitions: Tuple[StateSpaceTransition, ...],
) -> str:
    """Classify the overall state-space topology.

    Returns one of:
        stable_basin, collapse_recovery, drifting, multi_attractor, chaotic
    """
    if len(nodes) == 0:
        return "stable_basin"

    n_transitions = len(transitions)
    if n_transitions == 0:
        return "stable_basin"

    basin_switches = sum(1 for t in transitions if t.basin_switch)
    switch_ratio = basin_switches / n_transitions

    attractors = detect_shared_attractors_from_nodes(nodes)
    recovery = detect_recovery_paths_from_nodes(nodes)

    # Collapse-recovery detected (takes precedence over chaotic)
    if len(recovery) > 0:
        return "collapse_recovery"

    # Chaotic: high switch ratio
    if switch_ratio > 0.6:
        return "chaotic"

    # Multi-attractor: multiple distinct attractor clusters
    if len(attractors) >= 4:
        return "multi_attractor"

    # Drifting: moderate switching, no clear attractor
    if basin_switches > 0 and len(attractors) == 0:
        return "drifting"

    return "stable_basin"


def classify_state_space(
    report: UnifiedStateSpaceReport,
) -> str:
    """Public API: classify a unified state-space report."""
    return classify_from_nodes_and_transitions(report.nodes, report.transitions)
