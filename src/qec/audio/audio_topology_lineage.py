"""Temporal Lineage Analysis of Audio Topology — v136.6.2

Extends pairwise comparisons into a full temporal topology lineage engine.
Models: audio states → transitions → basins → attractors → recovery paths.

Consumes real outputs from quantum_audio_logic_lab.py.
All structures are frozen, deterministic, and replay-safe.
"""

import math
from dataclasses import dataclass
from typing import Tuple

from qec.audio.quantum_audio_logic_lab import (
    QuantumAudioLogicReport,
    _cluster_tightness,
    analyze_quantum_audio_file,
)


# ── Frozen dataclasses ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class AudioTopologyNode:
    """A single node in the audio topology state space."""

    filename: str
    coherence_score: float
    spectral_entropy: float
    harmonic_density: float
    cluster_tightness: float
    mapping_2d: Tuple[float, float]
    topology_label: str


@dataclass(frozen=True)
class AudioTopologyTransition:
    """An ordered transition between two topology nodes."""

    from_node: str
    to_node: str
    coherence_delta: float
    entropy_delta: float
    distance_2d: float
    psd_similarity: float
    transition_label: str


@dataclass(frozen=True)
class AudioLineageReport:
    """Complete temporal lineage analysis of an audio trajectory."""

    nodes: Tuple[AudioTopologyNode, ...]
    transitions: Tuple[AudioTopologyTransition, ...]
    basin_switch_count: int
    attractor_count: int
    recovery_path_detected: bool
    lineage_label: str


# ── Node construction ───────────────────────────────────────────────────────


def _topology_label_for_report(report: QuantumAudioLogicReport) -> str:
    """Assign a deterministic topology label from a single report."""
    c = report.coherence_score
    e = report.spectral_entropy
    if c >= 0.7 and e < 0.5:
        return "stable"
    if c < 0.3:
        return "collapsed"
    if e >= 0.8:
        return "chaotic"
    if c >= 0.5:
        return "recovering"
    return "transitional"


def _report_to_node(report: QuantumAudioLogicReport) -> AudioTopologyNode:
    """Convert a QuantumAudioLogicReport into an AudioTopologyNode."""
    tightness = _cluster_tightness(report.cluster_points)
    label = _topology_label_for_report(report)
    return AudioTopologyNode(
        filename=report.filename,
        coherence_score=report.coherence_score,
        spectral_entropy=report.spectral_entropy,
        harmonic_density=report.harmonic_density,
        cluster_tightness=tightness,
        mapping_2d=report.mapping_2d,
        topology_label=label,
    )


# ── Transition computation ──────────────────────────────────────────────────


def _distance_2d(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """Euclidean distance between two 2D mapping points."""
    return math.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2)


def _transition_label(
    coherence_delta: float,
    entropy_delta: float,
    distance: float,
) -> str:
    """Deterministic transition classification."""
    if abs(coherence_delta) < 0.02 and distance < 0.1:
        return "stable"
    if coherence_delta < -0.05:
        return "degrading"
    if coherence_delta > 0.05:
        return "recovering"
    if distance > 0.25:
        return "basin_jump"
    return "drifting"


def _compute_psd_similarity_from_nodes(
    node_a: AudioTopologyNode,
    node_b: AudioTopologyNode,
) -> float:
    """Deterministic PSD similarity proxy from topology features.

    Uses a feature-distance-based similarity measure that is fully
    deterministic and does not require raw waveform access.  This keeps
    the topology layer independent of file I/O while remaining
    scientifically meaningful: nodes with similar spectral entropy,
    harmonic density, and coherence produce similar PSDs.
    """
    d_coherence = abs(node_a.coherence_score - node_b.coherence_score)
    d_entropy = abs(node_a.spectral_entropy - node_b.spectral_entropy)
    d_density = abs(node_a.harmonic_density - node_b.harmonic_density)
    # Weighted feature distance → similarity in [0, 1]
    feature_dist = 0.4 * d_coherence + 0.35 * d_entropy + 0.25 * d_density
    return max(0.0, min(1.0, 1.0 - feature_dist))


def _compute_transition(
    node_a: AudioTopologyNode,
    node_b: AudioTopologyNode,
) -> AudioTopologyTransition:
    """Compute an ordered transition from node_a to node_b."""
    c_delta = node_b.coherence_score - node_a.coherence_score
    e_delta = node_b.spectral_entropy - node_a.spectral_entropy
    dist = _distance_2d(node_a.mapping_2d, node_b.mapping_2d)
    psd_sim = _compute_psd_similarity_from_nodes(node_a, node_b)
    label = _transition_label(c_delta, e_delta, dist)
    return AudioTopologyTransition(
        from_node=node_a.filename,
        to_node=node_b.filename,
        coherence_delta=c_delta,
        entropy_delta=e_delta,
        distance_2d=dist,
        psd_similarity=psd_sim,
        transition_label=label,
    )


# ── Basin & attractor detection ─────────────────────────────────────────────


def detect_basin_switches(report: AudioLineageReport) -> int:
    """Count basin switches in a lineage report.

    A basin switch occurs if any of:
      - distance_2d > 0.25
      - coherence_delta > 0.05  (absolute value)
      - psd_similarity < 0.7
    """
    count = 0
    for t in report.transitions:
        if (
            t.distance_2d > 0.25
            or abs(t.coherence_delta) > 0.05
            or t.psd_similarity < 0.7
        ):
            count += 1
    return count


def _count_attractors(nodes: Tuple[AudioTopologyNode, ...]) -> int:
    """Count attractor regions: consecutive nodes within 0.05 distance.

    An attractor is a maximal run of >= 2 consecutive nodes where each
    adjacent pair has distance_2d <= 0.05.
    """
    if len(nodes) < 2:
        return 0
    attractors = 0
    in_attractor = False
    for i in range(len(nodes) - 1):
        dist = _distance_2d(nodes[i].mapping_2d, nodes[i + 1].mapping_2d)
        if dist <= 0.05:
            if not in_attractor:
                attractors += 1
                in_attractor = True
        else:
            in_attractor = False
    return attractors


# ── Recovery path detection ─────────────────────────────────────────────────


def detect_recovery_path(report: AudioLineageReport) -> bool:
    """Detect a stable → collapse → recovery arc.

    A recovery path exists if:
      - coherence decreases then later increases
      - cluster tightness decreases at some point
      - final coherence > first coherence
    """
    nodes = report.nodes
    if len(nodes) < 3:
        return False

    coherences = [n.coherence_score for n in nodes]
    tightnesses = [n.cluster_tightness for n in nodes]

    # Check: final coherence > first coherence
    if coherences[-1] <= coherences[0]:
        return False

    # Check: coherence decreases then later increases
    found_decrease = False
    found_increase_after = False
    for i in range(1, len(coherences)):
        if coherences[i] < coherences[i - 1]:
            found_decrease = True
        elif found_decrease and coherences[i] > coherences[i - 1]:
            found_increase_after = True
            break

    if not (found_decrease and found_increase_after):
        return False

    # Check: cluster tightness decreases at some point
    tightness_decreased = any(
        tightnesses[i] < tightnesses[i - 1]
        for i in range(1, len(tightnesses))
    )

    return tightness_decreased


# ── Lineage classification ──────────────────────────────────────────────────


def classify_lineage(report: AudioLineageReport) -> str:
    """Classify the overall lineage trajectory.

    Returns one of:
      "stable_basin"
      "collapse_recovery"
      "drifting"
      "multi_attractor"
      "chaotic"
    """
    if report.recovery_path_detected:
        return "collapse_recovery"

    if len(report.nodes) <= 1:
        return "stable_basin"

    if report.attractor_count >= 3:
        return "multi_attractor"

    if report.basin_switch_count == 0 and report.attractor_count >= 1:
        return "stable_basin"

    # Chaotic: many basin switches relative to transitions
    n_transitions = len(report.transitions)
    if n_transitions > 0 and report.basin_switch_count / n_transitions > 0.6:
        return "chaotic"

    return "drifting"


# ── Main entry point ────────────────────────────────────────────────────────


def build_audio_lineage(paths: list) -> AudioLineageReport:
    """Build a full temporal topology lineage from ordered audio paths.

    Parameters
    ----------
    paths : list[str]
        Ordered list of audio file paths.  The sequence order is
        preserved exactly — this defines the temporal trajectory.

    Returns
    -------
    AudioLineageReport
        Frozen, deterministic lineage report with nodes, transitions,
        basin switches, attractors, recovery detection, and classification.
    """
    if len(paths) == 0:
        return AudioLineageReport(
            nodes=(),
            transitions=(),
            basin_switch_count=0,
            attractor_count=0,
            recovery_path_detected=False,
            lineage_label="stable_basin",
        )

    # Analyze each file and convert to topology nodes (preserve order)
    reports = [analyze_quantum_audio_file(p) for p in paths]
    nodes = tuple(_report_to_node(r) for r in reports)

    # Compute ordered transitions: node_i → node_(i+1)
    transitions = tuple(
        _compute_transition(nodes[i], nodes[i + 1])
        for i in range(len(nodes) - 1)
    )

    # Build preliminary report for detection functions
    preliminary = AudioLineageReport(
        nodes=nodes,
        transitions=transitions,
        basin_switch_count=0,
        attractor_count=0,
        recovery_path_detected=False,
        lineage_label="",
    )

    basin_switches = detect_basin_switches(preliminary)
    attractor_count = _count_attractors(nodes)

    # Build report with counts for recovery detection
    with_counts = AudioLineageReport(
        nodes=nodes,
        transitions=transitions,
        basin_switch_count=basin_switches,
        attractor_count=attractor_count,
        recovery_path_detected=False,
        lineage_label="",
    )

    recovery = detect_recovery_path(with_counts)

    # Final report with all fields
    final = AudioLineageReport(
        nodes=nodes,
        transitions=transitions,
        basin_switch_count=basin_switches,
        attractor_count=attractor_count,
        recovery_path_detected=recovery,
        lineage_label="",
    )

    label = classify_lineage(final)

    return AudioLineageReport(
        nodes=nodes,
        transitions=transitions,
        basin_switch_count=basin_switches,
        attractor_count=attractor_count,
        recovery_path_detected=recovery,
        lineage_label=label,
    )


# ── Node/report construction from raw values (for programmatic use) ─────────


def build_node(
    filename: str,
    coherence_score: float,
    spectral_entropy: float,
    harmonic_density: float,
    cluster_tightness: float,
    mapping_2d: Tuple[float, float],
    topology_label: str,
) -> AudioTopologyNode:
    """Construct an AudioTopologyNode from raw values."""
    return AudioTopologyNode(
        filename=filename,
        coherence_score=coherence_score,
        spectral_entropy=spectral_entropy,
        harmonic_density=harmonic_density,
        cluster_tightness=cluster_tightness,
        mapping_2d=mapping_2d,
        topology_label=topology_label,
    )


def build_lineage_from_nodes(
    nodes: Tuple[AudioTopologyNode, ...],
) -> AudioLineageReport:
    """Build a lineage report from pre-constructed nodes.

    Useful for programmatic construction and testing without file I/O.
    """
    transitions = tuple(
        _compute_transition(nodes[i], nodes[i + 1])
        for i in range(len(nodes) - 1)
    )

    preliminary = AudioLineageReport(
        nodes=nodes,
        transitions=transitions,
        basin_switch_count=0,
        attractor_count=0,
        recovery_path_detected=False,
        lineage_label="",
    )

    basin_switches = detect_basin_switches(preliminary)
    attractor_count = _count_attractors(nodes)

    with_counts = AudioLineageReport(
        nodes=nodes,
        transitions=transitions,
        basin_switch_count=basin_switches,
        attractor_count=attractor_count,
        recovery_path_detected=False,
        lineage_label="",
    )

    recovery = detect_recovery_path(with_counts)

    final = AudioLineageReport(
        nodes=nodes,
        transitions=transitions,
        basin_switch_count=basin_switches,
        attractor_count=attractor_count,
        recovery_path_detected=recovery,
        lineage_label="",
    )

    label = classify_lineage(final)

    return AudioLineageReport(
        nodes=nodes,
        transitions=transitions,
        basin_switch_count=basin_switches,
        attractor_count=attractor_count,
        recovery_path_detected=recovery,
        lineage_label=label,
    )
