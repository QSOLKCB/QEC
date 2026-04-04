"""v137.0.14 — E8 Triality Traversal + Path Mesh.

Theory-coupled traversal mesh absorbing real semantics from:

  papers/Sound_as_a_Fractal_Golden_E8_Dimension_i.pdf
  qec_theory_diagram.txt
  papers/Comparative_Cosmology_Layer_Mapping_Harm.pdf

Core theory constructs absorbed:

  1. E8 triality topology — deterministic triality lane classification
     Each node classified into one of five axis classes derived from E8
     triality structure: PRIMARY_AXIS, DUAL_AXIS, BOUNDARY_AXIS,
     RESONANCE_LINK, TOROIDAL_RETURN.

  2. Ouroboros feedback loop — deterministic cyclic return paths
     Graph supports loopback edges forming self-consistent cycles.
     Each node carries a loopback_index for cycle membership.

  3. Phi-weighted path distance — golden ratio traversal cost
     Edge cost follows: weighted_cost = edge_count * PHI + resonance_penalty
     Absorbs golden weighting from phi-shell progression theory.

Pipeline law:

  world
  -> camera
  -> projection
  -> lighting
  -> phi-shell rasterization
  -> traversal mesh
  -> replay-safe path ledger

Layer 4 -- Analysis.
Does not import or modify decoder internals.
Fully deterministic: no randomness, no global state, no input mutation.

Theory Upgrade Source:
- file: papers/Sound_as_a_Fractal_Golden_E8_Dimension_i.pdf
- concept: E8 triality axis classification, ouroboros cyclic return, phi-weighted path cost
- implementation: triality-classified path mesh with phi-weighted traversal decisions
- invariant tested: E8_TRIALITY_LOCK, OUROBOROS_FEEDBACK_LOOP, PHI_PATH_WEIGHT
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

# ---------------------------------------------------------------------------
# Version
# ---------------------------------------------------------------------------

RETRO_TRAVERSAL_VERSION: str = "v137.0.14"

# ---------------------------------------------------------------------------
# Constants -- E8 triality axis classes (theory-coupled)
# ---------------------------------------------------------------------------
# From Sound_as_a_Fractal_Golden_E8_Dimension_i.pdf:
# E8 triality structure mandates three primary axes plus two boundary
# link types.  Classification is deterministic via node_index % 5.

PRIMARY_AXIS: str = "PRIMARY_AXIS"
DUAL_AXIS: str = "DUAL_AXIS"
BOUNDARY_AXIS: str = "BOUNDARY_AXIS"
RESONANCE_LINK: str = "RESONANCE_LINK"
TOROIDAL_RETURN: str = "TOROIDAL_RETURN"

VALID_AXIS_CLASSES: Tuple[str, ...] = (
    PRIMARY_AXIS,
    DUAL_AXIS,
    BOUNDARY_AXIS,
    RESONANCE_LINK,
    TOROIDAL_RETURN,
)

# ---------------------------------------------------------------------------
# Constants -- phi-weighted path cost (theory-coupled)
# ---------------------------------------------------------------------------
# From Sound_as_a_Fractal_Golden_E8_Dimension_i.pdf:
# Golden ratio weighting for traversal cost.

PHI_PATH_WEIGHT: float = 1.618

# ---------------------------------------------------------------------------
# Constants -- edge classes
# ---------------------------------------------------------------------------

FORWARD_EDGE: str = "FORWARD_EDGE"
LOOPBACK_EDGE: str = "LOOPBACK_EDGE"

VALID_EDGE_CLASSES: Tuple[str, ...] = (
    FORWARD_EDGE,
    LOOPBACK_EDGE,
)

# ---------------------------------------------------------------------------
# Constants -- path classes
# ---------------------------------------------------------------------------

LINEAR_PATH: str = "LINEAR_PATH"
CYCLIC_PATH: str = "CYCLIC_PATH"

VALID_PATH_CLASSES: Tuple[str, ...] = (
    LINEAR_PATH,
    CYCLIC_PATH,
)

# ---------------------------------------------------------------------------
# Constants -- defaults
# ---------------------------------------------------------------------------

FLOAT_PRECISION: int = 12

# ---------------------------------------------------------------------------
# Frozen dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RetroPathNode:
    """Immutable path mesh node with triality axis classification."""

    node_index: int
    axis_class: str
    loopback_index: int
    resonance_weight: float
    stable_hash: str
    version: str = RETRO_TRAVERSAL_VERSION


@dataclass(frozen=True)
class RetroPathEdge:
    """Immutable path mesh edge with phi-weighted cost."""

    source_index: int
    target_index: int
    edge_cost: float
    edge_class: str
    stable_hash: str
    version: str = RETRO_TRAVERSAL_VERSION


@dataclass(frozen=True)
class RetroTraversalDecision:
    """Immutable traversal decision artifact."""

    path_nodes: Tuple[RetroPathNode, ...]
    path_edges: Tuple[RetroPathEdge, ...]
    total_cost: float
    path_class: str
    symbolic_trace: str
    stable_hash: str
    version: str = RETRO_TRAVERSAL_VERSION


@dataclass(frozen=True)
class RetroTraversalLedger:
    """Immutable ledger of traversal decisions."""

    decisions: Tuple[RetroTraversalDecision, ...]
    decision_count: int
    stable_hash: str
    version: str = RETRO_TRAVERSAL_VERSION


# ---------------------------------------------------------------------------
# Helpers -- canonical JSON & hashing
# ---------------------------------------------------------------------------


def _canonical_json(obj: Any) -> str:
    """Produce canonical JSON: sorted keys, compact separators."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True)


def _round(value: float) -> float:
    """Round to canonical precision for deterministic hashing."""
    return round(value, FLOAT_PRECISION)


# ---------------------------------------------------------------------------
# Helpers -- hashing
# ---------------------------------------------------------------------------


def _compute_node_hash(
    node_index: int,
    axis_class: str,
    loopback_index: int,
    resonance_weight: float,
) -> str:
    """SHA-256 of canonical JSON of a path node."""
    payload = {
        "axis_class": axis_class,
        "loopback_index": loopback_index,
        "node_index": node_index,
        "resonance_weight": _round(resonance_weight),
        "version": RETRO_TRAVERSAL_VERSION,
    }
    return hashlib.sha256(
        _canonical_json(payload).encode("utf-8")
    ).hexdigest()


def _compute_edge_hash(
    source_index: int,
    target_index: int,
    edge_cost: float,
    edge_class: str,
) -> str:
    """SHA-256 of canonical JSON of a path edge."""
    payload = {
        "edge_class": edge_class,
        "edge_cost": _round(edge_cost),
        "source_index": source_index,
        "target_index": target_index,
        "version": RETRO_TRAVERSAL_VERSION,
    }
    return hashlib.sha256(
        _canonical_json(payload).encode("utf-8")
    ).hexdigest()


def _compute_decision_hash(
    node_hashes: Tuple[str, ...],
    edge_hashes: Tuple[str, ...],
    total_cost: float,
    path_class: str,
    symbolic_trace: str,
) -> str:
    """SHA-256 of canonical JSON of a traversal decision."""
    payload = {
        "edge_hashes": list(edge_hashes),
        "node_hashes": list(node_hashes),
        "path_class": path_class,
        "symbolic_trace": symbolic_trace,
        "total_cost": _round(total_cost),
        "version": RETRO_TRAVERSAL_VERSION,
    }
    return hashlib.sha256(
        _canonical_json(payload).encode("utf-8")
    ).hexdigest()


def _compute_ledger_hash(
    decisions: Tuple[RetroTraversalDecision, ...],
) -> str:
    """SHA-256 of ordered decision hashes."""
    hashes = [d.stable_hash for d in decisions]
    return hashlib.sha256(
        _canonical_json(hashes).encode("utf-8")
    ).hexdigest()


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def _validate_positive_int(value: Any, field_name: str) -> int:
    """Validate a positive integer (not bool)."""
    if isinstance(value, bool):
        raise TypeError(f"{field_name} must be int, got bool")
    if not isinstance(value, int):
        raise TypeError(
            f"{field_name} must be int, got {type(value).__name__}"
        )
    if value < 1:
        raise ValueError(f"{field_name} must be >= 1, got {value}")
    return value


def _validate_non_negative_int(value: Any, field_name: str) -> int:
    """Validate a non-negative integer (not bool)."""
    if isinstance(value, bool):
        raise TypeError(f"{field_name} must be int, got bool")
    if not isinstance(value, int):
        raise TypeError(
            f"{field_name} must be int, got {type(value).__name__}"
        )
    if value < 0:
        raise ValueError(f"{field_name} must be >= 0, got {value}")
    return value


def _validate_non_negative_float(value: Any, field_name: str) -> float:
    """Validate a non-negative finite float."""
    if isinstance(value, bool):
        raise TypeError(f"{field_name} must be numeric, got bool")
    if not isinstance(value, (int, float)):
        raise TypeError(
            f"{field_name} must be numeric, got {type(value).__name__}"
        )
    fv = float(value)
    if not math.isfinite(fv):
        raise ValueError(f"{field_name} must be finite, got {fv}")
    if fv < 0.0:
        raise ValueError(f"{field_name} must be >= 0, got {fv}")
    return fv


# ---------------------------------------------------------------------------
# Core functions -- triality axis classification
# ---------------------------------------------------------------------------


def classify_triality_axis(node_index: int) -> str:
    """Classify a node into an E8 triality axis class.

    Deterministic classification rule: ``node_index % 5`` maps to:

        0 -> PRIMARY_AXIS
        1 -> DUAL_AXIS
        2 -> BOUNDARY_AXIS
        3 -> RESONANCE_LINK
        4 -> TOROIDAL_RETURN

    This mirrors the E8 triality structure from the theory corpus:
    three primary axes (PRIMARY, DUAL, BOUNDARY) plus two boundary
    link types (RESONANCE_LINK, TOROIDAL_RETURN).

    Args:
        node_index: Non-negative integer node index.

    Returns:
        Axis class string.

    Raises:
        TypeError: If node_index is not int or is bool.
        ValueError: If node_index is negative.
    """
    idx = _validate_non_negative_int(node_index, "node_index")
    return VALID_AXIS_CLASSES[idx % 5]


# ---------------------------------------------------------------------------
# Core functions -- phi-weighted path cost
# ---------------------------------------------------------------------------


def compute_phi_weighted_path_cost(
    edge_count: int,
    resonance_penalty: float = 0.0,
) -> float:
    """Compute phi-weighted traversal cost.

    From theory corpus (Sound_as_a_Fractal_Golden_E8_Dimension_i.pdf):
    Golden ratio weighting for path traversal cost.

    Formula:
        weighted_cost = edge_count * PHI_PATH_WEIGHT + resonance_penalty

    Rounded to canonical precision for deterministic hashing.

    Args:
        edge_count: Number of edges in path (must be >= 1).
        resonance_penalty: Additional penalty (must be >= 0).

    Returns:
        Deterministic phi-weighted cost value.

    Raises:
        TypeError: If inputs are wrong type.
        ValueError: If inputs are out of range.
    """
    ec = _validate_positive_int(edge_count, "edge_count")
    rp = _validate_non_negative_float(resonance_penalty, "resonance_penalty")
    cost = ec * PHI_PATH_WEIGHT + rp
    return _round(cost)


# ---------------------------------------------------------------------------
# Core functions -- resonance weight
# ---------------------------------------------------------------------------


def _compute_resonance_weight(_node_index: int, axis_class: str) -> float:
    """Compute deterministic resonance weight for a node.

    RESONANCE_LINK nodes receive weight PHI_PATH_WEIGHT.
    TOROIDAL_RETURN nodes receive weight PHI_PATH_WEIGHT * 0.5.
    All other nodes receive weight 0.0.

    The ``_node_index`` parameter is intentionally unused and retained for
    signature compatibility. Resonance weight is currently a pure function
    of ``axis_class``.
    """
    if axis_class == RESONANCE_LINK:
        return _round(PHI_PATH_WEIGHT)
    if axis_class == TOROIDAL_RETURN:
        return _round(PHI_PATH_WEIGHT * 0.5)
    return 0.0


# ---------------------------------------------------------------------------
# Core functions -- loopback index
# ---------------------------------------------------------------------------


def _compute_loopback_index(
    node_index: int,
    node_count: int,
    allow_loopback: bool,
) -> int:
    """Compute deterministic loopback index for ouroboros cycle.

    For TOROIDAL_RETURN nodes (node_index % 5 == 4), the loopback
    target is the nearest preceding PRIMARY_AXIS node (node_index % 5 == 0).

    If allow_loopback is False, or the node is not TOROIDAL_RETURN,
    loopback_index is -1 (no loopback).

    Args:
        node_index: The node's index.
        node_count: Total number of nodes in the mesh.
        allow_loopback: Whether loopback edges are allowed.

    Returns:
        Target node index for loopback, or -1 if none.
    """
    if not allow_loopback:
        return -1
    if node_index % 5 != 4:
        return -1
    # Find nearest preceding PRIMARY_AXIS node (idx % 5 == 0)
    # For node_index=4, target=0; for node_index=9, target=5; etc.
    target = node_index - (node_index % 5)
    if target < 0 or target >= node_count:
        return -1
    return target


# ---------------------------------------------------------------------------
# Core functions -- build path mesh
# ---------------------------------------------------------------------------


def build_path_mesh(
    node_count: int,
    allow_loopback: bool = True,
) -> Tuple[Tuple[RetroPathNode, ...], Tuple[RetroPathEdge, ...]]:
    """Build a deterministic E8 triality path mesh.

    Constructs nodes with triality axis classification and edges with
    phi-weighted cost.  If allow_loopback is True, TOROIDAL_RETURN
    nodes generate loopback edges to their nearest preceding
    PRIMARY_AXIS node (ouroboros feedback loop).

    Args:
        node_count: Number of nodes in the mesh (must be >= 1).
        allow_loopback: Whether to include loopback edges.

    Returns:
        Tuple of (nodes, edges) where each is a tuple of frozen
        dataclass instances.

    Raises:
        TypeError: If node_count is not int or is bool.
        ValueError: If node_count is < 1.
    """
    nc = _validate_positive_int(node_count, "node_count")

    # Build nodes
    nodes: List[RetroPathNode] = []
    for i in range(nc):
        axis_class = classify_triality_axis(i)
        loopback_idx = _compute_loopback_index(i, nc, allow_loopback)
        res_weight = _compute_resonance_weight(i, axis_class)
        h = _compute_node_hash(i, axis_class, loopback_idx, res_weight)
        nodes.append(RetroPathNode(
            node_index=i,
            axis_class=axis_class,
            loopback_index=loopback_idx,
            resonance_weight=res_weight,
            stable_hash=h,
        ))

    # Build edges: forward edges between consecutive nodes
    edges: List[RetroPathEdge] = []
    for i in range(nc - 1):
        cost = compute_phi_weighted_path_cost(1, nodes[i].resonance_weight)
        h = _compute_edge_hash(i, i + 1, cost, FORWARD_EDGE)
        edges.append(RetroPathEdge(
            source_index=i,
            target_index=i + 1,
            edge_cost=cost,
            edge_class=FORWARD_EDGE,
            stable_hash=h,
        ))

    # Build loopback edges for TOROIDAL_RETURN nodes
    if allow_loopback:
        for node in nodes:
            if node.loopback_index >= 0:
                # Loopback edge from this node back to target
                cost = compute_phi_weighted_path_cost(
                    1, node.resonance_weight,
                )
                h = _compute_edge_hash(
                    node.node_index, node.loopback_index,
                    cost, LOOPBACK_EDGE,
                )
                edges.append(RetroPathEdge(
                    source_index=node.node_index,
                    target_index=node.loopback_index,
                    edge_cost=cost,
                    edge_class=LOOPBACK_EDGE,
                    stable_hash=h,
                ))

    return tuple(nodes), tuple(edges)


# ---------------------------------------------------------------------------
# Core functions -- symbolic trace
# ---------------------------------------------------------------------------


def _build_symbolic_trace(nodes: Tuple[RetroPathNode, ...]) -> str:
    """Build symbolic trace from axis classes of path nodes.

    Produces a deduplicated transition trace, e.g.:
        PRIMARY_AXIS -> DUAL_AXIS -> TOROIDAL_RETURN

    Consecutive duplicates are collapsed.

    Args:
        nodes: Tuple of RetroPathNode objects.

    Returns:
        Symbolic trace string.
    """
    if len(nodes) == 0:
        return ""
    classes: List[str] = []
    for n in nodes:
        if len(classes) == 0 or classes[-1] != n.axis_class:
            classes.append(n.axis_class)
    return " -> ".join(classes)


# ---------------------------------------------------------------------------
# Core functions -- traversal decision
# ---------------------------------------------------------------------------


def build_traversal_decision(
    node_count: int,
    allow_loopback: bool = True,
) -> RetroTraversalDecision:
    """Build a complete traversal decision artifact.

    Constructs path mesh, computes total phi-weighted cost, classifies
    the path as LINEAR_PATH or CYCLIC_PATH, and builds symbolic trace.

    Args:
        node_count: Number of nodes (must be >= 1).
        allow_loopback: Whether to include loopback edges.

    Returns:
        Frozen RetroTraversalDecision artifact.

    Raises:
        TypeError: If node_count is wrong type.
        ValueError: If node_count is out of range.
    """
    nc = _validate_positive_int(node_count, "node_count")

    nodes, edges = build_path_mesh(nc, allow_loopback)

    # Compute total cost
    total_cost = _round(sum(e.edge_cost for e in edges)) if edges else 0.0

    # Classify path
    has_loopback = any(e.edge_class == LOOPBACK_EDGE for e in edges)
    path_class = CYCLIC_PATH if has_loopback else LINEAR_PATH

    # Symbolic trace
    trace = _build_symbolic_trace(nodes)

    # Hash
    node_hashes = tuple(n.stable_hash for n in nodes)
    edge_hashes = tuple(e.stable_hash for e in edges)
    h = _compute_decision_hash(
        node_hashes, edge_hashes, total_cost, path_class, trace,
    )

    return RetroTraversalDecision(
        path_nodes=nodes,
        path_edges=edges,
        total_cost=total_cost,
        path_class=path_class,
        symbolic_trace=trace,
        stable_hash=h,
    )


# ---------------------------------------------------------------------------
# Core functions -- traversal ledger
# ---------------------------------------------------------------------------


def build_traversal_ledger(
    decisions: Tuple[RetroTraversalDecision, ...],
) -> RetroTraversalLedger:
    """Build an immutable ledger of traversal decisions.

    Args:
        decisions: Tuple of RetroTraversalDecision objects.

    Returns:
        Frozen RetroTraversalLedger artifact.

    Raises:
        TypeError: If decisions is not a tuple of RetroTraversalDecision.
        ValueError: If decisions is empty.
    """
    if not isinstance(decisions, tuple):
        raise TypeError(
            f"decisions must be a tuple, got {type(decisions).__name__}"
        )
    if len(decisions) == 0:
        raise ValueError("decisions must not be empty")
    for i, d in enumerate(decisions):
        if not isinstance(d, RetroTraversalDecision):
            raise TypeError(
                f"decisions[{i}] must be RetroTraversalDecision, "
                f"got {type(d).__name__}"
            )

    h = _compute_ledger_hash(decisions)
    return RetroTraversalLedger(
        decisions=decisions,
        decision_count=len(decisions),
        stable_hash=h,
    )


# ---------------------------------------------------------------------------
# Export -- canonical JSON
# ---------------------------------------------------------------------------


def _node_to_dict(node: RetroPathNode) -> Dict[str, Any]:
    """Convert a path node to a canonical dict."""
    return {
        "axis_class": node.axis_class,
        "loopback_index": node.loopback_index,
        "node_index": node.node_index,
        "resonance_weight": _round(node.resonance_weight),
        "stable_hash": node.stable_hash,
        "version": node.version,
    }


def _edge_to_dict(edge: RetroPathEdge) -> Dict[str, Any]:
    """Convert a path edge to a canonical dict."""
    return {
        "edge_class": edge.edge_class,
        "edge_cost": _round(edge.edge_cost),
        "source_index": edge.source_index,
        "stable_hash": edge.stable_hash,
        "target_index": edge.target_index,
        "version": edge.version,
    }


def _decision_to_dict(decision: RetroTraversalDecision) -> Dict[str, Any]:
    """Convert a traversal decision to a canonical dict."""
    return {
        "path_class": decision.path_class,
        "path_edges": [_edge_to_dict(e) for e in decision.path_edges],
        "path_nodes": [_node_to_dict(n) for n in decision.path_nodes],
        "stable_hash": decision.stable_hash,
        "symbolic_trace": decision.symbolic_trace,
        "total_cost": _round(decision.total_cost),
        "version": decision.version,
    }


def export_traversal_ledger(
    ledger: RetroTraversalLedger,
) -> Dict[str, Any]:
    """Export ledger as canonical dict for JSON serialization.

    Args:
        ledger: The traversal ledger to export.

    Returns:
        Canonical dict suitable for deterministic JSON serialization.
    """
    return {
        "decision_count": ledger.decision_count,
        "decisions": [_decision_to_dict(d) for d in ledger.decisions],
        "stable_hash": ledger.stable_hash,
        "version": ledger.version,
    }


def export_traversal_bundle(
    ledger: RetroTraversalLedger,
) -> str:
    """Export ledger as canonical JSON string with SHA-256 envelope.

    Returns:
        Canonical JSON string. Byte-identical for identical inputs.
    """
    payload = export_traversal_ledger(ledger)
    json_str = _canonical_json(payload)
    envelope = {
        "data": payload,
        "sha256": hashlib.sha256(json_str.encode("utf-8")).hexdigest(),
        "version": RETRO_TRAVERSAL_VERSION,
    }
    return _canonical_json(envelope)
