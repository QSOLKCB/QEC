"""v87.2.0 — Motif Transition Graph (State Graph of Trajectory Dynamics).

Builds a deterministic state graph from ternary trajectory series:
  - Nodes = unique states (or motifs)
  - Edges = transitions with frequency counts
  - Normalized edge weights (transition probabilities)
  - Graph-level metrics (degree, entropy)

Layer 5 — Experiments.
Does not modify decoder internals.  Fully deterministic.  Read-only on inputs.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Step 1 — State Transition Extraction
# ---------------------------------------------------------------------------


def extract_state_transitions(
    series: List[Tuple[int, ...]],
) -> Dict[str, Any]:
    """Extract transition counts from consecutive state pairs.

    For each consecutive pair (p_i, p_{i+1}), count occurrences.

    Returns
    -------
    dict with ``transitions``: list of ``{"from": p, "to": q, "count": n}``,
    sorted by (from, to) for deterministic ordering.
    """
    counts: Dict[Tuple[Tuple[int, ...], Tuple[int, ...]], int] = {}
    for i in range(len(series) - 1):
        key = (series[i], series[i + 1])
        counts[key] = counts.get(key, 0) + 1

    transitions = [
        {"from": k[0], "to": k[1], "count": v}
        for k, v in sorted(counts.items())
    ]
    return {"transitions": transitions}


# ---------------------------------------------------------------------------
# Step 2 — Build State Graph
# ---------------------------------------------------------------------------


def build_state_graph(
    transitions: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Build a state graph from transition list.

    Returns
    -------
    dict with ``nodes`` (sorted unique states), ``edges`` (transitions),
    ``n_nodes``, ``n_edges``.
    """
    node_set: set[Tuple[int, ...]] = set()
    for t in transitions:
        node_set.add(t["from"])
        node_set.add(t["to"])
    nodes = sorted(node_set)

    return {
        "nodes": nodes,
        "edges": transitions,
        "n_nodes": len(nodes),
        "n_edges": len(transitions),
    }


# ---------------------------------------------------------------------------
# Step 3 — Normalize Edge Weights
# ---------------------------------------------------------------------------


def normalize_transition_weights(
    edges: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Normalize edge counts to transition probabilities per source node.

    For each source node, prob = count / sum(outgoing counts).

    Returns
    -------
    List of edges with added ``prob`` field.
    """
    # Sum outgoing counts per source.
    outgoing: Dict[Tuple[int, ...], int] = {}
    for e in edges:
        src = e["from"]
        outgoing[src] = outgoing.get(src, 0) + e["count"]

    normalized = []
    for e in edges:
        total = outgoing[e["from"]]
        normalized.append({
            "from": e["from"],
            "to": e["to"],
            "count": e["count"],
            "prob": e["count"] / total if total > 0 else 0.0,
        })
    return normalized


# ---------------------------------------------------------------------------
# Step 4 — Motif-Level Graph
# ---------------------------------------------------------------------------


def _match_motif_index(
    series: List[Tuple[int, ...]],
    pos: int,
    motifs: List[Dict[str, Any]],
) -> Optional[int]:
    """Return the index of the first motif whose pattern matches at *pos*."""
    for idx, m in enumerate(motifs):
        pat = m["pattern"]
        end = pos + len(pat)
        if end <= len(series) and list(series[pos:end]) == list(pat):
            return idx
    return None


def build_motif_graph(
    series: List[Tuple[int, ...]],
    motifs: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Build a transition graph at the motif level.

    Maps series positions to motif indices (first match, greedy scan),
    then builds transitions between consecutive matched motifs.

    Returns empty graph when *motifs* is empty or no matches found.
    """
    if not motifs or not series:
        return {"motif_nodes": [], "motif_edges": []}

    # Greedy scan: match motifs along the series.
    matched_indices: List[int] = []
    pos = 0
    while pos < len(series):
        idx = _match_motif_index(series, pos, motifs)
        if idx is not None:
            matched_indices.append(idx)
            pos += len(motifs[idx]["pattern"])
        else:
            pos += 1

    if len(matched_indices) < 2:
        nodes = sorted(set(matched_indices))
        return {"motif_nodes": nodes, "motif_edges": []}

    # Count transitions between motif indices.
    counts: Dict[Tuple[int, int], int] = {}
    for i in range(len(matched_indices) - 1):
        key = (matched_indices[i], matched_indices[i + 1])
        counts[key] = counts.get(key, 0) + 1

    motif_edges = [
        {"from": k[0], "to": k[1], "count": v}
        for k, v in sorted(counts.items())
    ]
    motif_nodes = sorted(set(matched_indices))
    return {"motif_nodes": motif_nodes, "motif_edges": motif_edges}


# ---------------------------------------------------------------------------
# Step 5 — Graph Metrics
# ---------------------------------------------------------------------------


def compute_graph_metrics(
    graph: Dict[str, Any],
) -> Dict[str, Any]:
    """Compute degree and entropy metrics for a state graph.

    Returns
    -------
    dict with ``max_out_degree``, ``mean_out_degree``, ``transition_entropy``.
    """
    edges = graph.get("edges", [])
    nodes = graph.get("nodes", [])

    if not nodes:
        return {
            "max_out_degree": 0,
            "mean_out_degree": 0.0,
            "transition_entropy": 0.0,
        }

    # Out-degree per node.
    out_degree: Dict[Tuple[int, ...], int] = {n: 0 for n in nodes}
    for e in edges:
        src = e["from"]
        if src in out_degree:
            out_degree[src] += 1

    degrees = list(out_degree.values())
    max_out = max(degrees)
    mean_out = sum(degrees) / len(degrees)

    # Transition entropy from normalized probabilities.
    normalized = normalize_transition_weights(edges)
    entropy = 0.0
    for e in normalized:
        p = e["prob"]
        if p > 0.0:
            entropy -= p * math.log(p)

    return {
        "max_out_degree": max_out,
        "mean_out_degree": mean_out,
        "transition_entropy": entropy,
    }


# ---------------------------------------------------------------------------
# Step 6 — Full Pipeline
# ---------------------------------------------------------------------------


def run_motif_graph_analysis(
    series: List[Tuple[int, ...]],
    motifs: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Run full motif transition graph analysis.

    Parameters
    ----------
    series:
        Ordered list of ternary-encoded tuples.
    motifs:
        Optional list of motif dicts with ``pattern`` and ``count`` keys.

    Returns
    -------
    dict with ``state_graph``, ``normalized_edges``, ``metrics``,
    ``motif_graph``.
    """
    result = extract_state_transitions(series)
    state_graph = build_state_graph(result["transitions"])
    normalized = normalize_transition_weights(state_graph["edges"])
    metrics = compute_graph_metrics(state_graph)

    motif_graph: Dict[str, Any] = {"motif_nodes": [], "motif_edges": []}
    if motifs:
        motif_graph = build_motif_graph(series, motifs)

    return {
        "state_graph": state_graph,
        "normalized_edges": normalized,
        "metrics": metrics,
        "motif_graph": motif_graph,
    }
