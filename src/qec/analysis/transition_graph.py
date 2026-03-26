"""v102.5.0 — Transition graph and state flow analysis.

Models how strategies move between taxonomy types across runs by
aggregating individual type trajectories into a global transition graph.

All functions are:
- deterministic (identical inputs -> identical outputs)
- side-effect free (no mutation of inputs)
- bounded outputs

Dependencies: stdlib only.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple


def build_transition_graph(
    trajectories: Dict[str, List[str]],
) -> Dict[Tuple[str, str], int]:
    """Build a global transition graph from per-strategy type trajectories.

    Parameters
    ----------
    trajectories : dict
        Maps strategy names to lists of taxonomy type strings.
        Output of ``build_type_trajectory``.

    Returns
    -------
    dict
        Maps ``(source_type, target_type)`` tuples to transition counts.
        Only edges with count >= 1 are included.
    """
    counts: Dict[Tuple[str, str], int] = {}

    for name in sorted(trajectories.keys()):
        seq = trajectories[name]
        if len(seq) < 2:
            continue
        for i in range(len(seq) - 1):
            edge = (seq[i], seq[i + 1])
            counts[edge] = counts.get(edge, 0) + 1

    return counts


def compute_node_stats(
    graph: Dict[Tuple[str, str], int],
) -> Dict[str, Dict[str, int]]:
    """Compute per-node statistics from the transition graph.

    Parameters
    ----------
    graph : dict
        Output of ``build_transition_graph``.

    Returns
    -------
    dict
        Keyed by type name (sorted).  Each value contains:

        - ``in_degree`` : int — total incoming transition count
        - ``out_degree`` : int — total outgoing transition count
        - ``total_flow`` : int — in_degree + out_degree
    """
    in_counts: Dict[str, int] = {}
    out_counts: Dict[str, int] = {}

    for (src, tgt), count in graph.items():
        out_counts[src] = out_counts.get(src, 0) + count
        in_counts[tgt] = in_counts.get(tgt, 0) + count

    all_nodes = sorted(set(list(in_counts.keys()) + list(out_counts.keys())))

    result: Dict[str, Dict[str, int]] = {}
    for node in all_nodes:
        in_deg = in_counts.get(node, 0)
        out_deg = out_counts.get(node, 0)
        result[node] = {
            "in_degree": in_deg,
            "out_degree": out_deg,
            "total_flow": in_deg + out_deg,
        }

    return result


def rank_transitions(
    graph: Dict[Tuple[str, str], int],
) -> List[Tuple[str, str, int]]:
    """Rank transitions by count, descending.

    Parameters
    ----------
    graph : dict
        Output of ``build_transition_graph``.

    Returns
    -------
    list of (source, target, count)
        Sorted by count descending, then lexicographically by
        (source, target) for determinism.
    """
    entries = [(src, tgt, count) for (src, tgt), count in graph.items()]
    entries.sort(key=lambda x: (-x[2], x[0], x[1]))
    return entries


def extract_dominant_flows(
    graph: Dict[Tuple[str, str], int],
    threshold: int = 2,
) -> List[Tuple[str, str, int]]:
    """Extract transitions with count >= threshold.

    Parameters
    ----------
    graph : dict
        Output of ``build_transition_graph``.
    threshold : int
        Minimum count for inclusion (default 2).

    Returns
    -------
    list of (source, target, count)
        Sorted by count descending, then lexicographically.
    """
    entries = [
        (src, tgt, count)
        for (src, tgt), count in graph.items()
        if count >= threshold
    ]
    entries.sort(key=lambda x: (-x[2], x[0], x[1]))
    return entries


def detect_transition_patterns(
    graph: Dict[Tuple[str, str], int],
) -> Dict[str, Any]:
    """Detect structural patterns in the transition graph.

    Parameters
    ----------
    graph : dict
        Output of ``build_transition_graph``.

    Returns
    -------
    dict
        Contains:

        - ``bidirectional`` : list of (type_a, type_b) — pairs where
          both A->B and B->A exist, sorted lexicographically
        - ``self_loops`` : list of str — types with A->A transitions,
          sorted
        - ``sources`` : list of str — types with out_degree > 0 and
          in_degree == 0, sorted
        - ``sinks`` : list of str — types with in_degree > 0 and
          out_degree == 0, sorted
    """
    node_stats = compute_node_stats(graph)

    # Bidirectional pairs.
    bidirectional: List[Tuple[str, str]] = []
    seen_pairs: set = set()
    for (src, tgt) in graph.keys():
        if src == tgt:
            continue
        pair = tuple(sorted((src, tgt)))
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)
        if (src, tgt) in graph and (tgt, src) in graph:
            bidirectional.append(pair)
    bidirectional.sort()

    # Self-loops.
    self_loops = sorted(
        src for (src, tgt) in graph.keys() if src == tgt
    )

    # Sources: out_degree > 0, in_degree == 0.
    sources = sorted(
        node for node, stats in node_stats.items()
        if stats["out_degree"] > 0 and stats["in_degree"] == 0
    )

    # Sinks: in_degree > 0, out_degree == 0.
    sinks = sorted(
        node for node, stats in node_stats.items()
        if stats["in_degree"] > 0 and stats["out_degree"] == 0
    )

    return {
        "bidirectional": bidirectional,
        "self_loops": self_loops,
        "sources": sources,
        "sinks": sinks,
    }


__all__ = [
    "build_transition_graph",
    "compute_node_stats",
    "rank_transitions",
    "extract_dominant_flows",
    "detect_transition_patterns",
]
