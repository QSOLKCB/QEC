"""v102.6.0 — Phase space mapping and attractor detection.

Interprets transition graph structure to identify:
- attractors (stable states with strong self-loops)
- basins (regions of stability with high in-degree)
- escape dynamics (outgoing transition rates)
- phase classifications (strong/weak attractor, basin, transient, neutral)

All functions are:
- deterministic (identical inputs -> identical outputs)
- side-effect free (no mutation of inputs)
- bounded outputs

Dependencies: stdlib only.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple


def detect_attractors(
    graph: Dict[Tuple[str, str], int],
    node_stats: Dict[str, Dict[str, int]],
) -> Dict[str, Dict[str, Any]]:
    """Detect attractor states in the transition graph.

    A node is an attractor if it has at least one self-loop and its
    attractor score (self_loop_count + in_degree - out_degree) is positive.

    Parameters
    ----------
    graph : dict
        Maps ``(source_type, target_type)`` tuples to transition counts.
        Output of ``build_transition_graph``.
    node_stats : dict
        Per-node statistics from ``compute_node_stats``.

    Returns
    -------
    dict
        Keyed by type name.  Each value contains:

        - ``is_attractor`` : bool
        - ``score`` : float — rounded to 12 decimals
    """
    result: Dict[str, Dict[str, Any]] = {}

    for node in sorted(node_stats.keys()):
        stats = node_stats[node]
        self_loop_count = graph.get((node, node), 0)
        in_degree = stats["in_degree"]
        out_degree = stats["out_degree"]

        score = round(
            float(self_loop_count + in_degree - out_degree), 12
        )

        is_attractor = self_loop_count >= 1 and score > 0

        result[node] = {
            "is_attractor": is_attractor,
            "score": score,
        }

    return result


def detect_basins(
    graph: Dict[Tuple[str, str], int],
    node_stats: Dict[str, Dict[str, int]],
) -> Dict[str, Dict[str, Any]]:
    """Detect basin characteristics for each node.

    Parameters
    ----------
    graph : dict
        Transition graph (unused, accepted for interface consistency).
    node_stats : dict
        Per-node statistics from ``compute_node_stats``.

    Returns
    -------
    dict
        Keyed by type name.  Each value contains:

        - ``basin_size`` : int — proxy based on in_degree
        - ``basin_strength`` : float — in_degree / (1 + out_degree)
    """
    result: Dict[str, Dict[str, Any]] = {}

    for node in sorted(node_stats.keys()):
        stats = node_stats[node]
        in_degree = stats["in_degree"]
        out_degree = stats["out_degree"]

        basin_size = in_degree
        basin_strength = round(float(in_degree / (1 + out_degree)), 12)

        result[node] = {
            "basin_size": basin_size,
            "basin_strength": basin_strength,
        }

    return result


def detect_escape_dynamics(
    graph: Dict[Tuple[str, str], int],
    node_stats: Dict[str, Dict[str, int]],
) -> Dict[str, Dict[str, Any]]:
    """Detect escape dynamics for each node.

    Parameters
    ----------
    graph : dict
        Transition graph (unused, accepted for interface consistency).
    node_stats : dict
        Per-node statistics from ``compute_node_stats``.

    Returns
    -------
    dict
        Keyed by type name.  Each value contains:

        - ``escape_rate`` : float — out_degree / (1 + in_degree)
    """
    result: Dict[str, Dict[str, Any]] = {}

    for node in sorted(node_stats.keys()):
        stats = node_stats[node]
        in_degree = stats["in_degree"]
        out_degree = stats["out_degree"]

        escape_rate = round(float(out_degree / (1 + in_degree)), 12)

        result[node] = {
            "escape_rate": escape_rate,
        }

    return result


def classify_phase_state(
    attractors: Dict[str, Dict[str, Any]],
    basins: Dict[str, Dict[str, Any]],
    escape: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Classify each node into a phase state.

    Classification rules (in priority order):

    A. ``strong_attractor`` — is_attractor and basin_strength > 1
    B. ``weak_attractor`` — is_attractor and basin_strength <= 1
    C. ``basin`` — not attractor and basin_strength > 0.5
    D. ``transient`` — escape_rate > 0.5
    E. ``neutral`` — fallback

    Parameters
    ----------
    attractors : dict
        Output of ``detect_attractors``.
    basins : dict
        Output of ``detect_basins``.
    escape : dict
        Output of ``detect_escape_dynamics``.

    Returns
    -------
    dict
        Keyed by type name.  Each value contains:

        - ``phase`` : str — one of the five classifications
    """
    all_nodes = sorted(
        set(list(attractors.keys()) + list(basins.keys()) + list(escape.keys()))
    )

    result: Dict[str, Dict[str, Any]] = {}

    for node in all_nodes:
        att = attractors.get(node, {"is_attractor": False, "score": 0.0})
        bas = basins.get(node, {"basin_size": 0, "basin_strength": 0.0})
        esc = escape.get(node, {"escape_rate": 0.0})

        is_attractor = att["is_attractor"]
        basin_strength = bas["basin_strength"]
        escape_rate = esc["escape_rate"]

        if is_attractor and basin_strength > 1:
            phase = "strong_attractor"
        elif is_attractor and basin_strength <= 1:
            phase = "weak_attractor"
        elif not is_attractor and basin_strength > 0.5:
            phase = "basin"
        elif escape_rate > 0.5:
            phase = "transient"
        else:
            phase = "neutral"

        result[node] = {"phase": phase}

    return result


__all__ = [
    "detect_attractors",
    "detect_basins",
    "detect_escape_dynamics",
    "classify_phase_state",
]
