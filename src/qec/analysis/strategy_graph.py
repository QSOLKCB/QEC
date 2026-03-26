"""v103.9.0 — Strategy graph and policy topology analysis.

Models transitions between policies over time, computing:
- policy transition graphs (frequency of A -> B transitions)
- per-policy node metrics (in/out degree, total flow)
- policy stability metrics (self-loop ratio, switch rate, streaks)
- transition pattern detection (bidirectional, sources, sinks)
- topology classification (stable, converging, diverging, cyclic, mixed)

All functions are:
- deterministic (identical inputs -> identical outputs)
- side-effect free (no mutation of inputs)
- bounded outputs

Dependencies: stdlib only.
"""

from __future__ import annotations

from typing import Any, Dict, List, Set, Tuple


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ROUND_PRECISION = 12


def _round(value: float) -> float:
    """Round to ``ROUND_PRECISION`` decimal places."""
    return round(float(value), ROUND_PRECISION)


# ---------------------------------------------------------------------------
# Public API — Build Policy Transition Graph
# ---------------------------------------------------------------------------


def build_policy_graph(
    history: List[str],
) -> Dict[Tuple[str, str], int]:
    """Build a policy transition graph from a sequence of policy selections.

    Parameters
    ----------
    history : list of str
        Ordered sequence of policy names selected over time.
        E.g. ``["stability_first", "balanced", "balanced", "sync_first"]``.

    Returns
    -------
    dict
        Maps ``(source_policy, target_policy)`` tuples to transition counts.
        Only edges with count >= 1 are included.  Self-loops (A -> A) are
        included when the same policy is selected consecutively.
    """
    counts: Dict[Tuple[str, str], int] = {}

    if len(history) < 2:
        return counts

    for i in range(len(history) - 1):
        edge = (history[i], history[i + 1])
        counts[edge] = counts.get(edge, 0) + 1

    return counts


# ---------------------------------------------------------------------------
# Public API — Node Metrics
# ---------------------------------------------------------------------------


def compute_policy_node_stats(
    graph: Dict[Tuple[str, str], int],
) -> Dict[str, Dict[str, int]]:
    """Compute per-policy node statistics from the transition graph.

    Parameters
    ----------
    graph : dict
        Output of ``build_policy_graph``.

    Returns
    -------
    dict
        Keyed by policy name (sorted).  Each value contains:

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


# ---------------------------------------------------------------------------
# Public API — Policy Stability
# ---------------------------------------------------------------------------


def compute_policy_stability(
    history: List[str],
) -> Dict[str, Any]:
    """Compute stability metrics for a policy selection history.

    Parameters
    ----------
    history : list of str
        Ordered sequence of policy names selected over time.

    Returns
    -------
    dict
        Contains:

        - ``self_loop_ratio`` : float — fraction of transitions that are
          self-loops (same policy repeated), in [0.0, 1.0]
        - ``switch_rate`` : float — fraction of transitions that are
          switches (different policy), in [0.0, 1.0]
        - ``dominant_policy`` : str — most frequently selected policy
          (ties broken lexicographically)
        - ``longest_streak`` : int — longest consecutive run of the
          same policy
    """
    if not history:
        return {
            "self_loop_ratio": 0.0,
            "switch_rate": 0.0,
            "dominant_policy": "",
            "longest_streak": 0,
        }

    if len(history) == 1:
        return {
            "self_loop_ratio": 0.0,
            "switch_rate": 0.0,
            "dominant_policy": history[0],
            "longest_streak": 1,
        }

    # Count self-loops and switches.
    self_loops = 0
    switches = 0
    n_transitions = len(history) - 1

    for i in range(n_transitions):
        if history[i] == history[i + 1]:
            self_loops += 1
        else:
            switches += 1

    self_loop_ratio = _round(self_loops / n_transitions)
    switch_rate = _round(switches / n_transitions)

    # Dominant policy (most frequent; ties broken lexicographically).
    counts: Dict[str, int] = {}
    for name in history:
        counts[name] = counts.get(name, 0) + 1
    dominant = max(sorted(counts.keys()), key=lambda n: counts[n])

    # Longest streak.
    longest_streak = 1
    current_streak = 1
    for i in range(1, len(history)):
        if history[i] == history[i - 1]:
            current_streak += 1
            if current_streak > longest_streak:
                longest_streak = current_streak
        else:
            current_streak = 1

    return {
        "self_loop_ratio": self_loop_ratio,
        "switch_rate": switch_rate,
        "dominant_policy": dominant,
        "longest_streak": longest_streak,
    }


# ---------------------------------------------------------------------------
# Public API — Transition Patterns
# ---------------------------------------------------------------------------


def detect_policy_patterns(
    graph: Dict[Tuple[str, str], int],
) -> Dict[str, Any]:
    """Detect structural patterns in the policy transition graph.

    Parameters
    ----------
    graph : dict
        Output of ``build_policy_graph``.

    Returns
    -------
    dict
        Contains:

        - ``bidirectional`` : list of (policy_a, policy_b) — pairs where
          both A->B and B->A exist, sorted lexicographically
        - ``self_loops`` : list of str — policies with A->A transitions,
          sorted
        - ``sources`` : list of str — policies with out_degree > 0 and
          in_degree == 0, sorted
        - ``sinks`` : list of str — policies with in_degree > 0 and
          out_degree == 0, sorted
    """
    node_stats = compute_policy_node_stats(graph)

    # Bidirectional pairs.
    bidirectional: List[Tuple[str, str]] = []
    seen_pairs: Set[Tuple[str, str]] = set()
    for (src, tgt) in sorted(graph.keys()):
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

    # Sources: out_degree > 0, in_degree == 0
    # (only count non-self-loop edges for source/sink detection).
    in_from_others: Dict[str, int] = {}
    out_to_others: Dict[str, int] = {}
    for (src, tgt), count in graph.items():
        if src != tgt:
            out_to_others[src] = out_to_others.get(src, 0) + count
            in_from_others[tgt] = in_from_others.get(tgt, 0) + count

    all_nodes = sorted(
        set(list(in_from_others.keys()) + list(out_to_others.keys()))
    )

    sources = sorted(
        node for node in all_nodes
        if out_to_others.get(node, 0) > 0
        and in_from_others.get(node, 0) == 0
    )

    sinks = sorted(
        node for node in all_nodes
        if in_from_others.get(node, 0) > 0
        and out_to_others.get(node, 0) == 0
    )

    return {
        "bidirectional": bidirectional,
        "self_loops": self_loops,
        "sources": sources,
        "sinks": sinks,
    }


# ---------------------------------------------------------------------------
# Public API — Topology Classification
# ---------------------------------------------------------------------------


def classify_policy_topology(
    graph: Dict[Tuple[str, str], int],
    stats: Dict[str, Dict[str, int]],
) -> str:
    """Classify the overall topology of the policy transition graph.

    Parameters
    ----------
    graph : dict
        Output of ``build_policy_graph``.
    stats : dict
        Output of ``compute_policy_node_stats``.

    Returns
    -------
    str
        One of:

        - ``"stable"`` — mostly self-loops (>= 70% of total flow)
        - ``"converging"`` — many policies flow into one dominant sink
        - ``"diverging"`` — one policy fans out to many others
        - ``"cyclic"`` — bidirectional transitions dominate
        - ``"mixed"`` — no single pattern dominates
    """
    if not graph:
        return "stable"

    # Compute total flow and self-loop flow.
    total_flow = sum(graph.values())
    self_loop_flow = sum(
        count for (src, tgt), count in graph.items() if src == tgt
    )

    if total_flow == 0:
        return "stable"

    # Stable: mostly self-loops.
    if self_loop_flow / total_flow >= 0.7:
        return "stable"

    # Count non-self-loop edges for structural analysis.
    non_self_edges: Dict[Tuple[str, str], int] = {
        (src, tgt): count
        for (src, tgt), count in graph.items()
        if src != tgt
    }
    non_self_flow = sum(non_self_edges.values())

    if non_self_flow == 0:
        return "stable"

    # Check for bidirectional dominance (cyclic).
    bidirectional_flow = 0
    seen: Set[Tuple[str, str]] = set()
    for (src, tgt), count in sorted(non_self_edges.items()):
        pair = tuple(sorted((src, tgt)))
        if pair in seen:
            continue
        seen.add(pair)
        if (src, tgt) in non_self_edges and (tgt, src) in non_self_edges:
            bidirectional_flow += (
                non_self_edges[(src, tgt)] + non_self_edges[(tgt, src)]
            )

    if non_self_flow > 0 and bidirectional_flow / non_self_flow >= 0.6:
        return "cyclic"

    # Converging: check if one node receives most incoming non-self flow.
    in_counts: Dict[str, int] = {}
    out_counts: Dict[str, int] = {}
    for (src, tgt), count in non_self_edges.items():
        in_counts[tgt] = in_counts.get(tgt, 0) + count
        out_counts[src] = out_counts.get(src, 0) + count

    if in_counts:
        max_in_node = max(sorted(in_counts.keys()), key=lambda n: in_counts[n])
        max_in = in_counts[max_in_node]
        if max_in / non_self_flow >= 0.5:
            return "converging"

    # Diverging: check if one node produces most outgoing non-self flow.
    if out_counts:
        max_out_node = max(
            sorted(out_counts.keys()), key=lambda n: out_counts[n]
        )
        max_out = out_counts[max_out_node]
        if max_out / non_self_flow >= 0.5:
            return "diverging"

    return "mixed"


# ---------------------------------------------------------------------------
# Public API — Formatting
# ---------------------------------------------------------------------------


def format_strategy_graph_summary(result: Dict[str, Any]) -> str:
    """Format strategy graph results as a human-readable summary.

    Parameters
    ----------
    result : dict
        Output of ``run_strategy_graph_analysis`` (from strategy_adapter).

    Returns
    -------
    str
        Multi-line summary string.
    """
    lines: List[str] = []
    lines.append("=== Strategy Graph ===")

    # Transitions.
    graph = result.get("graph", {})
    if graph:
        lines.append("")
        lines.append("Transitions:")
        ranked = sorted(
            [(src, tgt, count) for (src, tgt), count in graph.items()],
            key=lambda x: (-x[2], x[0], x[1]),
        )
        for src, tgt, count in ranked:
            lines.append(f"  {src} -> {tgt}: {count}")

    # Topology.
    topology = result.get("topology", "mixed")
    lines.append("")
    lines.append(f"Topology: {topology}")

    # Stability.
    stability = result.get("stability", {})
    if stability:
        dominant = stability.get("dominant_policy", "")
        switch_rate = stability.get("switch_rate", 0.0)
        self_loop_ratio = stability.get("self_loop_ratio", 0.0)
        longest_streak = stability.get("longest_streak", 0)

        lines.append("")
        if dominant:
            lines.append(f"Dominant Policy: {dominant}")
        lines.append(f"Switch Rate: {switch_rate:.4f}")
        lines.append(f"Self-Loop Ratio: {self_loop_ratio:.4f}")
        lines.append(f"Longest Streak: {longest_streak}")

    # Patterns.
    patterns = result.get("patterns", {})
    if patterns:
        bidirectional = patterns.get("bidirectional", [])
        self_loops = patterns.get("self_loops", [])
        sources = patterns.get("sources", [])
        sinks = patterns.get("sinks", [])

        if bidirectional or sources or sinks:
            lines.append("")
            lines.append("Patterns:")
            if bidirectional:
                for a, b in bidirectional:
                    lines.append(f"  Bidirectional: {a} <-> {b}")
            if self_loops:
                lines.append(f"  Self-loops: {', '.join(self_loops)}")
            if sources:
                lines.append(f"  Sources: {', '.join(sources)}")
            if sinks:
                lines.append(f"  Sinks: {', '.join(sinks)}")

    return "\n".join(lines)


__all__ = [
    "ROUND_PRECISION",
    "build_policy_graph",
    "classify_policy_topology",
    "compute_policy_node_stats",
    "compute_policy_stability",
    "detect_policy_patterns",
    "format_strategy_graph_summary",
]
