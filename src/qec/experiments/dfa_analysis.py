"""DFA structural analysis: branching, cycles, and complexity metrics.

Computes deterministic structural properties of a minimized DFA
without modifying the DFA itself. All algorithms are pure, deterministic,
and use only stdlib + math.

v88.3.0
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# PART A — Graph Extraction
# ---------------------------------------------------------------------------

def build_adjacency(dfa: Dict[str, Any]) -> Dict[int, List[int]]:
    """Return adjacency map {state: sorted list of unique next states}."""
    adj: Dict[int, List[int]] = {}
    transitions = dfa.get("transitions", {})
    for state in sorted(dfa.get("states", [])):
        targets = set()
        state_trans = transitions.get(state, {})
        for symbol in sorted(state_trans):
            targets.add(state_trans[symbol])
        adj[state] = sorted(targets)
    return adj


# ---------------------------------------------------------------------------
# PART B — Branching Metrics
# ---------------------------------------------------------------------------

def compute_branching(dfa: Dict[str, Any]) -> Dict[str, Any]:
    """Compute per-state out-degree and global branching metrics."""
    transitions = dfa.get("transitions", {})
    states = dfa.get("states", [])
    alphabet = dfa.get("alphabet", [])
    alphabet_size = len(alphabet)

    out_degrees: List[int] = []
    for state in sorted(states):
        state_trans = transitions.get(state, {})
        unique_targets = len(set(state_trans.values()))
        out_degrees.append(unique_targets)

    n_states = len(states)
    if n_states == 0:
        return {
            "mean_branching": 0.0,
            "max_branching": 0,
            "min_branching": 0,
            "density": 0.0,
        }

    total_transitions = sum(len(transitions.get(s, {})) for s in states)
    density_denom = n_states * alphabet_size if alphabet_size > 0 else 1

    return {
        "mean_branching": sum(out_degrees) / n_states,
        "max_branching": max(out_degrees),
        "min_branching": min(out_degrees),
        "density": total_transitions / density_denom,
    }


# ---------------------------------------------------------------------------
# PART C — SCC / Cycle Analysis (Tarjan's Algorithm)
# ---------------------------------------------------------------------------

def compute_scc(dfa: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Compute strongly connected components using Tarjan's algorithm.

    Returns list of {\"id\": int, \"states\": sorted list} in deterministic order.
    """
    adj = build_adjacency(dfa)
    states = sorted(dfa.get("states", []))

    index_counter = [0]
    stack: List[int] = []
    on_stack = set()
    indices: Dict[int, int] = {}
    lowlinks: Dict[int, int] = {}
    result: List[List[int]] = []

    def strongconnect(v: int) -> None:
        indices[v] = index_counter[0]
        lowlinks[v] = index_counter[0]
        index_counter[0] += 1
        stack.append(v)
        on_stack.add(v)

        for w in adj.get(v, []):
            if w not in indices:
                strongconnect(w)
                lowlinks[v] = min(lowlinks[v], lowlinks[w])
            elif w in on_stack:
                lowlinks[v] = min(lowlinks[v], indices[w])

        if lowlinks[v] == indices[v]:
            component: List[int] = []
            while True:
                w = stack.pop()
                on_stack.discard(w)
                component.append(w)
                if w == v:
                    break
            result.append(sorted(component))

    for v in states:
        if v not in indices:
            strongconnect(v)

    # Deterministic ordering: sort by (size, first state)
    result.sort(key=lambda c: (len(c), c[0]))

    return [{"id": i, "states": comp} for i, comp in enumerate(result)]


def classify_components(
    dfa: Dict[str, Any],
    sccs: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Classify SCCs into fixed points, transients, and cycles."""
    adj = build_adjacency(dfa)
    n_cycles = 0
    largest_cycle = 0

    for scc in sccs:
        states = scc["states"]
        if len(states) == 1:
            s = states[0]
            # Self-loop check
            if s in adj.get(s, []):
                scc["type"] = "fixed_point"
            else:
                scc["type"] = "transient"
        else:
            scc["type"] = "cycle"
            n_cycles += 1
            largest_cycle = max(largest_cycle, len(states))

    states_in_cycles = sum(
        len(scc["states"]) for scc in sccs if scc["type"] == "cycle"
    )
    total_states = len(dfa.get("states", []))

    return {
        "n_scc": len(sccs),
        "n_cycles": n_cycles,
        "largest_cycle": largest_cycle,
        "cycle_fraction": states_in_cycles / max(1, total_states),
        "components": sccs,
    }


# ---------------------------------------------------------------------------
# PART D — Complexity Metrics
# ---------------------------------------------------------------------------

def _compute_branching_entropy(dfa: Dict[str, Any]) -> float:
    """Compute mean branching entropy (uniform assumption).

    For each state: entropy = log(out_degree) if out_degree > 0.
    Global: mean over all states.
    """
    transitions = dfa.get("transitions", {})
    states = dfa.get("states", [])

    if not states:
        return 0.0

    entropies: List[float] = []
    for state in sorted(states):
        state_trans = transitions.get(state, {})
        out_degree = len(set(state_trans.values()))
        if out_degree > 0:
            entropies.append(math.log(out_degree))
        else:
            entropies.append(0.0)

    return sum(entropies) / len(entropies)


def _compute_path_diversity(dfa: Dict[str, Any]) -> float:
    """Compute path diversity: unique transitions / n_states."""
    transitions = dfa.get("transitions", {})
    states = dfa.get("states", [])
    n_states = len(states)
    if n_states == 0:
        return 0.0

    unique_transitions = set()
    for state in sorted(states):
        state_trans = transitions.get(state, {})
        for symbol in sorted(state_trans):
            unique_transitions.add((state, symbol, state_trans[symbol]))

    return len(unique_transitions) / n_states


def _compute_dead_state_metrics(dfa: Dict[str, Any]) -> Dict[str, Any]:
    """Compute dead state role metrics.

    # NOTE: dead_state currently represents missing transition sink.
    # In v89 this may be treated as:
    # - rejection state
    # - terminal invalid region
    """
    dead_state = dfa.get("dead_state")
    transitions = dfa.get("transitions", {})
    states = dfa.get("states", [])

    if dead_state is None:
        return {
            "has_dead_state": False,
            "transitions_to_dead": 0,
            "dead_proportion": 0.0,
        }

    total_transitions = 0
    transitions_to_dead = 0
    for state in sorted(states):
        state_trans = transitions.get(state, {})
        for symbol in sorted(state_trans):
            total_transitions += 1
            if state_trans[symbol] == dead_state:
                transitions_to_dead += 1

    return {
        "has_dead_state": True,
        "transitions_to_dead": transitions_to_dead,
        "dead_proportion": (
            transitions_to_dead / max(1, total_transitions)
        ),
    }


def compute_complexity(dfa: Dict[str, Any]) -> Dict[str, Any]:
    """Compute lightweight complexity metrics."""
    return {
        "branching_entropy": _compute_branching_entropy(dfa),
        "path_diversity": _compute_path_diversity(dfa),
        "dead_state": _compute_dead_state_metrics(dfa),
    }


# ---------------------------------------------------------------------------
# PART E — Integration
# ---------------------------------------------------------------------------

# NOTE: inverse maps can be memory-heavy for large DFAs.
# Future: compress or build lazily if scaling becomes an issue.

def analyze_dfa(dfa: Dict[str, Any]) -> Dict[str, Any]:
    """Compute full structural analysis of a DFA.

    Returns branching, cycle, and complexity metrics without
    modifying the input DFA.
    """
    sccs = compute_scc(dfa)
    cycle_info = classify_components(dfa, sccs)

    return {
        "branching": compute_branching(dfa),
        "cycles": cycle_info,
        "complexity": compute_complexity(dfa),
    }
