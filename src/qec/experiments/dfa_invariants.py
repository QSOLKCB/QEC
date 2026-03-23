"""Deterministic invariant discovery for DFA structures (v89.2.1).

Extracts provable, guaranteed-true structural invariants from a DFA
and its SCC hierarchy. No heuristics, no probabilistic reasoning —
only properties that are deterministically verifiable from the graph.

All functions are pure, deterministic, and non-mutating.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Dict, List, Optional, Set


# ---------------------------------------------------------------------------
# F1 — Dead State Invariants
# ---------------------------------------------------------------------------


def detect_dead_state_invariants(dfa: Dict[str, Any]) -> Dict[str, Any]:
    """Detect dead state invariants.

    A dead state is *absorbing* if all transitions from it loop back
    to itself.

    Returns {"has_dead_state": bool, "dead_state": int|None,
             "is_absorbing": bool, "all_self_loops": bool}.
    """
    dead_state = dfa.get("dead_state")
    if dead_state is None:
        return {
            "has_dead_state": False,
            "dead_state": None,
            "is_absorbing": False,
            "all_self_loops": False,
        }

    transitions = dfa.get("transitions", {})
    dead_trans = transitions.get(dead_state, {})

    # Check: all transitions from dead state loop to itself
    all_self = bool(dead_trans) and all(
        v == dead_state for v in dead_trans.values()
    )

    # Check: dead state has transitions for every symbol in the alphabet
    alphabet = dfa.get("alphabet", [])
    covers_all = set(dead_trans.keys()) >= set(alphabet)

    return {
        "has_dead_state": True,
        "dead_state": dead_state,
        "is_absorbing": all_self and covers_all,
        "all_self_loops": all_self,
    }


# ---------------------------------------------------------------------------
# F2 — Attractor Invariants
# ---------------------------------------------------------------------------


def detect_attractor_invariants(dfa: Dict[str, Any]) -> Dict[str, Any]:
    """Detect attractor (terminal SCC) invariants.

    For each terminal SCC, verifies:
    - no outgoing edges to other components
    - all internal transitions remain inside the SCC

    Returns {"attractors": list[dict]}.
    """
    from qec.experiments.dfa_analysis import compute_scc, classify_components
    from qec.experiments.dfa_engine import build_scc_graph, find_terminal_sccs

    sccs = compute_scc(dfa)
    classify_components(dfa, sccs)
    dag = build_scc_graph(sccs, dfa)
    terminal_ids = set(find_terminal_sccs(dag))

    transitions = dfa.get("transitions", {})

    attractors = []
    for scc in sorted(sccs, key=lambda s: s["id"]):
        if scc["id"] not in terminal_ids:
            continue

        states_set = set(scc["states"])

        # Check no outgoing edges in the condensation
        no_outgoing = len(dag.get(scc["id"], [])) == 0

        # Check all transitions from attractor states stay internal
        all_internal = True
        for s in sorted(states_set):
            for symbol in sorted(transitions.get(s, {})):
                ns = transitions[s][symbol]
                if ns not in states_set:
                    all_internal = False
                    break
            if not all_internal:
                break

        attractors.append({
            "component_id": scc["id"],
            "states": scc["states"],
            "no_outgoing_edges": no_outgoing,
            "all_transitions_internal": all_internal,
        })

    return {"attractors": attractors}


# ---------------------------------------------------------------------------
# F3 — Reachability Invariants
# ---------------------------------------------------------------------------


def detect_reachability_invariants(dfa: Dict[str, Any]) -> Dict[str, Any]:
    """Compute reachable and unreachable states from the initial state.

    Uses BFS from initial_state following all transitions.

    Returns {"reachable": sorted list, "unreachable": sorted list}.
    """
    initial = dfa.get("initial_state", 0)
    transitions = dfa.get("transitions", {})
    all_states = set(dfa.get("states", []))

    visited: Set[int] = set()
    queue: deque[int] = deque([initial])
    visited.add(initial)

    while queue:
        s = queue.popleft()
        for symbol in sorted(transitions.get(s, {})):
            ns = transitions[s][symbol]
            if ns not in visited:
                visited.add(ns)
                queue.append(ns)

    unreachable = all_states - visited
    return {
        "reachable": sorted(visited & all_states),
        "unreachable": sorted(unreachable),
    }


# ---------------------------------------------------------------------------
# F4 — Transition Invariants
# ---------------------------------------------------------------------------


def detect_transition_invariants(dfa: Dict[str, Any]) -> Dict[str, Any]:
    """Detect transition-level invariants.

    - forbidden_transitions: (state, symbol) pairs with no defined transition
    - deterministic_sinks: states where all symbols lead to the same next state

    Returns {"forbidden_transitions": list, "deterministic_sinks": list}.
    """
    transitions = dfa.get("transitions", {})
    states = sorted(dfa.get("states", []))
    alphabet = sorted(dfa.get("alphabet", []))

    forbidden: List[List[int]] = []
    sinks: List[int] = []

    for s in states:
        state_trans = transitions.get(s, {})

        # Forbidden: missing transitions
        for sym in alphabet:
            if sym not in state_trans:
                forbidden.append([s, sym])

        # Deterministic sink: all symbols → same next state
        if state_trans:
            targets = set(state_trans.values())
            if len(targets) == 1 and len(state_trans) == len(alphabet):
                sinks.append(s)

    return {
        "forbidden_transitions": forbidden,
        "deterministic_sinks": sinks,
    }


# ---------------------------------------------------------------------------
# F5 — Structural Invariants
# ---------------------------------------------------------------------------


def detect_structural_invariants(dfa: Dict[str, Any]) -> Dict[str, Any]:
    """Detect structural invariants using SCC hierarchy.

    - state_to_attractor: maps each state to the attractor it drains into
    - level_monotonic: whether all transitions go to same or lower level

    Returns {"state_to_attractor": dict, "level_monotonic": bool}.
    """
    from qec.experiments.dfa_analysis import compute_scc, classify_components
    from qec.experiments.dfa_engine import (
        build_scc_graph,
        compute_scc_levels,
        find_terminal_sccs,
        map_states_to_attractors,
    )

    sccs = compute_scc(dfa)
    classify_components(dfa, sccs)
    dag = build_scc_graph(sccs, dfa)
    terminal = find_terminal_sccs(dag)
    levels = compute_scc_levels(dag)
    attractor_info = map_states_to_attractors(dfa, sccs, dag, terminal)

    # Build state → level map
    state_to_comp: Dict[int, int] = {}
    for scc in sccs:
        for s in scc["states"]:
            state_to_comp[s] = scc["id"]

    state_to_level: Dict[int, int] = {}
    for s, cid in sorted(state_to_comp.items()):
        if cid in levels["levels"]:
            state_to_level[s] = levels["levels"][cid]

    # Check level monotonicity: transitions go to same or lower level
    transitions = dfa.get("transitions", {})
    monotonic = True
    for s in sorted(dfa.get("states", [])):
        s_level = state_to_level.get(s)
        if s_level is None:
            continue
        for symbol in sorted(transitions.get(s, {})):
            ns = transitions[s][symbol]
            ns_level = state_to_level.get(ns)
            if ns_level is None:
                continue
            if ns_level > s_level:
                monotonic = False
                break
        if not monotonic:
            break

    return {
        "state_to_attractor": {
            s: a for s, a in sorted(attractor_info["basins"].items())
        },
        "level_monotonic": monotonic,
    }


# ---------------------------------------------------------------------------
# F7 — Combined Output
# ---------------------------------------------------------------------------


def detect_invariants(
    dfa: Dict[str, Any],
    hierarchy: Dict[str, Any],
    analysis: Dict[str, Any],
) -> Dict[str, Any]:
    """Extract all deterministic invariants from DFA structure.

    Parameters:
        dfa: the DFA dict
        hierarchy: from compute_scc_levels
        analysis: dict with keys "terminal_sccs", "sccs", "scc_graph"

    Returns canonical invariant dict.
    """
    return {
        "invariants": {
            "dead_state": detect_dead_state_invariants(dfa),
            "attractors": detect_attractor_invariants(dfa),
            "reachability": detect_reachability_invariants(dfa),
            "transitions": detect_transition_invariants(dfa),
            "structure": detect_structural_invariants(dfa),
        }
    }
