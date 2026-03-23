"""Deterministic invariant discovery for DFA structures (v89.3.0).

Extracts provable, guaranteed-true structural invariants from a DFA
and its SCC hierarchy. No heuristics, no probabilistic reasoning —
only properties that are deterministically verifiable from the graph.

Includes invariant normalization and constraint derivation for the
invariant integration layer (observation → control bridge).

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


# ---------------------------------------------------------------------------
# G1 — Invariant Normalization
# ---------------------------------------------------------------------------


def normalize_invariants(inv: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize an invariant dict into canonical form.

    Guarantees:
    - All top-level sections exist (dead_state, attractors, reachability,
      transitions, structure).
    - Lists are sorted.
    - Sets are converted to sorted lists.
    - Missing sections become empty dicts.
    - Output is deterministic: normalize(inv) == normalize(inv).
    """
    raw = inv.get("invariants", inv) if isinstance(inv, dict) else {}

    sections = ("dead_state", "attractors", "reachability",
                "transitions", "structure")

    normalized: Dict[str, Any] = {}
    for section in sections:
        normalized[section] = _normalize_section(raw.get(section, {}))

    return {"invariants": normalized}


def _normalize_section(section: Any) -> Any:
    """Recursively normalize a section value into canonical form."""
    if isinstance(section, set):
        return sorted(section)
    if isinstance(section, list):
        # Sort lists of comparable items; leave lists of dicts as-is
        # but normalize each element.
        result = [_normalize_section(item) for item in section]
        # Sort if items are comparable (not dicts).
        if result and not isinstance(result[0], dict):
            try:
                result = sorted(result)
            except TypeError:
                pass
        return result
    if isinstance(section, dict):
        return {k: _normalize_section(v) for k, v in sorted(section.items())}
    return section


# ---------------------------------------------------------------------------
# G2 — Invariants → Constraints Derivation
# ---------------------------------------------------------------------------


def derive_constraints_from_invariants(
    inv: Dict[str, Any],
) -> Dict[str, Any]:
    """Derive deterministic constraints from proven invariants.

    Mapping rules (strict — no inference beyond provable facts):
    1. Dead state (if absorbing) → avoid_states.
    2. States that provably lead only to dead state → avoid_states.
    3. If state_to_attractor mapping is total and deterministic,
       optionally derive allow_only_states from reachable states.

    Returns {"avoid_states": set, "allow_only_states": set | None}.
    """
    normed = normalize_invariants(inv)
    sections = normed["invariants"]

    avoid: Set[int] = set()
    allow_only: Optional[Set[int]] = None

    # --- Rule 1: Dead state → avoid ---
    dead_info = sections.get("dead_state", {})
    if dead_info.get("has_dead_state") and dead_info.get("is_absorbing"):
        dead_state = dead_info.get("dead_state")
        if dead_state is not None:
            avoid.add(dead_state)

    # --- Rule 2: Forbidden regions (states leading only to dead state) ---
    structure = sections.get("structure", {})
    state_to_attractor = structure.get("state_to_attractor", {})
    if dead_info.get("has_dead_state") and dead_info.get("dead_state") is not None:
        dead_state = dead_info["dead_state"]
        # Find the attractor component containing the dead state.
        # States whose only attractor is the dead state's component
        # are forbidden regions.
        dead_attractor = state_to_attractor.get(
            dead_state if isinstance(dead_state, str) else dead_state
        )
        if dead_attractor is not None:
            for state_key, att_id in sorted(state_to_attractor.items()):
                state = int(state_key) if isinstance(state_key, str) else state_key
                if att_id == dead_attractor and state != dead_state:
                    avoid.add(state)

    # --- Rule 3: Allow-only from reachable states ---
    reachability = sections.get("reachability", {})
    reachable = reachability.get("reachable", [])
    if reachable and state_to_attractor:
        # Only apply if mapping is total over reachable states
        reachable_set = set(reachable)
        mapped = {
            int(k) if isinstance(k, str) else k
            for k in state_to_attractor
        }
        if reachable_set <= mapped:
            allow_only = reachable_set - avoid

    return {
        "avoid_states": avoid,
        "allow_only_states": allow_only,
    }
