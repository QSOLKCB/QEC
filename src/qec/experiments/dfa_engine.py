"""Deterministic DFA execution engine: prediction, validation, and control.

Transforms a minimized DFA from a passive validator into a deterministic
dynamical engine. All algorithms are pure, deterministic, and use only
stdlib + collections.

v89.0.0
"""

from __future__ import annotations

from collections import deque
from typing import Any, Dict, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# PART A — DFA Traversal Primitives
# ---------------------------------------------------------------------------


def get_next_states(dfa: Dict[str, Any], state: int) -> List[int]:
    """Return sorted list of states reachable from *state* via any symbol."""
    state_trans = dfa.get("transitions", {}).get(state, {})
    return sorted(set(state_trans.values()))


def step(dfa: Dict[str, Any], state: int, symbol: int) -> Optional[int]:
    """Return next state for (state, symbol), or None if undefined."""
    return dfa.get("transitions", {}).get(state, {}).get(symbol)


# ---------------------------------------------------------------------------
# PART B — Trajectory Validation
# ---------------------------------------------------------------------------


def validate_sequence(
    dfa: Dict[str, Any],
    sequence: List[int],
) -> Dict[str, Any]:
    """Validate a state sequence against DFA transitions.

    Checks that each consecutive pair (sequence[i], sequence[i+1]) is
    connected by some valid transition.  O(n) in sequence length.

    Returns {"valid": bool, "failure_index": int | None}.
    """
    transitions = dfa.get("transitions", {})

    for i in range(len(sequence) - 1):
        src = sequence[i]
        dst = sequence[i + 1]
        src_trans = transitions.get(src, {})
        # Check if dst is reachable from src via any symbol.
        if dst not in set(src_trans.values()):
            return {"valid": False, "failure_index": i}

    return {"valid": True, "failure_index": None}


# ---------------------------------------------------------------------------
# PART C — Deterministic Simulation
# ---------------------------------------------------------------------------


def simulate_from_state(
    dfa: Dict[str, Any],
    start_state: int,
    max_steps: int,
) -> Dict[str, Any]:
    """Generate all valid trajectories from *start_state* up to *max_steps*.

    Uses BFS with deterministic ordering.  Trajectories that revisit a
    state are marked cyclic and not expanded further to prevent infinite
    growth.

    Returns {"trajectories": list[list[int]], "n_trajectories": int}.
    """
    transitions = dfa.get("transitions", {})
    # Each entry: (path_so_far, visited_states_in_path)
    queue: deque[Tuple[List[int], Set[int]]] = deque()
    queue.append(([start_state], {start_state}))

    completed: List[List[int]] = []

    while queue:
        path, visited = queue.popleft()

        if len(path) - 1 >= max_steps:
            completed.append(path)
            continue

        current = path[-1]
        next_states = sorted(set(transitions.get(current, {}).values()))

        if not next_states:
            completed.append(path)
            continue

        expanded = False
        for ns in next_states:
            if ns in visited:
                # Cyclic — record path including the revisited state.
                completed.append(path + [ns])
            else:
                new_visited = visited | {ns}
                queue.append((path + [ns], new_visited))
                expanded = True

        if not expanded and not any(ns in visited for ns in next_states):
            completed.append(path)

    # Deterministic sort: lexicographic on path.
    completed.sort()

    return {
        "trajectories": completed,
        "n_trajectories": len(completed),
    }


# ---------------------------------------------------------------------------
# PART D — Goal-Oriented Control (BFS shortest path)
# ---------------------------------------------------------------------------


def find_control_path(
    dfa: Dict[str, Any],
    start_state: int,
    target_state: int,
    target_states: Optional[Set[int]] = None,
) -> Dict[str, Any]:
    """Find shortest path from *start_state* to a target using BFS.

    If *target_states* is provided, stop at the first one reached
    (multi-target mode).  Otherwise use *target_state*.

    Returns {"reachable": bool, "path": list, "symbols": list, "length": int}.
    """
    targets: Set[int] = target_states if target_states is not None else {target_state}
    transitions = dfa.get("transitions", {})

    if start_state in targets:
        return {
            "reachable": True,
            "path": [start_state],
            "symbols": [],
            "length": 0,
        }

    visited: Set[int] = {start_state}
    # (current_state, path_states, path_symbols)
    queue: deque[Tuple[int, List[int], List[int]]] = deque()
    queue.append((start_state, [start_state], []))

    while queue:
        current, path, symbols = queue.popleft()
        state_trans = transitions.get(current, {})

        for symbol in sorted(state_trans):
            ns = state_trans[symbol]
            if ns in visited:
                continue
            new_path = path + [ns]
            new_symbols = symbols + [symbol]

            if ns in targets:
                return {
                    "reachable": True,
                    "path": new_path,
                    "symbols": new_symbols,
                    "length": len(new_path) - 1,
                }

            visited.add(ns)
            queue.append((ns, new_path, new_symbols))

    return {
        "reachable": False,
        "path": [],
        "symbols": [],
        "length": 0,
    }


# ---------------------------------------------------------------------------
# PART E — Attractor Mapping (SCC Meta-Graph)
# ---------------------------------------------------------------------------


def build_scc_graph(
    sccs: List[Dict[str, Any]],
    dfa: Dict[str, Any],
) -> Dict[int, List[int]]:
    """Build the condensed DAG of SCC components.

    Returns {component_id: sorted list of next component_ids}.
    """
    transitions = dfa.get("transitions", {})

    # Map state → component id.
    state_to_comp: Dict[int, int] = {}
    for scc in sccs:
        cid = scc["id"]
        for s in scc["states"]:
            state_to_comp[s] = cid

    dag: Dict[int, Set[int]] = {scc["id"]: set() for scc in sccs}

    for scc in sccs:
        cid = scc["id"]
        for s in scc["states"]:
            for symbol in sorted(transitions.get(s, {})):
                ns = transitions[s][symbol]
                target_cid = state_to_comp.get(ns)
                if target_cid is not None and target_cid != cid:
                    dag[cid].add(target_cid)

    return {cid: sorted(targets) for cid, targets in sorted(dag.items())}


def find_terminal_sccs(scc_graph: Dict[int, List[int]]) -> List[int]:
    """Return sorted list of terminal (no outgoing edges) component IDs.

    Terminal SCCs are true dynamical attractors.
    """
    return sorted(cid for cid, targets in scc_graph.items() if not targets)


def map_states_to_attractors(
    dfa: Dict[str, Any],
    sccs: List[Dict[str, Any]],
    scc_graph: Dict[int, List[int]],
    terminal_sccs: List[int],
) -> Dict[str, Any]:
    """Map each state to its downstream attractor via forward reachability.

    Returns {"attractors": list[dict], "basins": {state: attractor_id}}.
    """
    terminal_set = set(terminal_sccs)

    # Map state → component id.
    state_to_comp: Dict[int, int] = {}
    for scc in sccs:
        for s in scc["states"]:
            state_to_comp[s] = scc["id"]

    # BFS on the DAG to find which terminal SCC each component drains into.
    # Build reverse DAG for backward reachability from terminals.
    reverse_dag: Dict[int, List[int]] = {cid: [] for cid in scc_graph}
    for cid, targets in scc_graph.items():
        for t in targets:
            reverse_dag[t].append(cid)

    comp_to_attractor: Dict[int, int] = {}
    # Process each terminal SCC.
    for tcid in sorted(terminal_set):
        queue: deque[int] = deque([tcid])
        while queue:
            cid = queue.popleft()
            if cid in comp_to_attractor:
                continue
            comp_to_attractor[cid] = tcid
            for pred in sorted(reverse_dag.get(cid, [])):
                if pred not in comp_to_attractor:
                    queue.append(pred)

    # Map states.
    basins: Dict[int, int] = {}
    for s in sorted(dfa.get("states", [])):
        cid = state_to_comp.get(s)
        if cid is not None:
            basins[s] = comp_to_attractor.get(cid, -1)

    # Build attractor info.
    scc_lookup = {scc["id"]: scc for scc in sccs}
    attractors = []
    for tcid in sorted(terminal_set):
        scc = scc_lookup[tcid]
        attractors.append({
            "component_id": tcid,
            "states": scc["states"],
            "basin_size": sum(1 for v in basins.values() if v == tcid),
        })

    return {
        "attractors": attractors,
        "basins": basins,
    }


# TODO (v89+):
# refine cycle classification:
# - simple loops
# - nested cycles
# - complex regions

# TODO (v89+):
# entropy weighting using visitation or sequence frequency


# ---------------------------------------------------------------------------
# PART F — Engine Wrapper
# ---------------------------------------------------------------------------


def run_dfa_engine(dfa: Dict[str, Any]) -> Dict[str, Any]:
    """Run the full DFA engine on a minimized DFA.

    Returns validation, simulation, control, and attractor analysis
    without modifying the input DFA.
    """
    from qec.experiments.dfa_analysis import compute_scc, classify_components

    states = sorted(dfa.get("states", []))
    initial = dfa.get("initial_state", 0)

    # --- Validation: check that every adjacent pair in the state list
    #     forms a valid transition (self-check). ---
    validation = validate_sequence(dfa, states)

    # --- Simulation from initial state. ---
    simulation = simulate_from_state(dfa, initial, max_steps=min(len(states), 20))

    # --- Control: paths from initial state to every other state. ---
    control_paths: Dict[str, Any] = {}
    for s in states:
        if s == initial:
            continue
        result = find_control_path(dfa, initial, s)
        control_paths[s] = result

    # --- SCC meta-graph / attractors. ---
    sccs = compute_scc(dfa)
    classify_components(dfa, sccs)
    scc_graph = build_scc_graph(sccs, dfa)
    terminal = find_terminal_sccs(scc_graph)
    attractor_info = map_states_to_attractors(dfa, sccs, scc_graph, terminal)

    return {
        "validation": validation,
        "simulation": simulation,
        "control": {
            "paths": control_paths,
            "reachable_count": sum(
                1 for v in control_paths.values() if v["reachable"]
            ),
            "unreachable_count": sum(
                1 for v in control_paths.values() if not v["reachable"]
            ),
        },
        "attractors": {
            "scc_graph": scc_graph,
            "terminal_sccs": terminal,
            **attractor_info,
        },
    }
