"""Deterministic DFA execution engine: prediction, validation, and control.

Transforms a minimized DFA from a passive validator into a deterministic
dynamical engine. All algorithms are pure, deterministic, and use only
stdlib + collections.

v91.1.0 — observability metrics and geometric correction integration.
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
# PART A2 — Constraint Composition
# ---------------------------------------------------------------------------


def normalize_constraints(
    constraints: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Normalize a constraints dict into canonical form.

    Returns {"avoid_states": set, "avoid_symbols": set,
             "max_depth": int|None, "allow_only_states": set|None}.
    """
    if not constraints:
        return {
            "avoid_states": set(),
            "avoid_symbols": set(),
            "max_depth": None,
            "allow_only_states": None,
        }
    return {
        "avoid_states": set(constraints.get("avoid_states", ())),
        "avoid_symbols": set(constraints.get("avoid_symbols", ())),
        "max_depth": constraints.get("max_depth"),
        "allow_only_states": (
            set(constraints["allow_only_states"])
            if constraints.get("allow_only_states") is not None
            else None
        ),
    }


def compose_constraints(
    base: Optional[Dict[str, Any]],
    extra: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Merge two constraint dicts deterministically.

    Rules:
    * avoid_states → union
    * avoid_symbols → union
    * allow_only_states → intersection (if both exist)
    * max_depth → min (if both exist)
    """
    b = normalize_constraints(base)
    e = normalize_constraints(extra)

    # allow_only_states: intersection when both present.
    b_allow = b["allow_only_states"]
    e_allow = e["allow_only_states"]
    if b_allow is not None and e_allow is not None:
        merged_allow: Optional[Set[int]] = b_allow & e_allow
    elif b_allow is not None:
        merged_allow = set(b_allow)
    elif e_allow is not None:
        merged_allow = set(e_allow)
    else:
        merged_allow = None

    # max_depth: min when both present.
    b_depth = b["max_depth"]
    e_depth = e["max_depth"]
    if b_depth is not None and e_depth is not None:
        merged_depth: Optional[int] = min(b_depth, e_depth)
    elif b_depth is not None:
        merged_depth = b_depth
    else:
        merged_depth = e_depth

    return {
        "avoid_states": b["avoid_states"] | e["avoid_states"],
        "avoid_symbols": b["avoid_symbols"] | e["avoid_symbols"],
        "max_depth": merged_depth,
        "allow_only_states": merged_allow,
    }


# ---------------------------------------------------------------------------
# PART A3 — Full Control Surface Composition
# ---------------------------------------------------------------------------


def compose_control_constraints(
    user_constraints: Optional[Dict[str, Any]] = None,
    invariant_constraints: Optional[Dict[str, Any]] = None,
    supervisor_result: Optional[Dict[str, Any]] = None,
    policy_constraints: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Compose user ⊕ invariant ⊕ supervisor ⊕ policy into full surface.

    Rules:
    - user + invariant compose via existing algebra (union avoid, intersect allow)
    - supervisor contributes hard transition disabling (forbidden states)
    - policy contributes allowed_symbols_by_state
    - all provenance retained

    Returns canonical composed constraint dict.
    """
    # Start with user ⊕ invariant.
    base = compose_constraints(user_constraints, invariant_constraints)

    # Supervisor contributes forbidden states as additional avoid_states.
    if supervisor_result:
        sup_avoid = set(supervisor_result.get("forbidden_states", []))
        sup_constraints = {"avoid_states": sup_avoid}
        base = compose_constraints(base, sup_constraints)

    # Extract final values.
    nc = normalize_constraints(base)
    avoid = sorted(nc["avoid_states"])
    allow_only = sorted(nc["allow_only_states"]) if nc["allow_only_states"] is not None else None
    max_depth = nc["max_depth"]

    # Policy contributes allowed_symbols_by_state.
    allowed_symbols: Optional[Dict[int, List[int]]] = None
    if policy_constraints and "allowed_symbols_by_state" in policy_constraints:
        allowed_symbols = {
            k: list(v)
            for k, v in sorted(policy_constraints["allowed_symbols_by_state"].items())
        }

    # Build provenance.
    sources: Dict[str, Any] = {}
    if user_constraints:
        uc = normalize_constraints(user_constraints)
        if uc["avoid_states"]:
            sources["user_avoid"] = sorted(uc["avoid_states"])
        if uc["allow_only_states"] is not None:
            sources["user_allow_only"] = sorted(uc["allow_only_states"])
    if invariant_constraints:
        ic = normalize_constraints(invariant_constraints)
        if ic["avoid_states"]:
            sources["invariant_avoid"] = sorted(ic["avoid_states"])
        if ic["allow_only_states"] is not None:
            sources["invariant_allow_only"] = sorted(ic["allow_only_states"])
    if supervisor_result and supervisor_result.get("forbidden_states"):
        sources["supervisor_forbidden"] = sorted(supervisor_result["forbidden_states"])
    if allowed_symbols is not None:
        sources["policy_symbols"] = True

    result: Dict[str, Any] = {
        "avoid_states": avoid,
        "allow_only_states": allow_only,
        "max_depth": max_depth,
        "sources": sources,
    }
    if allowed_symbols is not None:
        result["allowed_symbols_by_state"] = allowed_symbols
    return result


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
    pruning: Optional[Dict[str, Any]] = None,
    hierarchy: Optional[Dict[str, Any]] = None,
    use_invariants: bool = False,
    supervised_dfa: Optional[Dict[str, Any]] = None,
    constraint_bundle: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Generate all valid trajectories from *start_state* up to *max_steps*.

    Uses BFS with deterministic ordering.  Trajectories that revisit a
    state are marked cyclic and not expanded further to prevent infinite
    growth.

    Optional *pruning* dict supports:
        stop_at_attractors: stop expansion when entering a terminal SCC
        avoid_dead_state:   do not expand the dead state
        max_level:          skip states with hierarchy level > this value
        target_levels:      only expand states within these levels
        stop_at_levels:     stop expansion when entering these levels

    If *use_invariants* is True, derives constraints from DFA invariants
    and applies them as additional pruning (avoid provably dead regions).

    Optional *hierarchy* dict (from compute_scc_levels) provides level info.

    If *supervised_dfa* is provided, simulate on the supervised DFA.
    If *constraint_bundle* has allowed_symbols_by_state, enforce it.

    Returns {"trajectories": list[list[int]], "n_trajectories": int,
             "pruned": bool, "pruning_used": dict,
             "invariant_constraints_applied": bool}.
    """
    # Use supervised DFA if provided.
    active_dfa = supervised_dfa if supervised_dfa is not None else dfa
    sup_allowed_symbols: Optional[Dict[int, List[int]]] = None
    if constraint_bundle is not None:
        composed = constraint_bundle.get("composed", {})
        if "allowed_symbols_by_state" in composed:
            sup_allowed_symbols = composed["allowed_symbols_by_state"]

    transitions = active_dfa.get("transitions", {})
    pruned = pruning is not None and bool(pruning)
    stop_at_attractors = bool(pruning.get("stop_at_attractors")) if pruning else False
    avoid_dead = bool(pruning.get("avoid_dead_state")) if pruning else False
    max_level: Optional[int] = pruning.get("max_level") if pruning else None
    target_levels: Optional[Set[int]] = (
        set(pruning["target_levels"]) if pruning and pruning.get("target_levels") is not None else None
    )
    stop_at_levels: Optional[Set[int]] = (
        set(pruning["stop_at_levels"]) if pruning and pruning.get("stop_at_levels") is not None else None
    )

    # Invariant-derived pruning constraints.
    inv_avoid: Set[int] = set()
    inv_applied = False
    if use_invariants:
        from qec.experiments.dfa_invariants import (
            detect_invariants,
            derive_constraints_from_invariants,
        )
        inv_data = detect_invariants(dfa, {}, {})
        inv_constraints = derive_constraints_from_invariants(inv_data)
        inv_avoid = set(inv_constraints.get("avoid_states", set()))
        inv_applied = bool(inv_avoid)
        pruned = pruned or inv_applied

    dead_state = dfa.get("dead_state")

    # Build state → level map from hierarchy if provided.
    state_to_level: Dict[int, int] = {}
    if hierarchy is not None:
        levels_map = hierarchy.get("levels", {})
        # Map state → component id, then component level.
        state_to_comp: Dict[int, int] = {}
        if hierarchy.get("_sccs"):
            for scc in hierarchy["_sccs"]:
                for s in scc["states"]:
                    state_to_comp[s] = scc["id"]
            for s, cid in state_to_comp.items():
                if cid in levels_map:
                    state_to_level[s] = levels_map[cid]

    # Pre-compute terminal SCC states if needed.
    terminal_states: Set[int] = set()
    if stop_at_attractors:
        from qec.experiments.dfa_analysis import compute_scc, classify_components
        sccs = compute_scc(dfa)
        classify_components(dfa, sccs)
        scc_graph = build_scc_graph(sccs, dfa)
        terminal_ids = set(find_terminal_sccs(scc_graph))
        for scc in sccs:
            if scc["id"] in terminal_ids:
                for s in scc["states"]:
                    terminal_states.add(s)

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

        # Pruning: stop at attractor states (don't expand further).
        if stop_at_attractors and len(path) > 1 and current in terminal_states:
            completed.append(path)
            continue

        # Pruning: stop at specific hierarchy levels.
        if stop_at_levels and len(path) > 1 and state_to_level.get(current) in stop_at_levels:
            completed.append(path)
            continue

        # Apply allowed symbols filtering from supervisor policy.
        current_trans = transitions.get(current, {})
        if sup_allowed_symbols is not None and current in sup_allowed_symbols:
            allowed_set = set(sup_allowed_symbols[current])
            filtered_trans = {sym: ns for sym, ns in current_trans.items() if sym in allowed_set}
        else:
            filtered_trans = current_trans
        next_states = sorted(set(filtered_trans.values()))

        if not next_states:
            completed.append(path)
            continue

        expanded = False
        for ns in next_states:
            if avoid_dead and ns == dead_state:
                continue
            # Invariant-derived pruning: skip provably dead regions.
            if inv_avoid and ns in inv_avoid:
                continue
            # Hierarchy-based pruning: skip states exceeding max_level.
            if max_level is not None and state_to_level.get(ns, 0) > max_level:
                continue
            # Hierarchy-based pruning: skip states outside target_levels.
            if target_levels is not None and state_to_level.get(ns, 0) not in target_levels:
                continue
            if ns in visited:
                # Cyclic — record path including the revisited state.
                completed.append(path + [ns])
            else:
                new_visited = visited | {ns}
                queue.append((path + [ns], new_visited))
                expanded = True

        if not expanded and not any(
            ns in visited for ns in next_states
            if not (avoid_dead and ns == dead_state)
            and not (inv_avoid and ns in inv_avoid)
            and not (max_level is not None and state_to_level.get(ns, 0) > max_level)
            and not (target_levels is not None and state_to_level.get(ns, 0) not in target_levels)
        ):
            completed.append(path)

    # Deterministic sort: lexicographic on path.
    completed.sort()

    pruning_used: Dict[str, Any] = {}
    if pruning:
        if stop_at_attractors:
            pruning_used["stop_at_attractors"] = True
        if avoid_dead:
            pruning_used["avoid_dead_state"] = True
        if max_level is not None:
            pruning_used["max_level"] = max_level
        if target_levels is not None:
            pruning_used["target_levels"] = sorted(target_levels)
        if stop_at_levels is not None:
            pruning_used["stop_at_levels"] = sorted(stop_at_levels)

    result = {
        "trajectories": completed,
        "n_trajectories": len(completed),
        "pruned": pruned,
        "pruning_used": pruning_used,
        "invariant_constraints_applied": inv_applied,
    }
    return result


# ---------------------------------------------------------------------------
# PART D — Goal-Oriented Control (BFS shortest path)
# ---------------------------------------------------------------------------


def find_control_path(
    dfa: Dict[str, Any],
    start_state: int,
    target_state: int,
    target_states: Optional[Set[int]] = None,
    constraints: Optional[Dict[str, Any]] = None,
    hierarchy: Optional[Dict[str, Any]] = None,
    prefer_lower_levels: bool = False,
    use_invariants: bool = False,
    supervised_dfa: Optional[Dict[str, Any]] = None,
    constraint_bundle: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Find shortest path from *start_state* to a target using BFS.

    If *target_states* is provided, stop at the first one reached
    (multi-target mode).  Otherwise use *target_state*.

    Optional *constraints* dict supports:
        avoid_states:      set of states to skip during traversal
        avoid_symbols:     set of symbols to skip during traversal
        max_depth:         int upper bound on path length
        allow_only_states: set of states allowed (None = no restriction)

    If *use_invariants* is True, derives constraints from DFA invariants
    and composes them with user-provided constraints.

    Optional *hierarchy* dict (from compute_scc_levels) enables level-aware
    tie-breaking when *prefer_lower_levels* is True.

    If *supervised_dfa* is provided, BFS runs on the supervised DFA instead.
    If *constraint_bundle* has allowed_symbols_by_state, enforce it exactly.

    Returns {"reachable": bool, "path": list, "symbols": list,
             "length": int, "constrained": bool, "constraints_used": dict,
             "invariant_constraints_applied": bool}.
    """
    # Use supervised DFA if provided.
    active_dfa = supervised_dfa if supervised_dfa is not None else dfa
    allowed_symbols_by_state: Optional[Dict[int, List[int]]] = None
    if constraint_bundle is not None:
        composed = constraint_bundle.get("composed", {})
        if "allowed_symbols_by_state" in composed:
            allowed_symbols_by_state = composed["allowed_symbols_by_state"]

    targets: Set[int] = target_states if target_states is not None else {target_state}
    transitions = active_dfa.get("transitions", {})

    # Derive and compose invariant constraints if requested.
    inv_applied = False
    if use_invariants:
        from qec.experiments.dfa_invariants import (
            detect_invariants,
            derive_constraints_from_invariants,
        )
        inv_data = detect_invariants(dfa, {}, {})
        inv_constraints = derive_constraints_from_invariants(inv_data)
        # Only apply if constraints actually restrict something.
        has_avoid = bool(inv_constraints.get("avoid_states"))
        inv_allow = inv_constraints.get("allow_only_states")
        all_states = set(dfa.get("states", []))
        has_restrictive_allow = (
            inv_allow is not None and inv_allow != all_states
        )
        if has_avoid or has_restrictive_allow:
            constraints = compose_constraints(constraints, inv_constraints)
            inv_applied = True

    nc = normalize_constraints(constraints)
    avoid_states = nc["avoid_states"]
    avoid_symbols = nc["avoid_symbols"]
    max_depth = nc["max_depth"]
    allow_only = nc["allow_only_states"]

    constrained = constraints is not None and bool(constraints)

    # Build state → level map for tie-breaking.
    state_to_level: Dict[int, int] = {}
    if hierarchy is not None and prefer_lower_levels:
        levels_map = hierarchy.get("levels", {})
        if hierarchy.get("_sccs"):
            for scc in hierarchy["_sccs"]:
                for s in scc["states"]:
                    cid = scc["id"]
                    if cid in levels_map:
                        state_to_level[s] = levels_map[cid]

    # Check allow_only for start_state.
    if allow_only is not None and start_state not in allow_only:
        return {
            "reachable": False,
            "path": [],
            "symbols": [],
            "length": 0,
            "constrained": constrained,
            "constraints_used": _constraints_used_summary(nc),
            "invariant_constraints_applied": inv_applied,
        }

    if start_state in targets and start_state not in avoid_states:
        return {
            "reachable": True,
            "path": [start_state],
            "symbols": [],
            "length": 0,
            "constrained": constrained,
            "constraints_used": _constraints_used_summary(nc),
            "invariant_constraints_applied": inv_applied,
        }

    visited: Set[int] = {start_state}
    # (current_state, path_states, path_symbols)
    queue: deque[Tuple[int, List[int], List[int]]] = deque()
    queue.append((start_state, [start_state], []))

    while queue:
        current, path, symbols = queue.popleft()
        depth = len(path) - 1

        if max_depth is not None and depth >= max_depth:
            continue

        state_trans = transitions.get(current, {})

        # Collect candidate (symbol, next_state) pairs.
        candidates: List[Tuple[int, int]] = []
        # Determine valid symbols for this state.
        valid_symbols = sorted(state_trans)
        if allowed_symbols_by_state is not None and current in allowed_symbols_by_state:
            allowed_set = set(allowed_symbols_by_state[current])
            valid_symbols = [s for s in valid_symbols if s in allowed_set]
        for symbol in valid_symbols:
            if symbol in avoid_symbols:
                continue
            ns = state_trans[symbol]
            if ns in avoid_states:
                continue
            if allow_only is not None and ns not in allow_only:
                continue
            if ns in visited:
                continue
            candidates.append((symbol, ns))

        # Hierarchy tie-breaking: sort by (level, state_id) — stable.
        if prefer_lower_levels and state_to_level:
            candidates.sort(key=lambda pair: (state_to_level.get(pair[1], 0), pair[1]))

        for symbol, ns in candidates:
            new_path = path + [ns]
            new_symbols = symbols + [symbol]

            if ns in targets:
                return {
                    "reachable": True,
                    "path": new_path,
                    "symbols": new_symbols,
                    "length": len(new_path) - 1,
                    "constrained": constrained,
                    "constraints_used": _constraints_used_summary(nc),
                    "invariant_constraints_applied": inv_applied,
                }

            visited.add(ns)
            queue.append((ns, new_path, new_symbols))

    return {
        "reachable": False,
        "path": [],
        "symbols": [],
        "length": 0,
        "constrained": constrained,
        "constraints_used": _constraints_used_summary(nc),
        "invariant_constraints_applied": inv_applied,
    }


def _constraints_used_summary(nc: Dict[str, Any]) -> Dict[str, Any]:
    """Build a serializable summary of active constraints."""
    result: Dict[str, Any] = {}
    if nc["avoid_states"]:
        result["avoid_states"] = sorted(nc["avoid_states"])
    if nc["avoid_symbols"]:
        result["avoid_symbols"] = sorted(nc["avoid_symbols"])
    if nc["max_depth"] is not None:
        result["max_depth"] = nc["max_depth"]
    if nc["allow_only_states"] is not None:
        result["allow_only_states"] = sorted(nc["allow_only_states"])
    return result


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


def compute_scc_levels(
    scc_graph: Dict[int, List[int]],
) -> Dict[str, Any]:
    """Compute hierarchy levels for SCC components.

    Terminal SCCs get level 0, their parents level 1, etc.
    (reverse topological depth).

    Returns {"levels": {component_id: level}, "max_level": int}.
    """
    terminal = set(find_terminal_sccs(scc_graph))
    levels: Dict[int, int] = {}

    # Build reverse graph.
    reverse: Dict[int, List[int]] = {cid: [] for cid in scc_graph}
    for cid, targets in scc_graph.items():
        for t in targets:
            reverse[t].append(cid)

    # BFS from terminals upward.
    queue: deque[int] = deque()
    for t in sorted(terminal):
        levels[t] = 0
        queue.append(t)

    while queue:
        cid = queue.popleft()
        for pred in sorted(reverse.get(cid, [])):
            if pred not in levels:
                levels[pred] = levels[cid] + 1
                queue.append(pred)

    max_level = max(levels.values()) if levels else 0
    return {"levels": dict(sorted(levels.items())), "max_level": max_level}


def compute_scc_transition_summary(
    scc_graph: Dict[int, List[int]],
    terminal_sccs: List[int],
) -> Dict[int, Dict[str, Any]]:
    """Summarize each SCC's outgoing edges and attractor reachability.

    Returns {component_id: {"outgoing_count": int, "leads_to_attractor": bool}}.
    """
    terminal_set = set(terminal_sccs)

    # Forward reachability via BFS on DAG to check attractor access.
    reachable_to_terminal: Set[int] = set()
    reverse: Dict[int, List[int]] = {cid: [] for cid in scc_graph}
    for cid, targets in scc_graph.items():
        for t in targets:
            reverse[t].append(cid)

    queue: deque[int] = deque()
    for t in sorted(terminal_set):
        reachable_to_terminal.add(t)
        queue.append(t)
    while queue:
        cid = queue.popleft()
        for pred in sorted(reverse.get(cid, [])):
            if pred not in reachable_to_terminal:
                reachable_to_terminal.add(pred)
                queue.append(pred)

    summary: Dict[int, Dict[str, Any]] = {}
    for cid in sorted(scc_graph):
        summary[cid] = {
            "outgoing_count": len(scc_graph[cid]),
            "leads_to_attractor": cid in reachable_to_terminal,
        }
    return summary


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

    # --- SCC meta-graph / attractors (computed early for hierarchy). ---
    sccs = compute_scc(dfa)
    classify_components(dfa, sccs)
    scc_graph = build_scc_graph(sccs, dfa)
    terminal = find_terminal_sccs(scc_graph)
    attractor_info = map_states_to_attractors(dfa, sccs, scc_graph, terminal)

    # --- Attractor hierarchy. ---
    hierarchy = compute_scc_levels(scc_graph)
    transition_summary = compute_scc_transition_summary(scc_graph, terminal)

    # Enrich hierarchy with SCC info for state-level lookups.
    hierarchy_with_sccs = {**hierarchy, "_sccs": sccs}

    # --- Invariant constraints derivation. ---
    from qec.experiments.dfa_invariants import (
        detect_invariants,
        derive_constraints_from_invariants,
    )
    inv_data = detect_invariants(
        dfa, hierarchy, {"terminal_sccs": terminal, "sccs": sccs, "scc_graph": scc_graph}
    )
    inv_constraints = derive_constraints_from_invariants(inv_data)
    inv_has_constraints = bool(
        inv_constraints.get("avoid_states")
        or inv_constraints.get("allow_only_states") is not None
    )

    # --- Supervisor synthesis (v90.0.0). ---
    from qec.experiments.dfa_supervisor import (
        compute_supervisor_metrics,
        normalize_constraint_bundle,
        run_supervisor,
        stratify_forbidden_states,
        structure_invariant_constraints,
    )
    structured_inv = structure_invariant_constraints(inv_constraints, inv_data)
    supervisor = run_supervisor(dfa, inv_data, inv_constraints)

    # Build constraint bundle.
    constraint_bundle = normalize_constraint_bundle({
        "user": {"avoid_states": [], "allow_only_states": None},
        "invariant": structured_inv,
        "supervisor": {
            "avoid_states": supervisor["forbidden_states"],
            "allow_only_states": None,
        },
        "policy": supervisor.get("policy_constraints", {}),
        "composed": compose_control_constraints(
            user_constraints=None,
            invariant_constraints=inv_constraints,
            supervisor_result=supervisor,
            policy_constraints=supervisor.get("policy_constraints"),
        ),
    })

    # --- Simulation from initial state. ---
    simulation = simulate_from_state(
        dfa, initial, max_steps=min(len(states), 20),
        hierarchy=hierarchy_with_sccs,
    )

    # --- Control: paths from initial state to every other state. ---
    control_paths: Dict[str, Any] = {}
    for s in states:
        if s == initial:
            continue
        result = find_control_path(dfa, initial, s)
        control_paths[s] = result

    # v91.1.0 — supervisor metrics and forbidden strata.
    sup_metrics = supervisor.get("metrics")
    if sup_metrics is None:
        sup_metrics = compute_supervisor_metrics(dfa, supervisor["supervised_dfa"])
    forbidden_strata = supervisor.get("forbidden_strata")
    if forbidden_strata is None:
        forbidden_strata = stratify_forbidden_states(
            supervisor["forbidden_states"], supervisor["reasons"],
        )

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
        "attractor_hierarchy": {
            **hierarchy,
            "transition_summary": transition_summary,
        },
        "invariant_constraints": {
            "avoid_states": sorted(inv_constraints.get("avoid_states", set())),
            "allow_only_states": (
                sorted(inv_constraints["allow_only_states"])
                if inv_constraints.get("allow_only_states") is not None
                else None
            ),
            "has_constraints": inv_has_constraints,
            "applied": False,
        },
        "supervisor": {
            "supervised_dfa": supervisor["supervised_dfa"],
            "forbidden_states": supervisor["forbidden_states"],
            "disabled_transitions": supervisor["disabled_transitions"],
            "n_pruned_states": supervisor["n_pruned_states"],
            "n_disabled_transitions": supervisor["n_disabled_transitions"],
            "reasons": supervisor["reasons"],
            "policy": supervisor.get("policy", {}),
        },
        "constraint_bundle": constraint_bundle,
        # v91.1.0 — observability.
        "supervisor_metrics": sup_metrics,
        "forbidden_strata": forbidden_strata,
    }
