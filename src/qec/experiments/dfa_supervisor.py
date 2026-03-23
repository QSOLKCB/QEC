"""Deterministic supervisory control for DFA structures (v91.1.0).

Derives forbidden states from provable invariant structure, disables
unsafe transitions, and synthesizes the maximally permissive safe sub-DFA
by iterative trimming. All algorithms are pure, deterministic, and
non-mutating.

Constraint composition order:
    user ⊕ invariant ⊕ supervisor ⊕ policy

No randomness, no probabilities, no heuristics.
stdlib + collections only.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Dict, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# PART A — Constraint Visibility / Provenance
# ---------------------------------------------------------------------------


def structure_invariant_constraints(
    inv_constraints: Dict[str, Any],
    inv: Dict[str, Any],
) -> Dict[str, Any]:
    """Upgrade raw invariant constraints to structured form with provenance.

    Returns canonical dict with avoid_states, allow_only_states, and
    per-state source annotations.
    """
    sections = inv.get("invariants", inv) if isinstance(inv, dict) else {}

    avoid_states = sorted(inv_constraints.get("avoid_states", set()))
    raw_allow = inv_constraints.get("allow_only_states")
    allow_only_states = sorted(raw_allow) if raw_allow is not None else None

    # Build provenance for avoid_states.
    avoid_sources: Dict[int, List[str]] = {}
    dead_info = sections.get("dead_state", {})
    dead_state = dead_info.get("dead_state")
    structure = sections.get("structure", {})
    state_to_attractor = structure.get("state_to_attractor", {})

    for s in avoid_states:
        reasons: List[str] = []
        if dead_state is not None and s == dead_state:
            reasons.append("dead_state")
        elif dead_state is not None and dead_info.get("has_dead_state"):
            dead_att = state_to_attractor.get(dead_state)
            s_att = state_to_attractor.get(s)
            if dead_att is not None and s_att == dead_att:
                reasons.append("drain_basin")
        if not reasons:
            reasons.append("forbidden_region")
        avoid_sources[s] = reasons

    # Build provenance for allow_only_states.
    allow_sources: Dict[int, List[str]] = {}
    if allow_only_states is not None:
        for s in allow_only_states:
            allow_sources[s] = ["total_attractor_mapping"]

    return {
        "avoid_states": avoid_states,
        "allow_only_states": allow_only_states,
        "sources": {
            "avoid_states": {k: v for k, v in sorted(avoid_sources.items())},
            "allow_only_states": {k: v for k, v in sorted(allow_sources.items())},
        },
    }


def normalize_constraint_bundle(
    bundle: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Normalize a constraint bundle to canonical schema.

    Ensures all sections (user, invariant, supervisor, policy, composed)
    always exist.
    """
    if bundle is None:
        bundle = {}

    empty_section: Dict[str, Any] = {
        "avoid_states": [],
        "allow_only_states": None,
    }

    result: Dict[str, Any] = {}
    for key in ("user", "invariant", "supervisor", "policy", "composed"):
        section = bundle.get(key)
        if section is None:
            result[key] = dict(empty_section)
        else:
            result[key] = dict(section)
    return result


# ---------------------------------------------------------------------------
# PART B — Forbidden State Extraction
# ---------------------------------------------------------------------------


def derive_forbidden_states(
    dfa: Dict[str, Any],
    invariants: Dict[str, Any],
    invariant_constraints: Dict[str, Any],
) -> Dict[str, Any]:
    """Derive forbidden states from provable invariant structure.

    Sources:
    - dead_state: the absorbing dead state itself
    - dead_drain: states proven to lead only to dead state
    - outside_allowed_region: states outside allow_only if total and valid

    Returns {"forbidden_states": sorted list, "reasons": {state: [reasons]}}.
    """
    forbidden: Set[int] = set()
    reasons: Dict[int, List[str]] = {}
    all_states = set(dfa.get("states", []))

    sections = invariants.get("invariants", invariants) if isinstance(invariants, dict) else {}
    dead_info = sections.get("dead_state", {})
    dead_state = dead_info.get("dead_state")
    structure = sections.get("structure", {})
    state_to_attractor = structure.get("state_to_attractor", {})

    # Source 1: dead state itself.
    if dead_state is not None and dead_info.get("is_absorbing"):
        forbidden.add(dead_state)
        reasons.setdefault(dead_state, []).append("dead_state")

    # Source 2: states that drain only to dead state.
    if dead_state is not None and dead_info.get("has_dead_state"):
        dead_att = state_to_attractor.get(dead_state)
        if dead_att is not None:
            for s in sorted(all_states):
                if s == dead_state:
                    continue
                s_att = state_to_attractor.get(s)
                if s_att == dead_att:
                    forbidden.add(s)
                    reasons.setdefault(s, []).append("dead_drain")

    # Source 3: states from invariant-derived avoid_states.
    for s in invariant_constraints.get("avoid_states", []):
        if isinstance(s, int) and s in all_states and s not in forbidden:
            forbidden.add(s)
            reasons.setdefault(s, []).append("invariant_avoid")

    # Source 4: outside allowed region.
    allow_only = invariant_constraints.get("allow_only_states")
    if allow_only is not None:
        allow_set = set(allow_only)
        for s in sorted(all_states):
            if s not in allow_set and s not in forbidden:
                forbidden.add(s)
                reasons.setdefault(s, []).append("outside_allowed_region")

    return {
        "forbidden_states": sorted(forbidden),
        "reasons": {k: v for k, v in sorted(reasons.items())},
    }


# ---------------------------------------------------------------------------
# PART C — Safe Transition Pruning
# ---------------------------------------------------------------------------


def prune_unsafe_transitions(
    dfa: Dict[str, Any],
    forbidden_states: List[int],
) -> Tuple[Dict[str, Any], List[Tuple[int, int, int]]]:
    """Remove transitions whose target is forbidden.

    Returns (pruned_dfa, disabled_transitions).
    Does NOT mutate the input DFA.
    """
    forbidden_set = set(forbidden_states)
    transitions = dfa.get("transitions", {})
    states = sorted(dfa.get("states", []))
    alphabet = sorted(dfa.get("alphabet", []))

    new_transitions: Dict[int, Dict[int, int]] = {}
    disabled: List[Tuple[int, int, int]] = []

    for s in states:
        state_trans = transitions.get(s, {})
        new_state_trans: Dict[int, int] = {}
        for sym in sorted(state_trans):
            ns = state_trans[sym]
            if ns in forbidden_set:
                disabled.append((s, sym, ns))
            else:
                new_state_trans[sym] = ns
        if new_state_trans:
            new_transitions[s] = new_state_trans

    pruned_dfa = {
        "states": list(states),
        "alphabet": list(alphabet),
        "transitions": new_transitions,
        "initial_state": dfa.get("initial_state", 0),
        "dead_state": dfa.get("dead_state"),
    }

    return pruned_dfa, sorted(disabled)


# ---------------------------------------------------------------------------
# PART D — Fixed-Point Supervisor Synthesis
# ---------------------------------------------------------------------------


def trim_unreachable(dfa: Dict[str, Any]) -> Dict[str, Any]:
    """Remove states not reachable from initial state via forward BFS.

    Returns a new DFA with only reachable states.
    """
    initial = dfa.get("initial_state", 0)
    transitions = dfa.get("transitions", {})

    reachable: Set[int] = set()
    queue: deque[int] = deque([initial])
    reachable.add(initial)

    while queue:
        s = queue.popleft()
        for sym in sorted(transitions.get(s, {})):
            ns = transitions[s][sym]
            if ns not in reachable:
                reachable.add(ns)
                queue.append(ns)

    # Build trimmed DFA.
    new_transitions: Dict[int, Dict[int, int]] = {}
    for s in sorted(reachable):
        state_trans = transitions.get(s, {})
        filtered = {sym: ns for sym, ns in sorted(state_trans.items()) if ns in reachable}
        if filtered:
            new_transitions[s] = filtered

    dead_state = dfa.get("dead_state")
    if dead_state is not None and dead_state not in reachable:
        dead_state = None

    return {
        "states": sorted(reachable),
        "alphabet": sorted(dfa.get("alphabet", [])),
        "transitions": new_transitions,
        "initial_state": initial,
        "dead_state": dead_state,
    }


def trim_noncoaccessible(
    dfa: Dict[str, Any],
    safe_targets: Optional[Set[int]] = None,
) -> Dict[str, Any]:
    """Remove states that cannot reach any safe target via reverse BFS.

    If safe_targets is None, uses all states that have no outgoing
    transitions to forbidden/removed states (i.e., all remaining states
    with at least one transition or terminal states).
    """
    transitions = dfa.get("transitions", {})
    all_states = set(dfa.get("states", []))

    if safe_targets is None:
        # Use all states as safe targets (no further pruning).
        safe_targets = set(all_states)

    if not safe_targets:
        # Nothing is safe — return empty DFA.
        return {
            "states": [],
            "alphabet": sorted(dfa.get("alphabet", [])),
            "transitions": {},
            "initial_state": dfa.get("initial_state", 0),
            "dead_state": None,
        }

    # Reverse BFS from safe targets.
    reverse: Dict[int, Set[int]] = {s: set() for s in all_states}
    for s in sorted(all_states):
        for sym in sorted(transitions.get(s, {})):
            ns = transitions[s][sym]
            if ns in all_states:
                reverse[ns].add(s)

    coaccessible: Set[int] = set()
    queue: deque[int] = deque()
    for t in sorted(safe_targets):
        if t in all_states:
            coaccessible.add(t)
            queue.append(t)

    while queue:
        s = queue.popleft()
        for pred in sorted(reverse.get(s, set())):
            if pred not in coaccessible:
                coaccessible.add(pred)
                queue.append(pred)

    # Build trimmed DFA.
    new_transitions: Dict[int, Dict[int, int]] = {}
    for s in sorted(coaccessible):
        state_trans = transitions.get(s, {})
        filtered = {sym: ns for sym, ns in sorted(state_trans.items()) if ns in coaccessible}
        if filtered:
            new_transitions[s] = filtered

    dead_state = dfa.get("dead_state")
    if dead_state is not None and dead_state not in coaccessible:
        dead_state = None

    return {
        "states": sorted(coaccessible),
        "alphabet": sorted(dfa.get("alphabet", [])),
        "transitions": new_transitions,
        "initial_state": dfa.get("initial_state", 0),
        "dead_state": dead_state,
    }


def synthesize_supervisor(
    dfa: Dict[str, Any],
    invariants: Dict[str, Any],
    invariant_constraints: Dict[str, Any],
) -> Dict[str, Any]:
    """Synthesize maximally permissive safe sub-DFA by iterative trimming.

    Algorithm (fixed-point iteration):
    1. Derive forbidden states
    2. Prune unsafe transitions
    3. Trim unreachable states
    4. Trim non-coaccessible states
    5. If new dead-end states emerge, treat as forbidden and repeat

    Returns supervisor result with supervised_dfa, forbidden states,
    disabled transitions, and provenance.
    """
    all_forbidden: Set[int] = set()
    all_disabled: List[Tuple[int, int, int]] = []
    all_reasons: Dict[int, List[str]] = {}

    # Initial forbidden state derivation.
    fb = derive_forbidden_states(dfa, invariants, invariant_constraints)
    for s in fb["forbidden_states"]:
        all_forbidden.add(s)
    for s, r in fb["reasons"].items():
        all_reasons.setdefault(s, []).extend(r)

    current_dfa = dfa
    max_iterations = len(dfa.get("states", [])) + 1

    for _ in range(max_iterations):
        # Prune transitions to forbidden states.
        pruned_dfa, disabled = prune_unsafe_transitions(current_dfa, sorted(all_forbidden))
        all_disabled.extend(disabled)

        # Trim unreachable.
        trimmed = trim_unreachable(pruned_dfa)

        # Trim non-coaccessible (states that can't reach any safe state).
        safe_states = set(trimmed.get("states", []))
        safe_targets = safe_states - all_forbidden
        if safe_targets:
            trimmed = trim_noncoaccessible(trimmed, safe_targets)
        else:
            trimmed = trim_noncoaccessible(trimmed, None)

        # Check for new dead-end states (states with no outgoing transitions
        # that are not terminal attractors).
        transitions = trimmed.get("transitions", {})
        remaining_states = set(trimmed.get("states", []))
        new_forbidden: Set[int] = set()

        for s in sorted(remaining_states):
            state_trans = transitions.get(s, {})
            # Filter to transitions within remaining states.
            valid_targets = {ns for ns in state_trans.values() if ns in remaining_states}
            if not valid_targets and s != trimmed.get("initial_state"):
                # Dead-end: no outgoing transitions (not initial state).
                # Only mark as forbidden if it has no self-loop.
                if s not in state_trans.values():
                    new_forbidden.add(s)

        if not new_forbidden:
            # Fixed point reached.
            current_dfa = trimmed
            break

        for s in new_forbidden:
            all_forbidden.add(s)
            all_reasons.setdefault(s, []).append("dead_end_trimmed")

        current_dfa = trimmed
    else:
        current_dfa = trimmed

    # Compute final metrics.
    original_states = set(dfa.get("states", []))
    final_states = set(current_dfa.get("states", []))
    n_pruned = len(original_states) - len(final_states)

    # Deduplicate disabled transitions.
    unique_disabled = sorted(set(all_disabled))

    return {
        "supervised_dfa": current_dfa,
        "forbidden_states": sorted(all_forbidden),
        "disabled_transitions": unique_disabled,
        "n_pruned_states": n_pruned,
        "n_disabled_transitions": len(unique_disabled),
        "reasons": {k: v for k, v in sorted(all_reasons.items())},
    }


# ---------------------------------------------------------------------------
# PART E — Supervisor Policy
# ---------------------------------------------------------------------------


def extract_supervisor_policy(
    supervised_dfa: Dict[str, Any],
) -> Dict[int, List[int]]:
    """Extract allowed symbols per state from supervised DFA.

    Returns {state_id: [allowed_symbol_1, ...]} with sorted values.
    """
    transitions = supervised_dfa.get("transitions", {})
    policy: Dict[int, List[int]] = {}

    for s in sorted(supervised_dfa.get("states", [])):
        state_trans = transitions.get(s, {})
        policy[s] = sorted(state_trans.keys())

    return policy


def derive_policy_constraints(
    policy: Dict[int, List[int]],
) -> Dict[str, Any]:
    """Derive informational constraints from supervisor policy.

    Returns {"allowed_symbols_by_state": {state: [symbols]}}.
    Informational only in v90.0.0 (no soft weighting).
    """
    return {
        "allowed_symbols_by_state": {
            k: list(v) for k, v in sorted(policy.items())
        },
    }


# ---------------------------------------------------------------------------
# PART F — Supervisor Metrics
# ---------------------------------------------------------------------------


def compute_supervisor_metrics(
    original_dfa: Dict[str, Any],
    supervised_dfa: Dict[str, Any],
) -> Dict[str, Any]:
    """Compute deterministic comparison metrics between original and supervised DFA.

    Returns:
        Dict with states_before, states_after, transitions_before, transitions_after,
        density_before, density_after, control_strength.
    """
    orig_states = len(original_dfa.get("states", []))
    sup_states = len(supervised_dfa.get("states", []))
    symbols = len(original_dfa.get("alphabet", []))

    # Count transitions.
    orig_trans = sum(
        len(t) for t in original_dfa.get("transitions", {}).values()
    )
    sup_trans = sum(
        len(t) for t in supervised_dfa.get("transitions", {}).values()
    )

    # density = transitions / (states * symbols), safe division.
    orig_denom = orig_states * symbols
    sup_denom = sup_states * symbols
    density_before = orig_trans / orig_denom if orig_denom > 0 else 0.0
    density_after = sup_trans / sup_denom if sup_denom > 0 else 0.0

    # control_strength = fraction of transitions removed.
    control_strength = (
        1.0 - (sup_trans / orig_trans) if orig_trans > 0 else 0.0
    )

    return {
        "states_before": orig_states,
        "states_after": sup_states,
        "transitions_before": orig_trans,
        "transitions_after": sup_trans,
        "density_before": density_before,
        "density_after": density_after,
        "control_strength": control_strength,
    }


# ---------------------------------------------------------------------------
# PART F2 — Forbidden State Stratification
# ---------------------------------------------------------------------------


def stratify_forbidden_states(
    forbidden_states: List[int],
    provenance: Dict[int, List[str]],
) -> Dict[str, Any]:
    """Partition forbidden states into hard and structural categories.

    Hard forbidden: states with provenance 'dead_state' or 'dead_drain'
        (unreachable / absorbing dead ends).
    Structural forbidden: states with provenance 'invariant_avoid',
        'outside_allowed_region', 'forbidden_region', 'dead_end_trimmed',
        'drain_basin' (constraint-based).

    Returns:
        {"hard_forbidden": sorted list, "structural_forbidden": sorted list}
    """
    hard_tags = {"dead_state", "dead_drain"}
    hard: List[int] = []
    structural: List[int] = []

    for s in sorted(forbidden_states):
        reasons = provenance.get(s, [])
        if any(r in hard_tags for r in reasons):
            hard.append(s)
        else:
            structural.append(s)

    return {
        "hard_forbidden": sorted(hard),
        "structural_forbidden": sorted(structural),
    }


# ---------------------------------------------------------------------------
# PART G1 — Run Supervisor (integration helper)
# ---------------------------------------------------------------------------


def run_supervisor(
    dfa: Dict[str, Any],
    invariants: Dict[str, Any],
    invariant_constraints: Dict[str, Any],
) -> Dict[str, Any]:
    """Run full supervisor pipeline: synthesis + policy extraction.

    Returns supervisor result enriched with policy, metrics, and strata.
    """
    result = synthesize_supervisor(dfa, invariants, invariant_constraints)
    policy = extract_supervisor_policy(result["supervised_dfa"])
    policy_constraints = derive_policy_constraints(policy)

    result["policy"] = policy
    result["policy_constraints"] = policy_constraints

    # v91.1.0 — supervisor metrics.
    result["metrics"] = compute_supervisor_metrics(dfa, result["supervised_dfa"])

    # v91.1.0 — forbidden state stratification.
    result["forbidden_strata"] = stratify_forbidden_states(
        result["forbidden_states"], result["reasons"],
    )

    return result
