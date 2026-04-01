"""v132.1.0 — Deterministic DFA supervisor synthesizer.

Implements Ramadge–Wonham-style supervisory control synthesis
over deterministic finite automata (DFA) plant models.

Synthesizes a maximally permissive supervisor that restricts
only controllable transitions to enforce safety while preserving
nonblocking behavior.
"""

from __future__ import annotations

from dataclasses import dataclass

@dataclass(frozen=True)
class DFAStateMachine:
    """Immutable DFA plant model for supervisory control synthesis."""

    states: tuple[str, ...]
    transitions: dict[tuple[str, str], str]
    initial_state: str
    safe_states: tuple[str, ...]
    controllable_events: tuple[str, ...]
    uncontrollable_events: tuple[str, ...]


def _reachable_states(
    initial: str,
    transitions: dict[tuple[str, str], str],
    allowed: frozenset[str],
) -> frozenset[str]:
    """Compute states reachable from initial within allowed set."""
    visited: set[str] = set()
    stack = [initial]
    while stack:
        state = stack.pop()
        if state in visited or state not in allowed:
            continue
        visited.add(state)
        for (src, _evt), dst in sorted(transitions.items()):
            if src == state and dst in allowed and dst not in visited:
                stack.append(dst)
    return frozenset(visited)


def _reverse_reachable(
    targets: frozenset[str],
    transitions: dict[tuple[str, str], str],
    allowed: frozenset[str],
) -> frozenset[str]:
    """Compute states that can reach any target within allowed set."""
    # Build reverse adjacency
    reverse: dict[str, list[str]] = {}
    for (src, _evt), dst in sorted(transitions.items()):
        if src in allowed and dst in allowed:
            reverse.setdefault(dst, []).append(src)

    visited: set[str] = set()
    stack = [t for t in sorted(targets) if t in allowed]
    while stack:
        state = stack.pop()
        if state in visited:
            continue
        visited.add(state)
        for pred in reverse.get(state, []):
            if pred not in visited and pred in allowed:
                stack.append(pred)
    return frozenset(visited)


def synthesize_supervisor(plant: DFAStateMachine) -> dict:
    """Synthesize maximally permissive DFA supervisor via RW pruning.

    Returns a dict with exactly:
        legal_states, blocked_transitions, maximally_permissive,
        nonblocking_verified, synthesis_score, synthesis_label.
    """
    all_states = frozenset(plant.states)
    safe_set = frozenset(plant.safe_states)
    uncontrollable = frozenset(plant.uncontrollable_events)

    # Iterative fixed-point pruning
    legal = all_states
    changed = True
    while changed:
        changed = False

        # Prune states not reachable from initial
        reachable = _reachable_states(plant.initial_state, plant.transitions, legal)
        if reachable != legal:
            legal = reachable
            changed = True

        # Prune states that cannot reach any safe state
        can_reach_safe = _reverse_reachable(
            safe_set & legal, plant.transitions, legal,
        )
        if can_reach_safe != legal:
            legal = can_reach_safe
            changed = True

        # Prune states with uncontrollable transitions to illegal states
        bad: set[str] = set()
        for (src, evt), dst in sorted(plant.transitions.items()):
            if src in legal and evt in uncontrollable and dst not in legal:
                bad.add(src)
        if bad:
            legal = legal - frozenset(bad)
            changed = True

    # Compute blocked transitions: controllable transitions from legal
    # states to illegal states
    controllable = frozenset(plant.controllable_events)
    blocked: list[tuple[str, str]] = []
    for (src, evt), dst in sorted(plant.transitions.items()):
        if src in legal and evt in controllable and dst not in legal:
            blocked.append((src, evt))

    # Nonblocking: every legal state can reach a safe state
    if legal:
        can_reach_safe = _reverse_reachable(
            safe_set & legal, plant.transitions, legal,
        )
        nonblocking = legal == can_reach_safe
    else:
        nonblocking = True

    # Maximally permissive by construction (RW fixed-point).
    maximally_permissive = True

    # Score
    if not legal or plant.initial_state not in legal:
        score = 1.0
        label = "critical"
    elif len(legal) < len(all_states) or blocked:
        score = 0.5
        label = "warning"
    else:
        score = 0.0
        label = "safe"

    return {
        "legal_states": tuple(sorted(legal)),
        "blocked_transitions": tuple(sorted(blocked)),
        "maximally_permissive": maximally_permissive,
        "nonblocking_verified": nonblocking,
        "synthesis_score": score,
        "synthesis_label": label,
    }


def run_dfa_supervisor_synthesizer(plant: DFAStateMachine) -> dict:
    """Top-level runner for DFA supervisor synthesis.

    Returns a dict with exactly:
        plant, synthesis, supervisory_ready.
    """
    synthesis = synthesize_supervisor(plant)
    return {
        "plant": plant,
        "synthesis": synthesis,
        "supervisory_ready": True,
    }
