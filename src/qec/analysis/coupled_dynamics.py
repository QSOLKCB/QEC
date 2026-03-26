"""v102.9.0 — Coupled dynamics and interaction modeling.

Models interactions between strategies:
- joint transitions (pair dynamics)
- coupling strength between strategy pairs
- synchronization detection
- coupled phase behavior (alignment from multistate)
- unified interaction summary

All functions are:
- deterministic (identical inputs -> identical outputs)
- side-effect free (no mutation of inputs)
- bounded outputs

Dependencies: stdlib only.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ROUND_PRECISION = 12

SYNC_HIGH = 0.8
SYNC_PARTIAL = 0.5

ALIGNMENT_STRONG = 0.7
ALIGNMENT_WEAK = 0.4


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _round(value: float) -> float:
    """Round to ``ROUND_PRECISION`` decimal places."""
    return round(float(value), ROUND_PRECISION)


def _sorted_pair(a: str, b: str) -> Tuple[str, str]:
    """Return a canonically ordered pair for deterministic keying."""
    if a <= b:
        return (a, b)
    return (b, a)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_joint_transitions(
    trajectories: Dict[str, List[str]],
) -> Dict[Tuple[str, str], Dict[Tuple[Tuple[str, str], Tuple[str, str]], int]]:
    """Extract joint transition counts for all strategy pairs.

    For each pair of strategies (A, B) and each timestep *t*, records the
    joint transition ``(A_t, B_t) -> (A_{t+1}, B_{t+1})``.

    Parameters
    ----------
    trajectories : dict
        Keyed by strategy name.  Each value is a list of type strings
        in run order (output of ``build_type_trajectory``).

    Returns
    -------
    dict
        Keyed by canonically ordered strategy pair ``(A, B)``.
        Each value is a dict mapping
        ``((A_state, B_state), (A_next, B_next))`` to an integer count.
    """
    names = sorted(trajectories.keys())
    result: Dict[
        Tuple[str, str],
        Dict[Tuple[Tuple[str, str], Tuple[str, str]], int],
    ] = {}

    for i, name_a in enumerate(names):
        for name_b in names[i + 1:]:
            pair = _sorted_pair(name_a, name_b)
            seq_a = trajectories[name_a]
            seq_b = trajectories[name_b]
            length = min(len(seq_a), len(seq_b))

            counts: Dict[Tuple[Tuple[str, str], Tuple[str, str]], int] = {}
            for t in range(length - 1):
                state = (seq_a[t], seq_b[t])
                next_state = (seq_a[t + 1], seq_b[t + 1])
                key = (state, next_state)
                counts[key] = counts.get(key, 0) + 1

            result[pair] = counts

    return result


def compute_coupling_strength(
    joint_transitions: Dict[
        Tuple[str, str],
        Dict[Tuple[Tuple[str, str], Tuple[str, str]], int],
    ],
) -> Dict[Tuple[str, str], float]:
    """Compute coupling strength for each strategy pair.

    Coupling strength is defined as::

        unique_joint_transitions / (1 + total_transitions)

    Low values indicate independent strategies; high values indicate
    strongly coupled dynamics.

    Parameters
    ----------
    joint_transitions : dict
        Output of ``build_joint_transitions``.

    Returns
    -------
    dict
        Keyed by strategy pair.  Each value is a float in [0, 1).
    """
    result: Dict[Tuple[str, str], float] = {}

    for pair in sorted(joint_transitions.keys()):
        counts = joint_transitions[pair]
        total = sum(counts.values())
        unique = len(counts)
        strength = _round(unique / (1 + total))
        result[pair] = strength

    return result


def detect_synchronization(
    trajectories: Dict[str, List[str]],
) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """Detect synchronization between strategy pairs.

    For each pair, counts the fraction of timesteps where both strategies
    are in the same state.

    Classification:
    - sync_ratio > 0.8: ``"synchronized"``
    - 0.5 <= sync_ratio <= 0.8: ``"partially_synchronized"``
    - sync_ratio < 0.5: ``"independent"``

    Parameters
    ----------
    trajectories : dict
        Keyed by strategy name.  Each value is a list of type strings.

    Returns
    -------
    dict
        Keyed by canonically ordered strategy pair.  Each value contains
        ``"sync_ratio"`` (float) and ``"classification"`` (str).
    """
    names = sorted(trajectories.keys())
    result: Dict[Tuple[str, str], Dict[str, Any]] = {}

    for i, name_a in enumerate(names):
        for name_b in names[i + 1:]:
            pair = _sorted_pair(name_a, name_b)
            seq_a = trajectories[name_a]
            seq_b = trajectories[name_b]
            length = min(len(seq_a), len(seq_b))

            if length == 0:
                result[pair] = {
                    "sync_ratio": _round(0.0),
                    "classification": "independent",
                }
                continue

            matches = sum(1 for t in range(length) if seq_a[t] == seq_b[t])
            ratio = _round(matches / length)

            if ratio > SYNC_HIGH:
                classification = "synchronized"
            elif ratio >= SYNC_PARTIAL:
                classification = "partially_synchronized"
            else:
                classification = "independent"

            result[pair] = {
                "sync_ratio": ratio,
                "classification": classification,
            }

    return result


def classify_coupled_phase(
    multistate: Dict[str, Dict[str, Any]],
) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """Classify coupled phase behavior from multistate outputs.

    For each pair (A, B), compares their phase membership distributions
    and computes overlap::

        overlap = sum(min(A[k], B[k]) for k in all_states)

    Classification:
    - overlap > 0.7: ``"strongly_aligned"``
    - 0.4 <= overlap <= 0.7: ``"weakly_aligned"``
    - overlap < 0.4: ``"divergent"``

    Parameters
    ----------
    multistate : dict
        Keyed by strategy name.  Each value must contain a
        ``"membership"`` dict mapping phase state names to floats.

    Returns
    -------
    dict
        Keyed by canonically ordered strategy pair.  Each value contains
        ``"overlap"`` (float) and ``"alignment"`` (str).
    """
    names = sorted(multistate.keys())
    result: Dict[Tuple[str, str], Dict[str, Any]] = {}

    for i, name_a in enumerate(names):
        for name_b in names[i + 1:]:
            pair = _sorted_pair(name_a, name_b)
            mem_a = multistate[name_a].get("membership", {})
            mem_b = multistate[name_b].get("membership", {})

            all_keys = sorted(set(list(mem_a.keys()) + list(mem_b.keys())))
            overlap = _round(
                sum(min(mem_a.get(k, 0.0), mem_b.get(k, 0.0)) for k in all_keys)
            )

            if overlap > ALIGNMENT_STRONG:
                alignment = "strongly_aligned"
            elif overlap >= ALIGNMENT_WEAK:
                alignment = "weakly_aligned"
            else:
                alignment = "divergent"

            result[pair] = {
                "overlap": overlap,
                "alignment": alignment,
            }

    return result


def build_coupled_summary(
    coupling_strength: Dict[Tuple[str, str], float],
    synchronization: Dict[Tuple[str, str], Dict[str, Any]],
    coupled_phase: Dict[Tuple[str, str], Dict[str, Any]],
) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """Build a unified interaction summary for all strategy pairs.

    Parameters
    ----------
    coupling_strength : dict
        Output of ``compute_coupling_strength``.
    synchronization : dict
        Output of ``detect_synchronization``.
    coupled_phase : dict
        Output of ``classify_coupled_phase``.

    Returns
    -------
    dict
        Keyed by canonically ordered strategy pair.  Each value contains:

        - ``coupling_strength`` : float
        - ``sync_ratio`` : float
        - ``sync_classification`` : str
        - ``overlap`` : float
        - ``alignment`` : str
    """
    all_pairs = sorted(
        set(
            list(coupling_strength.keys())
            + list(synchronization.keys())
            + list(coupled_phase.keys())
        )
    )

    result: Dict[Tuple[str, str], Dict[str, Any]] = {}

    for pair in all_pairs:
        sync = synchronization.get(pair, {})
        phase = coupled_phase.get(pair, {})
        result[pair] = {
            "coupling_strength": coupling_strength.get(pair, 0.0),
            "sync_ratio": sync.get("sync_ratio", 0.0),
            "sync_classification": sync.get("classification", "independent"),
            "overlap": phase.get("overlap", 0.0),
            "alignment": phase.get("alignment", "divergent"),
        }

    return result


__all__ = [
    "ALIGNMENT_STRONG",
    "ALIGNMENT_WEAK",
    "ROUND_PRECISION",
    "SYNC_HIGH",
    "SYNC_PARTIAL",
    "build_coupled_summary",
    "build_joint_transitions",
    "classify_coupled_phase",
    "compute_coupling_strength",
    "detect_synchronization",
]
