"""Deterministic Meta-Law Miner (v98.0.0).

Extracts higher-order patterns (meta-laws) from existing laws by
identifying co-occurring conditions, redundant laws, and shared
action clusters.

Meta-laws are deterministic patterns:
- co-occurring conditions across laws with the same action
- redundant law detection
- conflicting law class identification
- invariant condition sets

All algorithms are pure, deterministic, and use only stdlib + numpy.
No mutation of inputs. No randomness.
"""

from typing import Any, Dict, List, Tuple

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------

_NORM_DIGITS = 12
MIN_SUPPORT = 1
MIN_CONFIDENCE = 0.0


def _norm(x: float) -> float:
    """Normalize a float to fixed precision to avoid drift."""
    return round(float(x), _NORM_DIGITS)


# ---------------------------------------------------------------------------
# CONDITION SIGNATURE
# ---------------------------------------------------------------------------


def _condition_signature(condition: Any) -> str:
    """Deterministic string key for a condition object."""
    return f"{condition.metric}:{condition.operator}:{condition.value}"


def _law_condition_set(law: Any) -> Tuple[str, ...]:
    """Sorted tuple of condition signatures for a law."""
    sigs = [_condition_signature(c) for c in law.conditions]
    return tuple(sorted(sigs))


# ---------------------------------------------------------------------------
# STEP 1 — GROUP LAWS BY ACTION
# ---------------------------------------------------------------------------


def group_by_action(laws: List[Any]) -> Dict[str, List[Any]]:
    """Group laws by their action string.

    Returns {action: [laws]} with deterministic ordering.
    """
    groups: Dict[str, List[Any]] = {}
    for law in laws:
        action = law.action
        if action not in groups:
            groups[action] = []
        groups[action].append(law)
    return groups


# ---------------------------------------------------------------------------
# STEP 2 — EXTRACT SHARED CONDITIONS
# ---------------------------------------------------------------------------


def _extract_shared_conditions(
    laws: List[Any],
) -> List[str]:
    """Find condition signatures shared across all laws in a group."""
    if not laws:
        return []
    sets = [set(_law_condition_set(law)) for law in laws]
    shared = sets[0]
    for s in sets[1:]:
        shared = shared & s
    return sorted(shared)


# ---------------------------------------------------------------------------
# STEP 3 — EXTRACT CO-OCCURRING CONDITION PAIRS
# ---------------------------------------------------------------------------


def _extract_cooccurrences(
    laws: List[Any],
) -> Dict[Tuple[str, str], int]:
    """Count how often pairs of conditions co-occur across laws."""
    counts: Dict[Tuple[str, str], int] = {}
    for law in laws:
        sigs = sorted(set(_condition_signature(c) for c in law.conditions))
        for i in range(len(sigs)):
            for j in range(i + 1, len(sigs)):
                pair = (sigs[i], sigs[j])
                counts[pair] = counts.get(pair, 0) + 1
    return counts


# ---------------------------------------------------------------------------
# STEP 4 — BUILD META-LAWS
# ---------------------------------------------------------------------------


def _build_meta_law(
    conditions: List[str],
    action: str,
    support: int,
    total: int,
) -> Dict[str, Any]:
    """Build a single meta-law record."""
    confidence = _norm(support / total) if total > 0 else 0.0
    return {
        "conditions": list(conditions),
        "implies": action,
        "support": support,
        "confidence": confidence,
    }


# ---------------------------------------------------------------------------
# STEP 5 — DETECT REDUNDANT LAWS
# ---------------------------------------------------------------------------


def detect_redundant_laws(laws: List[Any]) -> List[Tuple[str, str]]:
    """Find pairs of laws with identical condition sets and actions.

    Returns sorted list of (law_id_a, law_id_b) pairs where a < b.
    """
    sig_map: Dict[Tuple[Tuple[str, ...], str], List[str]] = {}
    for law in laws:
        key = (_law_condition_set(law), law.action)
        if key not in sig_map:
            sig_map[key] = []
        sig_map[key].append(law.id)

    pairs: List[Tuple[str, str]] = []
    for key in sorted(sig_map.keys()):
        ids = sorted(sig_map[key])
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                pairs.append((ids[i], ids[j]))
    return pairs


# ---------------------------------------------------------------------------
# STEP 6 — DETECT CONFLICTING LAW CLASSES
# ---------------------------------------------------------------------------


def detect_conflicts(laws: List[Any]) -> List[Dict[str, Any]]:
    """Find groups of laws with overlapping conditions but different actions.

    Two laws conflict if they share the exact same condition set but
    prescribe different actions.

    Returns sorted list of conflict records.
    """
    sig_map: Dict[Tuple[str, ...], Dict[str, List[str]]] = {}
    for law in laws:
        cond_key = _law_condition_set(law)
        if cond_key not in sig_map:
            sig_map[cond_key] = {}
        action = law.action
        if action not in sig_map[cond_key]:
            sig_map[cond_key][action] = []
        sig_map[cond_key][action].append(law.id)

    conflicts: List[Dict[str, Any]] = []
    for cond_key in sorted(sig_map.keys()):
        actions = sig_map[cond_key]
        if len(actions) > 1:
            conflicts.append({
                "conditions": list(cond_key),
                "actions": {
                    a: sorted(ids) for a, ids in sorted(actions.items())
                },
            })
    return conflicts


# ---------------------------------------------------------------------------
# MAIN ENTRY POINT
# ---------------------------------------------------------------------------


def extract_meta_laws(laws: List[Any]) -> Dict[str, Any]:
    """Extract higher-order patterns from a list of laws.

    Steps:
    1. Group laws by action
    2. For each action group, identify shared conditions
    3. For each action group, identify co-occurring condition pairs
    4. Build meta-laws from shared conditions
    5. Detect redundant laws
    6. Detect conflicting law classes

    Returns a dict with:
    - meta_laws: list of meta-law records
    - redundant_pairs: list of (id_a, id_b) tuples
    - conflicts: list of conflict records
    - action_groups: count of laws per action
    """
    if not laws:
        return {
            "meta_laws": [],
            "redundant_pairs": [],
            "conflicts": [],
            "action_groups": {},
        }

    groups = group_by_action(laws)
    meta_laws: List[Dict[str, Any]] = []
    total_laws = len(laws)

    # Build meta-laws from shared conditions within each action group
    for action in sorted(groups.keys()):
        group = groups[action]
        support = len(group)

        # Shared conditions across all laws in the group
        shared = _extract_shared_conditions(group)
        if shared:
            meta_laws.append(
                _build_meta_law(shared, action, support, total_laws)
            )

        # Co-occurring pairs (only if group has enough laws)
        if len(group) >= 2:
            cooccurrences = _extract_cooccurrences(group)
            for pair in sorted(cooccurrences.keys()):
                count = cooccurrences[pair]
                if count >= 2:
                    pair_conditions = [pair[0], pair[1]]
                    # Only add if not a subset of an already-added shared set
                    if not shared or not all(
                        c in shared for c in pair_conditions
                    ):
                        meta_laws.append(
                            _build_meta_law(
                                pair_conditions, action, count, total_laws
                            )
                        )

    # Detect redundant laws and conflicts
    redundant = detect_redundant_laws(laws)
    conflicts = detect_conflicts(laws)

    return {
        "meta_laws": meta_laws,
        "redundant_pairs": redundant,
        "conflicts": conflicts,
        "action_groups": {
            a: len(g) for a, g in sorted(groups.items())
        },
    }
