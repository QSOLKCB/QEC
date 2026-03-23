"""v86.5.0 — Ternary & Geometric Syndrome Extension.

Upgrades binary syndrome representation (v86.3) to:
  - Ternary invariant states (-1, 0, +1)
  - Geometric (Euclidean) distance metric
  - Multi-layer encoding (binary + ternary, backward compatible)

Layer 5 — Experiments.
Does not modify decoder internals.  Fully deterministic.  Read-only on inputs.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

from .phase_syndrome_analysis import extract_phase_syndrome, encode_syndrome


# ---------------------------------------------------------------------------
# Step 1 — Ternary Syndrome Extraction
# ---------------------------------------------------------------------------

_STABLE_PHASES = frozenset({
    "stable", "locked", "convergent", "stable_region",
})
_UNSTABLE_PHASES = frozenset({
    "unstable_region", "chaotic_transition", "chaotic", "divergent",
})


def extract_ternary_syndrome(
    result: Dict[str, Any],
    *,
    target_class: Optional[str] = None,
) -> Dict[str, int]:
    """Extract a ternary (-1, 0, +1) syndrome from a single result.

    Parameters
    ----------
    result:
        A result dict containing ``"best_pair"`` or direct fields
        (``class``, ``phase``, ``compatibility``, ``score``,
        optionally ``normalized_score``).
    target_class:
        If provided, ``class_state`` checks match.  If *None*, defaults
        to ``0`` (neutral).

    Returns
    -------
    dict with ``class_state``, ``phase_state``, ``structure_state``,
    ``stability_state``, each in {-1, 0, +1}.
    """
    # Unwrap best_pair if present.
    if "best_pair" in result:
        pair = result["best_pair"]
    else:
        pair = result

    # --- class_state ---
    if target_class is not None:
        class_state = 1 if pair.get("class") == target_class else -1
    else:
        class_state = 0

    # --- phase_state ---
    phase = pair.get("phase", "")
    if phase in _STABLE_PHASES:
        phase_state = 1
    elif phase in _UNSTABLE_PHASES:
        phase_state = -1
    else:
        phase_state = 0

    # --- structure_state (compatibility) ---
    compatibility = float(pair.get("compatibility", 0.0))
    if compatibility >= 0.75:
        structure_state = 1
    elif compatibility >= 0.4:
        structure_state = 0
    else:
        structure_state = -1

    # --- stability_state (normalized_score, fallback to score) ---
    if "normalized_score" in pair:
        stability_val = float(pair["normalized_score"])
    else:
        stability_val = float(pair.get("score", 0.0))

    if stability_val >= 0.75:
        stability_state = 1
    elif stability_val >= 0.4:
        stability_state = 0
    else:
        stability_state = -1

    return {
        "class_state": class_state,
        "phase_state": phase_state,
        "structure_state": structure_state,
        "stability_state": stability_state,
    }


# ---------------------------------------------------------------------------
# Step 2 — Ternary Encoding
# ---------------------------------------------------------------------------

_TERNARY_ORDER = ("class_state", "phase_state",
                  "structure_state", "stability_state")

_TERNARY_CHAR = {1: "+", 0: "0", -1: "-"}


def encode_ternary_syndrome(
    syndrome: Dict[str, int],
) -> Tuple[Tuple[int, ...], str]:
    """Encode a ternary syndrome to tuple and string representations.

    Returns
    -------
    (tuple_form, string_form) where tuple_form is e.g. ``(1, 0, -1, 1)``
    and string_form is e.g. ``"+0-+"``.
    """
    values = tuple(syndrome[k] for k in _TERNARY_ORDER)
    chars = "".join(_TERNARY_CHAR[v] for v in values)
    return values, chars


# ---------------------------------------------------------------------------
# Step 3 — Geometric Distance
# ---------------------------------------------------------------------------


def compute_syndrome_distance(
    s1: Tuple[int, ...],
    s2: Tuple[int, ...],
) -> float:
    """Compute Euclidean distance between two ternary syndrome tuples.

    sqrt(sum((a - b)**2))  — replaces Hamming as primary distance.
    """
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(s1, s2)))


# ---------------------------------------------------------------------------
# Step 4 — Series Extraction
# ---------------------------------------------------------------------------


def extract_ternary_series(
    results: List[Dict[str, Any]],
    *,
    target_class: Optional[str] = None,
) -> Dict[str, Any]:
    """Extract ternary syndromes and encodings for an ordered result list.

    Returns
    -------
    dict with ``ternary`` (list of syndrome dicts), ``encoded`` (list of
    tuples), ``encoded_strings`` (list of strings), ``n_steps``.
    """
    ternary: List[Dict[str, int]] = []
    encoded: List[Tuple[int, ...]] = []
    encoded_strings: List[str] = []

    for r in results:
        s = extract_ternary_syndrome(r, target_class=target_class)
        t, c = encode_ternary_syndrome(s)
        ternary.append(s)
        encoded.append(t)
        encoded_strings.append(c)

    return {
        "ternary": ternary,
        "encoded": encoded,
        "encoded_strings": encoded_strings,
        "n_steps": len(results),
    }


# ---------------------------------------------------------------------------
# Step 5 — Geometric Transitions
# ---------------------------------------------------------------------------


def detect_geometric_transitions(
    encoded: List[Tuple[int, ...]],
) -> List[Dict[str, Any]]:
    """Detect transitions between consecutive ternary encodings.

    Returns list of transition records with ``index``, ``from``, ``to``,
    ``distance`` (Euclidean).
    """
    transitions: List[Dict[str, Any]] = []
    for i in range(len(encoded) - 1):
        if encoded[i] != encoded[i + 1]:
            transitions.append({
                "index": i,
                "from": encoded[i],
                "to": encoded[i + 1],
                "distance": compute_syndrome_distance(encoded[i], encoded[i + 1]),
            })
    return transitions


# ---------------------------------------------------------------------------
# Step 6 — Multi-Layer Encoding
# ---------------------------------------------------------------------------


def build_multilayer_syndrome(
    result: Dict[str, Any],
    *,
    target_class: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a multi-layer syndrome combining binary and ternary layers.

    Returns
    -------
    dict with ``binary`` (4-char string from v86.3) and ``ternary``
    (tuple of ints from v86.5).
    """
    # Binary layer (v86.3).
    binary_syn = extract_phase_syndrome(result, target_class=target_class)
    binary_str = encode_syndrome(binary_syn)

    # Ternary layer (v86.5).
    ternary_syn = extract_ternary_syndrome(result, target_class=target_class)
    ternary_tuple, _ = encode_ternary_syndrome(ternary_syn)

    return {
        "binary": binary_str,
        "ternary": ternary_tuple,
    }


# ---------------------------------------------------------------------------
# Step 7 — Convenience: full ternary syndrome geometry analysis
# ---------------------------------------------------------------------------


def run_syndrome_geometry_analysis(
    results: List[Dict[str, Any]],
    *,
    target_class: Optional[str] = None,
) -> Dict[str, Any]:
    """Run full ternary syndrome extraction and geometric transition detection.

    Returns dict with ``ternary_series`` and ``transitions``.
    """
    series = extract_ternary_series(results, target_class=target_class)
    transitions = detect_geometric_transitions(series["encoded"])
    return {
        "ternary_series": series,
        "transitions": transitions,
    }
