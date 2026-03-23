"""v86.3.0 — Phase Syndrome Analysis (Discrete Invariant Signatures).

Extracts discrete syndrome representations from continuous phase behaviour.
Mirrors QEC theory: "measure the error, not the state."

Each result is reduced to a 4-bit syndrome encoding:
    [class_consistent, phase_consistent, structure_consistent, stability_ok]

Syndrome transitions and Hamming distances quantify discrete phase changes.

Layer 5 — Experiments.
Does not modify decoder internals.  Fully deterministic.  Read-only on inputs.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Step 1 — Extract syndrome from a single result
# ---------------------------------------------------------------------------


def extract_phase_syndrome(
    result: Dict[str, Any],
    *,
    target_class: Optional[str] = None,
) -> Dict[str, bool]:
    """Extract a 4-bit discrete syndrome from a single result.

    Parameters
    ----------
    result:
        A result dict containing ``"best_pair"`` (from ``run_target_sweep``)
        or direct fields (``class``, ``phase``, ``compatibility``, ``score``,
        optionally ``normalized_score``).
    target_class:
        If provided, ``class_consistent`` checks whether the result class
        matches this target.  If *None*, defaults to ``True``.

    Returns
    -------
    dict with keys ``class_consistent``, ``phase_consistent``,
    ``structure_consistent``, ``stability_ok``.
    """
    # Unwrap best_pair if present.
    if "best_pair" in result:
        pair = result["best_pair"]
    else:
        pair = result

    # --- class_consistent ---
    if target_class is not None:
        class_consistent = pair.get("class") == target_class
    else:
        class_consistent = True

    # --- phase_consistent ---
    # A result is phase-consistent when its phase is in one of the
    # stable categories (not in an unstable or chaotic transition).
    phase = pair.get("phase", "")
    phase_consistent = phase not in ("unstable_region", "chaotic_transition")

    # --- structure_consistent ---
    compatibility = pair.get("compatibility", 0.0)
    structure_consistent = compatibility >= 0.5

    # --- stability_ok ---
    if "normalized_score" in pair:
        stability_ok = pair["normalized_score"] >= 0.5
    else:
        stability_ok = pair.get("score", 0.0) >= 0.0

    return {
        "class_consistent": bool(class_consistent),
        "phase_consistent": bool(phase_consistent),
        "structure_consistent": bool(structure_consistent),
        "stability_ok": bool(stability_ok),
    }


# ---------------------------------------------------------------------------
# Step 2 — Encode syndrome to compact representation
# ---------------------------------------------------------------------------

_SYNDROME_ORDER = ("class_consistent", "phase_consistent",
                   "structure_consistent", "stability_ok")


def encode_syndrome(syndrome: Dict[str, bool]) -> str:
    """Encode a syndrome dict to a 4-character binary string.

    Bit ordering: ``[class, phase, structure, stability]``.
    ``"1"`` = True, ``"0"`` = False.
    """
    return "".join("1" if syndrome[k] else "0" for k in _SYNDROME_ORDER)


# ---------------------------------------------------------------------------
# Step 3 — Extract syndrome series from a list of results
# ---------------------------------------------------------------------------


def extract_syndrome_series(
    results: List[Dict[str, Any]],
    *,
    target_class: Optional[str] = None,
) -> Dict[str, Any]:
    """Extract syndromes and encodings for an ordered list of results.

    Parameters
    ----------
    results:
        Ordered list of result dicts (e.g. from ``run_target_sweep``).
    target_class:
        Forwarded to :func:`extract_phase_syndrome`.

    Returns
    -------
    dict with ``syndromes``, ``encoded``, ``n_steps``.
    """
    syndromes: List[Dict[str, bool]] = []
    encoded: List[str] = []
    for r in results:
        s = extract_phase_syndrome(r, target_class=target_class)
        syndromes.append(s)
        encoded.append(encode_syndrome(s))
    return {
        "syndromes": syndromes,
        "encoded": encoded,
        "n_steps": len(results),
    }


# ---------------------------------------------------------------------------
# Step 4 — Detect syndrome transitions
# ---------------------------------------------------------------------------


def _hamming_distance(a: str, b: str) -> int:
    """Compute Hamming distance between two equal-length binary strings."""
    return sum(ca != cb for ca, cb in zip(a, b))


def detect_syndrome_transitions(
    encoded: List[str],
) -> List[Dict[str, Any]]:
    """Detect transitions between consecutive syndrome encodings.

    Returns a list of transition records with ``index``, ``from``, ``to``,
    and ``hamming_distance``.
    """
    transitions: List[Dict[str, Any]] = []
    for i in range(len(encoded) - 1):
        if encoded[i] != encoded[i + 1]:
            transitions.append({
                "index": i,
                "from": encoded[i],
                "to": encoded[i + 1],
                "hamming_distance": _hamming_distance(encoded[i], encoded[i + 1]),
            })
    return transitions


# ---------------------------------------------------------------------------
# Step 5 — Full syndrome analysis (convenience)
# ---------------------------------------------------------------------------


def run_syndrome_analysis(
    results: List[Dict[str, Any]],
    *,
    target_class: Optional[str] = None,
) -> Dict[str, Any]:
    """Run full syndrome extraction and transition detection.

    Returns dict with ``series`` and ``transitions``.
    """
    series = extract_syndrome_series(results, target_class=target_class)
    transitions = detect_syndrome_transitions(series["encoded"])
    return {
        "series": series,
        "transitions": transitions,
    }
