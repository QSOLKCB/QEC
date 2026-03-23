"""v86.4.0 — Phase Syndrome Decoder (Symbolic Trajectory Classification).

Deterministic syndrome decoder: syndrome sequence -> regime classification.
Mirrors classical syndrome decoding: "given syndrome -> infer most likely state."

Layer 5 — Experiments.
Does not modify decoder internals.  Fully deterministic.  Read-only on inputs.
"""

from __future__ import annotations

from typing import Any, Dict, List


# ---------------------------------------------------------------------------
# Step 1 — Pattern Extraction
# ---------------------------------------------------------------------------


def _hamming_distance(a: str, b: str) -> int:
    """Compute Hamming distance between two equal-length binary strings."""
    return sum(ca != cb for ca, cb in zip(a, b))


def analyze_syndrome_patterns(
    encoded: List[str],
    transitions: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Extract deterministic pattern features from syndrome sequence.

    Parameters
    ----------
    encoded:
        Ordered list of syndrome binary strings (e.g. ``["1101", "1001"]``).
    transitions:
        Transition records with ``"from"``, ``"to"``, ``"hamming_distance"``.

    Returns
    -------
    dict with ``unique_count``, ``most_common``, ``max_run_length``,
    ``n_transitions``, ``mean_hamming``, ``n_steps``.
    """
    if not encoded:
        return {
            "unique_count": 0,
            "most_common": "",
            "max_run_length": 0,
            "n_transitions": 0,
            "mean_hamming": 0.0,
            "n_steps": 0,
        }

    # Frequency of each syndrome (deterministic ordering via sorted).
    freq: Dict[str, int] = {}
    for s in encoded:
        freq[s] = freq.get(s, 0) + 1

    # Most common — break ties lexicographically for determinism.
    most_common = sorted(freq.keys(), key=lambda k: (-freq[k], k))[0]

    # Longest consecutive run.
    max_run = 1
    current_run = 1
    for i in range(1, len(encoded)):
        if encoded[i] == encoded[i - 1]:
            current_run += 1
            if current_run > max_run:
                max_run = current_run
        else:
            current_run = 1

    # Mean Hamming distance from transitions.
    if transitions:
        total_hamming = sum(t["hamming_distance"] for t in transitions)
        mean_hamming = total_hamming / len(transitions)
    else:
        mean_hamming = 0.0

    return {
        "unique_count": len(freq),
        "most_common": most_common,
        "max_run_length": max_run,
        "n_transitions": len(transitions),
        "mean_hamming": mean_hamming,
        "n_steps": len(encoded),
    }


# ---------------------------------------------------------------------------
# Step 2 — Classification Rules
# ---------------------------------------------------------------------------


def classify_syndrome_trajectory(patterns: Dict[str, Any]) -> str:
    """Classify syndrome trajectory into a regime type.

    Deterministic rules applied in order:

    * ``"stable"`` — single unique syndrome.
    * ``"oscillatory"`` — exactly 2 unique syndromes with mean Hamming <= 1.
    * ``"chaotic"`` — 3+ unique syndromes with mean Hamming >= 2.
    * ``"boundary"`` — 2+ unique syndromes with mean Hamming == 1.
    * ``"undetermined"`` — fallback.
    """
    unique = patterns["unique_count"]
    mean_h = patterns["mean_hamming"]

    if unique == 0:
        return "undetermined"

    if unique == 1:
        return "stable"

    if unique == 2 and mean_h <= 1.0:
        return "oscillatory"

    if unique >= 3 and mean_h >= 2.0:
        return "chaotic"

    if unique >= 2 and mean_h == 1.0:
        return "boundary"

    return "undetermined"


# ---------------------------------------------------------------------------
# Step 3 — Confidence Score
# ---------------------------------------------------------------------------


def compute_decoder_confidence(patterns: Dict[str, Any]) -> float:
    """Compute decoder confidence as max_run_length / n_steps.

    Returns a value clamped to [0.0, 1.0].
    """
    n_steps = patterns["n_steps"]
    if n_steps == 0:
        return 0.0
    raw = patterns["max_run_length"] / n_steps
    return max(0.0, min(1.0, raw))


# ---------------------------------------------------------------------------
# Step 4 — Full Decoder
# ---------------------------------------------------------------------------


def decode_syndrome_trajectory(
    encoded: List[str],
    transitions: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Decode a syndrome trajectory into regime classification.

    Parameters
    ----------
    encoded:
        Ordered list of syndrome binary strings.
    transitions:
        Transition records from :func:`detect_syndrome_transitions`.

    Returns
    -------
    dict with ``regime_type``, ``confidence``, ``patterns``,
    ``dominant_syndrome``, ``n_transitions``.
    """
    patterns = analyze_syndrome_patterns(encoded, transitions)
    regime_type = classify_syndrome_trajectory(patterns)
    confidence = compute_decoder_confidence(patterns)

    return {
        "regime_type": regime_type,
        "confidence": confidence,
        "patterns": patterns,
        "dominant_syndrome": patterns["most_common"],
        "n_transitions": patterns["n_transitions"],
    }
