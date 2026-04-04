"""
Decoder Topology Discovery — v137.0.2

Read-only topology discovery layer that produces deterministic decoder
topology recommendations.

Scoring inputs:
    1. optional decoder_family hint (exact-match signal, weight 0.5)
    2. symbolic observation signature overlap (Jaccard bigram, weight 0.3)
    3. phase-bin proximity to canonical reference points (Manhattan, weight 0.2)
    4. deterministic tie-breaking (alphabetical family name)

Identity / export fields (carried but not scored):
    - symbolic_risk_lattice — preserved in decision identity and export bundle
    - known_portfolio — restricts candidate set when provided

Compatible with code zoo families and decoder portfolio registries
but does not import or depend on them directly.

Layer: analysis (Layer 4) — additive read-only diagnostics.
Never imports or mutates decoder internals.

All outputs are deterministic, frozen, and byte-identical on replay.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Version
# ---------------------------------------------------------------------------

TOPOLOGY_DISCOVERY_VERSION: str = "v137.0.2"

# Known decoder families (canonical sorted order)
KNOWN_DECODER_FAMILIES: Tuple[str, ...] = (
    "qldpc",
    "repetition",
    "surface",
    "toric",
)

# Topology similarity classes (ordered by quality)
SIMILARITY_CLASSES: Tuple[str, ...] = (
    "EXACT",
    "STRONG",
    "MODERATE",
    "WEAK",
    "NONE",
)

# Thresholds for similarity class assignment
SIMILARITY_THRESHOLDS: Tuple[float, ...] = (0.9, 0.7, 0.4, 0.15)

# Default recovery topology suggestions per family
_RECOVERY_SUGGESTIONS: dict[str, Tuple[str, ...]] = {
    "qldpc": ("qldpc_bp", "qldpc_osd", "qldpc_uf"),
    "repetition": ("repetition_majority", "repetition_ml"),
    "surface": ("surface_mwpm", "surface_uf", "surface_bp"),
    "toric": ("toric_mwpm", "toric_uf", "toric_bp"),
}

_DEFAULT_RECOVERY: Tuple[str, ...] = ("generic_bp", "generic_uf")

# Canonical reference phase bins per family (deterministic, module-level)
REFERENCE_PHASE_BINS: dict[str, Tuple[int, int]] = {
    "qldpc": (2, 3),
    "repetition": (0, 0),
    "surface": (1, 1),
    "toric": (1, 2),
}


# ---------------------------------------------------------------------------
# Frozen dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TopologyDiscoveryDecision:
    """Immutable result of a single topology discovery evaluation."""

    recommended_decoder_family: str
    topology_pairing_score: float
    recovery_topology_suggestions: Tuple[str, ...]
    similarity_class: str
    observation_signature: str
    symbolic_risk_lattice: str
    phase_bin_index: Tuple[int, int]
    stable_hash: str


@dataclass(frozen=True)
class TopologyDiscoveryLedger:
    """Immutable ledger of topology discovery decisions."""

    decisions: Tuple[TopologyDiscoveryDecision, ...]
    ledger_version: str
    stable_hash: str


# ---------------------------------------------------------------------------
# Deterministic helpers
# ---------------------------------------------------------------------------


def _canonical_json(obj: object) -> str:
    """Produce canonical JSON: sorted keys, compact separators."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256(payload: str) -> str:
    """Deterministic SHA-256 hex digest."""
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _bigrams(s: str) -> tuple[str, ...]:
    """Extract sorted unique character bigrams from a string."""
    if len(s) < 2:
        return (s,) if s else ()
    bg = sorted(set(s[i:i + 2] for i in range(len(s) - 1)))
    return tuple(bg)


def _jaccard_bigram(a: str, b: str) -> float:
    """Deterministic Jaccard similarity on character bigrams."""
    ba = set(_bigrams(a))
    bb = set(_bigrams(b))
    if not ba and not bb:
        return 1.0
    if not ba or not bb:
        return 0.0
    intersection = len(ba & bb)
    union = len(ba | bb)
    return intersection / union


def _phase_proximity(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    """Phase-bin proximity score: 1.0 for exact match, decays with Manhattan distance.

    Score = 1.0 / (1.0 + manhattan_distance).
    """
    dist = abs(a[0] - b[0]) + abs(a[1] - b[1])
    return 1.0 / (1.0 + dist)


def _classify_similarity(score: float) -> str:
    """Map a pairing score to a similarity class using fixed thresholds."""
    for i, threshold in enumerate(SIMILARITY_THRESHOLDS):
        if score >= threshold:
            return SIMILARITY_CLASSES[i]
    return SIMILARITY_CLASSES[-1]


def _compute_decision_hash(
    recommended_decoder_family: str,
    topology_pairing_score: float,
    recovery_topology_suggestions: Tuple[str, ...],
    similarity_class: str,
    observation_signature: str,
    symbolic_risk_lattice: str,
    phase_bin_index: Tuple[int, int],
) -> str:
    """Compute stable SHA-256 hash of a decision's content."""
    canonical = {
        "observation_signature": observation_signature,
        "phase_bin_index": list(phase_bin_index),
        "recommended_decoder_family": recommended_decoder_family,
        "recovery_topology_suggestions": list(recovery_topology_suggestions),
        "similarity_class": similarity_class,
        "symbolic_risk_lattice": symbolic_risk_lattice,
        "topology_pairing_score": f"{topology_pairing_score:.12e}",
        "version": TOPOLOGY_DISCOVERY_VERSION,
    }
    return _sha256(_canonical_json(canonical))


# ---------------------------------------------------------------------------
# Core topology discovery
# ---------------------------------------------------------------------------


def discover_decoder_topology(
    observation_signature: str,
    symbolic_risk_lattice: str,
    phase_bin_index: Tuple[int, int],
    decoder_family: Optional[str] = None,
    known_portfolio: Optional[Tuple[str, ...]] = None,
) -> TopologyDiscoveryDecision:
    """Discover the best decoder topology match for given observation state.

    Scoring combines three deterministic signals:
        1. Family match weight (0.5) — exact match on decoder_family
        2. Symbolic overlap weight (0.3) — Jaccard bigram similarity
           between observation_signature and candidate family name
        3. Phase proximity weight (0.2) — Manhattan proximity of
           phase_bin_index to a canonical reference point per family

    Parameters
    ----------
    observation_signature : str
        Symbolic observation signature string.
    symbolic_risk_lattice : str
        Quantized risk lattice label (e.g. "LOW", "CRITICAL").
    phase_bin_index : tuple of (int, int)
        Quantized phase-space bin coordinates.
    decoder_family : str or None
        Optional known decoder family hint.
    known_portfolio : tuple of str or None
        Optional tuple of known decoder family names to consider.

    Returns
    -------
    TopologyDiscoveryDecision
        Frozen, deterministic recommendation.
    """
    # Validate phase_bin_index at API boundary
    if (
        not isinstance(phase_bin_index, tuple)
        or len(phase_bin_index) != 2
        or not isinstance(phase_bin_index[0], int)
        or not isinstance(phase_bin_index[1], int)
    ):
        raise ValueError(
            f"phase_bin_index must be a tuple of two ints, got {phase_bin_index!r}"
        )
    if phase_bin_index[0] < 0 or phase_bin_index[1] < 0:
        raise ValueError(
            f"phase_bin_index entries must be non-negative, got {phase_bin_index!r}"
        )

    # Determine candidate families (sorted for determinism)
    if known_portfolio is not None:
        candidates = tuple(sorted(set(known_portfolio)))
    else:
        candidates = KNOWN_DECODER_FAMILIES

    if not candidates:
        candidates = KNOWN_DECODER_FAMILIES

    # Score each candidate
    scored: list[tuple[float, str]] = []
    for family in candidates:
        # 1. Family match (exact)
        family_score = 1.0 if (decoder_family is not None and family == decoder_family) else 0.0

        # 2. Symbolic overlap
        symbolic_score = _jaccard_bigram(observation_signature, family)

        # 3. Phase proximity
        ref_bin = REFERENCE_PHASE_BINS.get(family, (0, 0))
        phase_score = _phase_proximity(phase_bin_index, ref_bin)

        # Weighted combination
        total = 0.5 * family_score + 0.3 * symbolic_score + 0.2 * phase_score
        scored.append((total, family))

    # Deterministic tie-breaking: highest score first, then alphabetical family
    scored.sort(key=lambda x: (-x[0], x[1]))

    best_score, best_family = scored[0]

    # Recovery suggestions
    recovery = _RECOVERY_SUGGESTIONS.get(best_family, _DEFAULT_RECOVERY)

    # Similarity class
    similarity_class = _classify_similarity(best_score)

    # Stable hash
    stable_hash = _compute_decision_hash(
        recommended_decoder_family=best_family,
        topology_pairing_score=best_score,
        recovery_topology_suggestions=recovery,
        similarity_class=similarity_class,
        observation_signature=observation_signature,
        symbolic_risk_lattice=symbolic_risk_lattice,
        phase_bin_index=phase_bin_index,
    )

    return TopologyDiscoveryDecision(
        recommended_decoder_family=best_family,
        topology_pairing_score=best_score,
        recovery_topology_suggestions=recovery,
        similarity_class=similarity_class,
        observation_signature=observation_signature,
        symbolic_risk_lattice=symbolic_risk_lattice,
        phase_bin_index=phase_bin_index,
        stable_hash=stable_hash,
    )


# ---------------------------------------------------------------------------
# Ledger construction
# ---------------------------------------------------------------------------


def build_topology_ledger(
    decisions: Sequence[TopologyDiscoveryDecision],
) -> TopologyDiscoveryLedger:
    """Build an immutable ledger from a sequence of decisions.

    Accepts any sequence (tuple, list, etc.); normalizes to tuple internally.
    Decisions are stored in the provided order (caller controls ordering).
    The ledger hash covers all decisions deterministically.

    Parameters
    ----------
    decisions : sequence of TopologyDiscoveryDecision
        Ordered decisions to include.

    Returns
    -------
    TopologyDiscoveryLedger
        Frozen, deterministic ledger.
    """
    # Normalize to tuple to prevent mutable-sequence aliasing
    decisions = tuple(decisions)
    for i, d in enumerate(decisions):
        if not isinstance(d, TopologyDiscoveryDecision):
            raise TypeError(
                f"decisions[{i}] must be TopologyDiscoveryDecision, "
                f"got {type(d).__name__}"
            )

    canonical = {
        "decisions": [
            {
                "observation_signature": d.observation_signature,
                "phase_bin_index": list(d.phase_bin_index),
                "recommended_decoder_family": d.recommended_decoder_family,
                "recovery_topology_suggestions": list(d.recovery_topology_suggestions),
                "similarity_class": d.similarity_class,
                "symbolic_risk_lattice": d.symbolic_risk_lattice,
                "topology_pairing_score": f"{d.topology_pairing_score:.12e}",
            }
            for d in decisions
        ],
        "version": TOPOLOGY_DISCOVERY_VERSION,
    }
    ledger_hash = _sha256(_canonical_json(canonical))

    return TopologyDiscoveryLedger(
        decisions=decisions,
        ledger_version=TOPOLOGY_DISCOVERY_VERSION,
        stable_hash=ledger_hash,
    )


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def export_topology_bundle(
    ledger: TopologyDiscoveryLedger,
) -> str:
    """Export a topology ledger as canonical JSON.

    Returns byte-identical output for identical input.

    Parameters
    ----------
    ledger : TopologyDiscoveryLedger
        The ledger to export.

    Returns
    -------
    str
        Canonical JSON string.
    """
    bundle = {
        "decisions": [
            {
                "observation_signature": d.observation_signature,
                "phase_bin_index": list(d.phase_bin_index),
                "recommended_decoder_family": d.recommended_decoder_family,
                "recovery_topology_suggestions": list(d.recovery_topology_suggestions),
                "similarity_class": d.similarity_class,
                "stable_hash": d.stable_hash,
                "symbolic_risk_lattice": d.symbolic_risk_lattice,
                "topology_pairing_score": f"{d.topology_pairing_score:.12e}",
            }
            for d in ledger.decisions
        ],
        "ledger_hash": ledger.stable_hash,
        "version": ledger.ledger_version,
    }
    return _canonical_json(bundle)
