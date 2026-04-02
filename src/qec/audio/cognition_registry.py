"""
QEC Cognition Registry — Deterministic Failure-Recall Registry (v136.8.3).

Provides the cognition registry for matching audio fingerprints
to known QEC failure states and recommending policy actions.

Design invariants
-----------------
* frozen dataclasses only
* deterministic matching — same fingerprint always produces identical match
* cosine similarity on spectral feature vectors (centroid, rolloff, peak_bins)
* high-confidence threshold: 0.98
* stable ordering
* no hidden randomness
* no decoder imports
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HIGH_CONFIDENCE_THRESHOLD: float = 0.98
UNKNOWN_STATE: str = "UNKNOWN_STATE"
UNKNOWN_ACTION: str = "NO_ACTION"

REGISTRY_VERSION: str = "v136.8.3"


# ---------------------------------------------------------------------------
# Frozen dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AudioFingerprint:
    """Immutable spectral fingerprint of a QEC audio signature."""

    centroid: float
    rolloff: float
    peak_bins: Tuple[int, ...]
    psd_hash: str


@dataclass(frozen=True)
class CognitionMatch:
    """Immutable result of a cognition registry lookup."""

    confidence: float
    identity: str
    failure_mode: str
    recommended_action: str


@dataclass(frozen=True)
class CognitionRegistryEntry:
    """Immutable registry entry linking QEC state to audio fingerprint."""

    state_hash: str
    code_family: str
    error_type: str
    topology_state: str
    fingerprint: AudioFingerprint
    recommended_action: str
    failure_mode: str


@dataclass(frozen=True)
class CognitionRegistry:
    """Immutable ordered collection of cognition registry entries."""

    entries: Tuple[CognitionRegistryEntry, ...]
    version: str
    registry_hash: str


# ---------------------------------------------------------------------------
# Cosine similarity
# ---------------------------------------------------------------------------


def cosine_similarity(a: NDArray, b: NDArray) -> float:
    """Compute cosine similarity between two vectors.

    Returns value in [-1.0, 1.0]. Deterministic for identical inputs.
    """
    dot = float(np.dot(a, b))
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


# ---------------------------------------------------------------------------
# Fingerprint vector
# ---------------------------------------------------------------------------


def fingerprint_to_vector(fp: AudioFingerprint) -> NDArray:
    """Convert an AudioFingerprint to a numeric feature vector.

    The vector includes centroid, rolloff, and peak bin positions.
    Deterministic: same fingerprint always produces identical vector.
    """
    features = [fp.centroid, fp.rolloff] + list(fp.peak_bins)
    return np.array(features, dtype=np.float64)


# ---------------------------------------------------------------------------
# Registry operations
# ---------------------------------------------------------------------------


def compute_registry_hash(entries: Tuple[CognitionRegistryEntry, ...]) -> str:
    """Compute deterministic SHA-256 hash of registry entries."""
    canonical = []
    for e in entries:
        canonical.append({
            "code_family": e.code_family,
            "error_type": e.error_type,
            "failure_mode": e.failure_mode,
            "fingerprint": {
                "centroid": e.fingerprint.centroid,
                "peak_bins": list(e.fingerprint.peak_bins),
                "psd_hash": e.fingerprint.psd_hash,
                "rolloff": e.fingerprint.rolloff,
            },
            "recommended_action": e.recommended_action,
            "state_hash": e.state_hash,
            "topology_state": e.topology_state,
        })
    payload = json.dumps(
        {"entries": canonical, "version": REGISTRY_VERSION},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_registry(
    entries: Tuple[CognitionRegistryEntry, ...],
) -> CognitionRegistry:
    """Build a new CognitionRegistry from entries.

    Entries are sorted by (code_family, error_type, state_hash)
    for stable ordering. Registry hash is computed deterministically.
    """
    sorted_entries = tuple(sorted(
        entries,
        key=lambda e: (e.code_family, e.error_type, e.state_hash),
    ))
    registry_hash = compute_registry_hash(sorted_entries)
    return CognitionRegistry(
        entries=sorted_entries,
        version=REGISTRY_VERSION,
        registry_hash=registry_hash,
    )


def register_cognition_entry(
    entry: CognitionRegistryEntry,
    registry: CognitionRegistry | None = None,
) -> CognitionRegistry:
    """Register a new cognition entry into a registry.

    Returns a new immutable CognitionRegistry with the entry added.
    Maintains sorted order. Recomputes the registry hash.
    """
    if registry is None:
        existing: Tuple[CognitionRegistryEntry, ...] = ()
    else:
        existing = registry.entries

    new_entries = existing + (entry,)
    return build_registry(new_entries)


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------


def match_registry_signature(
    fingerprint: AudioFingerprint,
    registry: CognitionRegistry,
) -> CognitionMatch:
    """Match an audio fingerprint against registry entries.

    Uses cosine similarity on fingerprint feature vectors.
    Returns the best match if confidence >= HIGH_CONFIDENCE_THRESHOLD,
    otherwise returns UNKNOWN_STATE.

    Deterministic: same fingerprint + registry always produces identical match.
    """
    if not registry.entries:
        return CognitionMatch(
            confidence=0.0,
            identity=UNKNOWN_STATE,
            failure_mode=UNKNOWN_STATE,
            recommended_action=UNKNOWN_ACTION,
        )

    query_vec = fingerprint_to_vector(fingerprint)

    best_confidence = -1.0
    best_entry: CognitionRegistryEntry | None = None

    for entry in registry.entries:
        entry_vec = fingerprint_to_vector(entry.fingerprint)
        sim = cosine_similarity(query_vec, entry_vec)
        if sim > best_confidence:
            best_confidence = sim
            best_entry = entry

    assert best_entry is not None

    # Clamp to [0.0, 1.0] — cosine similarity can be negative for dissimilar vectors
    normalized_confidence = max(0.0, best_confidence)

    if best_confidence >= HIGH_CONFIDENCE_THRESHOLD:
        return CognitionMatch(
            confidence=normalized_confidence,
            identity=f"{best_entry.code_family}:{best_entry.error_type}",
            failure_mode=best_entry.failure_mode,
            recommended_action=best_entry.recommended_action,
        )

    return CognitionMatch(
        confidence=normalized_confidence,
        identity=UNKNOWN_STATE,
        failure_mode=UNKNOWN_STATE,
        recommended_action=UNKNOWN_ACTION,
    )


def recall_similar_failure_state(
    fingerprint: AudioFingerprint,
    registry: CognitionRegistry,
) -> CognitionMatch:
    """Recall the most similar failure state from the registry.

    Alias for match_registry_signature — provided for semantic clarity
    in the cognition cycle API.
    """
    return match_registry_signature(fingerprint, registry)
