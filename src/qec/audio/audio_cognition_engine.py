"""
QEC Audio Cognition Engine — Closed-Loop Deterministic Observability (v136.8.3).

Closed-loop law:

    QEC State -> Render -> Listen -> Fingerprint -> Registry Match
    -> Recall -> Policy Action Hook

Integrates with:
- v136.8.2 code zoo (src/qec/codes/code_zoo.py)
- triality signal engine
- cognition registry

Design invariants
-----------------
* frozen dataclasses only
* deterministic — same state always produces identical cognition cycle
* no hidden randomness
* no decoder imports
* stdlib + numpy + scipy only
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Tuple

import numpy as np
from numpy.typing import NDArray

from qec.audio.cognition_registry import (
    AudioFingerprint,
    CognitionMatch,
    CognitionRegistry,
    CognitionRegistryEntry,
    HIGH_CONFIDENCE_THRESHOLD,
    UNKNOWN_ACTION,
    UNKNOWN_STATE,
    match_registry_signature,
    recall_similar_failure_state,
    register_cognition_entry,
)
from qec.audio.triality_signal_engine import (
    NUM_SAMPLES,
    SAMPLE_RATE,
    TrialityParams,
    compute_psd,
    compute_psd_hash,
    compute_spectral_centroid,
    compute_spectral_rolloff,
    compute_peak_bins,
    derive_triality_params,
    synthesize_triality_waveform,
)


# ---------------------------------------------------------------------------
# Engine version
# ---------------------------------------------------------------------------

ENGINE_VERSION: str = "v136.8.3"


# ---------------------------------------------------------------------------
# Frozen dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CognitionCycleResult:
    """Immutable result of a full audio cognition cycle."""

    params: TrialityParams
    fingerprint: AudioFingerprint
    match: CognitionMatch
    engine_version: str


# ---------------------------------------------------------------------------
# Code zoo integration
# ---------------------------------------------------------------------------


def get_code_zoo_families() -> Tuple[str, ...]:
    """Return registered code families from the code zoo.

    Integrates with src/qec/codes/code_zoo.py (v136.8.2).
    Deterministic: always returns families in sorted order.
    """
    from qec.codes.code_zoo import build_default_code_zoo

    zoo = build_default_code_zoo()
    families = sorted(set(spec.family for spec in zoo.codes))
    return tuple(families)


def derive_carrier_freq_from_zoo(code_family: str) -> float | None:
    """Derive carrier frequency for a code family using the code zoo.

    Returns None if the code family is not in the zoo.
    Deterministic: same code family always maps to the same frequency.
    """
    from qec.audio.triality_signal_engine import _hash_to_float, CARRIER_FREQ_MIN, CARRIER_FREQ_MAX

    families = get_code_zoo_families()
    if code_family not in families:
        return None
    t = _hash_to_float(code_family, salt="carrier")
    return CARRIER_FREQ_MIN + t * (CARRIER_FREQ_MAX - CARRIER_FREQ_MIN)


# ---------------------------------------------------------------------------
# Core engine functions
# ---------------------------------------------------------------------------


def render_qec_audio_signature(
    code_family: str,
    error_type: str,
    topology_state: str,
    state_hash: str,
) -> NDArray:
    """Render a deterministic audio waveform from QEC state.

    Applies the triality law:
        Signal = Carrier(code_family) + Modulation(error_type) + Overlay(topology_state)

    Returns a 1-D float64 numpy array.
    Deterministic: same inputs always produce byte-identical output.
    """
    params = derive_triality_params(code_family, error_type, topology_state, state_hash)
    return synthesize_triality_waveform(params)


def compute_spectral_fingerprint(audio_buffer: NDArray) -> AudioFingerprint:
    """Compute a deterministic spectral fingerprint from an audio buffer.

    Extracts: spectral centroid, spectral rolloff, peak frequency bins, PSD hash.
    Deterministic: same buffer always produces identical fingerprint.
    """
    psd = compute_psd(audio_buffer)
    centroid = compute_spectral_centroid(psd)
    rolloff = compute_spectral_rolloff(psd)
    peaks = compute_peak_bins(psd)
    psd_h = compute_psd_hash(psd)

    return AudioFingerprint(
        centroid=centroid,
        rolloff=rolloff,
        peak_bins=peaks,
        psd_hash=psd_h,
    )


def run_cognition_cycle(
    code_family: str,
    error_type: str,
    topology_state: str,
    state_hash: str,
    registry: CognitionRegistry,
) -> CognitionCycleResult:
    """Run a full audio cognition cycle.

    Closed-loop:
        QEC State -> Render -> Fingerprint -> Registry Match -> Result

    Deterministic: same inputs + registry always produce identical result.
    """
    # Step 1: Derive parameters
    params = derive_triality_params(code_family, error_type, topology_state, state_hash)

    # Step 2: Render audio
    waveform = synthesize_triality_waveform(params)

    # Step 3: Fingerprint
    fingerprint = compute_spectral_fingerprint(waveform)

    # Step 4: Registry match
    match = match_registry_signature(fingerprint, registry)

    return CognitionCycleResult(
        params=params,
        fingerprint=fingerprint,
        match=match,
        engine_version=ENGINE_VERSION,
    )


# ---------------------------------------------------------------------------
# Export / serialization
# ---------------------------------------------------------------------------


def export_cognition_bundle(result: CognitionCycleResult) -> Dict[str, Any]:
    """Export a CognitionCycleResult as a canonical JSON-serializable dict.

    Deterministic: same result always produces identical export.
    """
    return {
        "engine_version": result.engine_version,
        "fingerprint": {
            "centroid": result.fingerprint.centroid,
            "peak_bins": list(result.fingerprint.peak_bins),
            "psd_hash": result.fingerprint.psd_hash,
            "rolloff": result.fingerprint.rolloff,
        },
        "match": {
            "confidence": result.match.confidence,
            "failure_mode": result.match.failure_mode,
            "identity": result.match.identity,
            "recommended_action": result.match.recommended_action,
        },
        "params": {
            "carrier_freq": result.params.carrier_freq,
            "mod_depth": result.params.mod_depth,
            "mod_freq": result.params.mod_freq,
            "overlay_base_freq": result.params.overlay_base_freq,
            "overlay_harmonics": result.params.overlay_harmonics,
            "state_hash": result.params.state_hash,
        },
    }


def export_cognition_bundle_json(result: CognitionCycleResult) -> str:
    """Export a CognitionCycleResult as canonical JSON string.

    Deterministic: same result always produces byte-identical JSON.
    """
    bundle = export_cognition_bundle(result)
    return json.dumps(bundle, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
