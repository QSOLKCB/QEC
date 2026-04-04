"""
Quantization-Aware Decoder Observation — v137.0.1

Read-only diagnostic observation layer that converts continuous decoder
signals into quantized symbolic signatures using the v136.10.0
cross-domain quantization framework.

This is NOT decoder mutation.

This is:
    diagnostic symbolic compression
    + observability enhancement
    + replay-safe metrics

Layer: analysis (Layer 4) — additive read-only diagnostics.
Never imports or mutates decoder internals.

All outputs are deterministic, frozen, and byte-identical on replay.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Optional, Tuple

from qec.analysis.cross_domain_quantization import (
    FLOAT_PRECISION,
    phase_space_quantize,
    risk_band_quantize,
    _float_key,
)


# ---------------------------------------------------------------------------
# Version
# ---------------------------------------------------------------------------

DECODER_OBSERVATION_VERSION: str = "v137.0.1"

# Stability band labels and thresholds
STABILITY_BANDS: Tuple[str, ...] = ("LOW", "MID", "HIGH")
STABILITY_THRESHOLDS: Tuple[float, ...] = (0.4, 0.7)

# Syndrome drift band labels and thresholds
DRIFT_BANDS: Tuple[str, ...] = ("LOW", "MID", "HIGH", "EXTREME")
DRIFT_THRESHOLDS: Tuple[float, ...] = (0.25, 0.5, 0.75)

# Default phase-space bin width
DEFAULT_PHASE_BIN_WIDTH: float = 0.5


# ---------------------------------------------------------------------------
# Frozen dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DecoderObservationSignature:
    """Immutable quantized observation of decoder diagnostic state."""
    symbolic_risk_lattice: str
    syndrome_drift_band: str
    phase_bin_index: Tuple[int, int]
    phase_quantized_coords: Tuple[float, float]
    decoder_quantization_signature: str
    stability_band: str
    decoder_family: Optional[str]
    stable_hash: str


@dataclass(frozen=True)
class ObservationLedger:
    """Immutable ordered ledger of decoder observation signatures."""
    signatures: Tuple[DecoderObservationSignature, ...]
    signature_count: int
    unique_symbol_count: int
    phase_bin_entropy_proxy: float
    symbolic_compression_ratio: float
    stable_hash: str


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _classify_band(
    value: float,
    thresholds: Tuple[float, ...],
    labels: Tuple[str, ...],
) -> str:
    """Classify a value into a band using ordered thresholds."""
    idx = 0
    for i, t in enumerate(thresholds):
        if value >= t:
            idx = i + 1
    return labels[idx]


def _hash_observation(
    symbolic_risk_lattice: str,
    syndrome_drift_band: str,
    phase_bin_index: Tuple[int, int],
    phase_quantized_coords: Tuple[float, float],
    decoder_quantization_signature: str,
    stability_band: str,
    decoder_family: Optional[str],
) -> str:
    """Compute SHA-256 stable hash for a decoder observation signature."""
    payload = json.dumps(
        {
            "decoder_family": decoder_family if decoder_family is not None else "",
            "decoder_quantization_signature": decoder_quantization_signature,
            "phase_bin_index": list(phase_bin_index),
            "phase_quantized_coords": [
                _float_key(phase_quantized_coords[0]),
                _float_key(phase_quantized_coords[1]),
            ],
            "stability_band": stability_band,
            "symbolic_risk_lattice": symbolic_risk_lattice,
            "syndrome_drift_band": syndrome_drift_band,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _hash_ledger(signatures: Tuple[DecoderObservationSignature, ...]) -> str:
    """Compute SHA-256 stable hash over entire observation ledger."""
    hashes = tuple(s.stable_hash for s in signatures)
    payload = json.dumps(hashes, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _build_symbolic_signature(
    risk_label: str,
    drift_band: str,
    phase_bin: Tuple[int, int],
    stability_band: str,
) -> str:
    """Build deterministic compressed symbolic signature string."""
    return (
        f"RISK:{risk_label} | "
        f"DRIFT:{drift_band} | "
        f"PHASE:({phase_bin[0]},{phase_bin[1]}) | "
        f"STAB:{stability_band}"
    )


# ---------------------------------------------------------------------------
# Core observation function
# ---------------------------------------------------------------------------

def observe_decoder_quantization(
    syndrome_drift: float,
    decoder_stability_score: float,
    phase_centroid_q: float,
    phase_centroid_p: float,
    risk_score: float,
    decoder_family: Optional[str] = None,
    phase_bin_width: float = DEFAULT_PHASE_BIN_WIDTH,
) -> DecoderObservationSignature:
    """Convert continuous decoder diagnostic signals into a quantized signature.

    This is a read-only observation — no decoder state is modified.

    Parameters
    ----------
    syndrome_drift : float
        Syndrome drift metric in [0, 1].
    decoder_stability_score : float
        Decoder stability score in [0, 1].
    phase_centroid_q : float
        Phase-space centroid position coordinate.
    phase_centroid_p : float
        Phase-space centroid momentum coordinate.
    risk_score : float
        Risk score in [0, 1].
    decoder_family : str, optional
        Decoder family identifier (e.g. "bp_reference").
    phase_bin_width : float
        Phase-space bin width (default 0.5).

    Returns
    -------
    DecoderObservationSignature
        Frozen quantized observation record.
    """
    if not (0.0 <= syndrome_drift <= 1.0):
        raise ValueError(f"syndrome_drift must be in [0, 1], got {syndrome_drift}")
    if not (0.0 <= decoder_stability_score <= 1.0):
        raise ValueError(
            f"decoder_stability_score must be in [0, 1], got {decoder_stability_score}"
        )
    if not (0.0 <= risk_score <= 1.0):
        raise ValueError(f"risk_score must be in [0, 1], got {risk_score}")
    if phase_bin_width <= 0.0:
        raise ValueError(f"phase_bin_width must be positive, got {phase_bin_width}")

    # Risk band via v136.10.0 framework
    risk_label, _ = risk_band_quantize(risk_score)

    # Phase-space quantization via v136.10.0 framework
    q_quant, p_quant, phase_bin, _ = phase_space_quantize(
        phase_centroid_q, phase_centroid_p, phase_bin_width,
    )

    # Stability band
    stability_band = _classify_band(
        decoder_stability_score, STABILITY_THRESHOLDS, STABILITY_BANDS,
    )

    # Syndrome drift band
    drift_band = _classify_band(
        syndrome_drift, DRIFT_THRESHOLDS, DRIFT_BANDS,
    )

    # Compressed symbolic signature
    sig_str = _build_symbolic_signature(
        risk_label, drift_band, phase_bin, stability_band,
    )

    # Stable hash
    h = _hash_observation(
        symbolic_risk_lattice=risk_label,
        syndrome_drift_band=drift_band,
        phase_bin_index=phase_bin,
        phase_quantized_coords=(q_quant, p_quant),
        decoder_quantization_signature=sig_str,
        stability_band=stability_band,
        decoder_family=decoder_family,
    )

    return DecoderObservationSignature(
        symbolic_risk_lattice=risk_label,
        syndrome_drift_band=drift_band,
        phase_bin_index=phase_bin,
        phase_quantized_coords=(q_quant, p_quant),
        decoder_quantization_signature=sig_str,
        stability_band=stability_band,
        decoder_family=decoder_family,
        stable_hash=h,
    )


# ---------------------------------------------------------------------------
# Ledger construction
# ---------------------------------------------------------------------------

def build_observation_ledger(
    signatures: Tuple[DecoderObservationSignature, ...],
) -> ObservationLedger:
    """Build an immutable observation ledger with compression metrics.

    Parameters
    ----------
    signatures : tuple of DecoderObservationSignature
        Ordered observation records.

    Returns
    -------
    ObservationLedger
        Frozen ledger with observability metrics and stable hash.
    """
    signatures = tuple(signatures)
    n = len(signatures)

    # Unique symbol count — number of distinct symbolic signatures
    unique_sigs = set(s.decoder_quantization_signature for s in signatures)
    unique_symbol_count = len(unique_sigs)

    # Phase bin entropy proxy — count distinct phase bins, normalize
    unique_bins = set(s.phase_bin_index for s in signatures)
    n_unique_bins = len(unique_bins)
    if n > 0 and n_unique_bins > 1:
        # Normalized log ratio as entropy proxy
        phase_bin_entropy_proxy = round(
            math.log(n_unique_bins) / math.log(max(n, 2)), FLOAT_PRECISION,
        )
    else:
        phase_bin_entropy_proxy = 0.0

    # Symbolic compression ratio
    if n > 0:
        symbolic_compression_ratio = round(
            unique_symbol_count / n, FLOAT_PRECISION,
        )
    else:
        symbolic_compression_ratio = 0.0

    h = _hash_ledger(signatures)

    return ObservationLedger(
        signatures=signatures,
        signature_count=n,
        unique_symbol_count=unique_symbol_count,
        phase_bin_entropy_proxy=phase_bin_entropy_proxy,
        symbolic_compression_ratio=symbolic_compression_ratio,
        stable_hash=h,
    )


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export_decoder_observation_bundle(
    ledger: ObservationLedger,
) -> str:
    """Export observation ledger as canonical JSON string.

    Deterministic serialization — sorted keys, minimal separators,
    fixed float precision. Byte-identical on replay.

    Parameters
    ----------
    ledger : ObservationLedger
        Immutable observation ledger.

    Returns
    -------
    str
        Canonical JSON string.
    """
    entries = []
    for s in ledger.signatures:
        entries.append({
            "decoder_family": s.decoder_family if s.decoder_family is not None else "",
            "decoder_quantization_signature": s.decoder_quantization_signature,
            "phase_bin_index": list(s.phase_bin_index),
            "phase_quantized_coords": [
                _float_key(s.phase_quantized_coords[0]),
                _float_key(s.phase_quantized_coords[1]),
            ],
            "stability_band": s.stability_band,
            "stable_hash": s.stable_hash,
            "symbolic_risk_lattice": s.symbolic_risk_lattice,
            "syndrome_drift_band": s.syndrome_drift_band,
        })
    bundle = {
        "compression_metrics": {
            "phase_bin_entropy_proxy": _float_key(ledger.phase_bin_entropy_proxy),
            "symbolic_compression_ratio": _float_key(ledger.symbolic_compression_ratio),
            "unique_symbol_count": ledger.unique_symbol_count,
        },
        "signature_count": ledger.signature_count,
        "signatures": entries,
        "stable_hash": ledger.stable_hash,
        "version": DECODER_OBSERVATION_VERSION,
    }
    return json.dumps(bundle, sort_keys=True, separators=(",", ":"))
