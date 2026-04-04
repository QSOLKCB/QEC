"""v137.0.3 — Closed-Loop Auditory + Phase Control.

Maps phase instability into deterministic sonification signatures:
  phase instability → sonification signature → audio quantization
  → symbolic instability trace → governed recovery memory.

Read-only observability + symbolic control support.

Layer 4 — Analysis.
Does not import or modify decoder internals.
Fully deterministic: no randomness, no global state, no input mutation.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Dict, Tuple

# ---------------------------------------------------------------------------
# Version
# ---------------------------------------------------------------------------

AUDITORY_PHASE_VERSION: str = "v137.0.3"

# ---------------------------------------------------------------------------
# Constants — deterministic mappings
# ---------------------------------------------------------------------------

# Risk-score thresholds (upper-exclusive except COLLAPSE).
_RISK_BANDS: Tuple[Tuple[str, float, float], ...] = (
    ("LOW", 0.0, 0.2),
    ("WATCH", 0.2, 0.4),
    ("WARNING", 0.4, 0.6),
    ("CRITICAL", 0.6, 0.8),
    ("COLLAPSE", 0.8, float("inf")),
)

_FREQUENCY_MAP: Dict[str, float] = {
    "LOW": 220.0,
    "WATCH": 440.0,
    "WARNING": 880.0,
    "CRITICAL": 1760.0,
    "COLLAPSE": 3520.0,
}

_BIT_DEPTH_MAP: Dict[str, int] = {
    "LOW": 24,
    "WATCH": 16,
    "WARNING": 12,
    "CRITICAL": 8,
    "COLLAPSE": 4,
}

# ---------------------------------------------------------------------------
# Frozen dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AuditoryPhaseSignature:
    """Immutable sonification signature for a single phase observation."""

    phase_bin_index: Tuple[int, int]
    instability_frequency_hz: float
    amplitude_band: str
    bit_depth_level: int
    audio_symbolic_trace: str
    governed_route_hint: str
    stable_hash: str
    version: str = AUDITORY_PHASE_VERSION


@dataclass(frozen=True)
class AuditoryPhaseLedger:
    """Immutable ledger of auditory phase signatures."""

    signatures: Tuple[AuditoryPhaseSignature, ...]
    signature_count: int
    stable_hash: str


# ---------------------------------------------------------------------------
# Helpers — canonical JSON & hashing
# ---------------------------------------------------------------------------


def _canonical_json(obj: Any) -> str:
    """Produce canonical JSON: sorted keys, compact separators."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True)


def _signature_to_canonical_dict(sig: AuditoryPhaseSignature) -> Dict[str, Any]:
    """Convert signature to a canonical dict for hashing/export."""
    return {
        "amplitude_band": sig.amplitude_band,
        "audio_symbolic_trace": sig.audio_symbolic_trace,
        "bit_depth_level": sig.bit_depth_level,
        "governed_route_hint": sig.governed_route_hint,
        "instability_frequency_hz": sig.instability_frequency_hz,
        "phase_bin_index": list(sig.phase_bin_index),
        "version": sig.version,
    }


def _compute_signature_hash(sig: AuditoryPhaseSignature) -> str:
    """SHA-256 of canonical JSON of an auditory phase signature."""
    payload = _signature_to_canonical_dict(sig)
    canonical = _canonical_json(payload)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _compute_ledger_hash(signatures: Tuple[AuditoryPhaseSignature, ...]) -> str:
    """SHA-256 of ordered signature hashes."""
    hashes = tuple(s.stable_hash for s in signatures)
    canonical = _canonical_json(list(hashes))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Risk classification
# ---------------------------------------------------------------------------


def _classify_risk(risk_score: float) -> str:
    """Map a finite risk score in [0, inf) to a deterministic band label."""
    if not isinstance(risk_score, (int, float)):
        raise TypeError(f"risk_score must be numeric, got {type(risk_score).__name__}")
    if not math.isfinite(risk_score) or risk_score < 0.0:
        raise ValueError(f"risk_score must be finite and >= 0, got {risk_score}")
    for label, lo, hi in _RISK_BANDS:
        if lo <= risk_score < hi:
            return label
    # Unreachable for valid floats, but satisfy exhaustiveness.
    return "COLLAPSE"  # pragma: no cover


# ---------------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------------


def observe_auditory_phase_control(
    phase_bin_index: Tuple[int, int],
    spectral_drift: float,
    risk_score: float,
    governed_route: str,
) -> AuditoryPhaseSignature:
    """Observe phase instability and produce a deterministic sonification signature.

    Parameters
    ----------
    phase_bin_index : tuple[int, int]
        Two-element index identifying the phase bin.
    spectral_drift : float
        Spectral drift magnitude (informational, encoded in trace).
    risk_score : float
        Risk score in [0, inf). Determines frequency, amplitude band,
        and bit-depth via deterministic mapping.
    governed_route : str
        Route hint for governed recovery (e.g. ``"RECOVERY"``).

    Returns
    -------
    AuditoryPhaseSignature
        Frozen, hash-stable sonification signature.

    Raises
    ------
    TypeError
        If ``phase_bin_index`` is not a 2-tuple of ints, or
        ``governed_route`` is not a string.
    ValueError
        If ``risk_score`` is negative, NaN, or infinite.
    """
    # Validate governed_route.
    if not isinstance(governed_route, str):
        raise TypeError(
            f"governed_route must be a str, got {type(governed_route).__name__}"
        )

    # Validate phase_bin_index.
    if (
        not isinstance(phase_bin_index, tuple)
        or len(phase_bin_index) != 2
        or not all(isinstance(v, int) for v in phase_bin_index)
    ):
        raise TypeError(
            "phase_bin_index must be a 2-tuple of ints, "
            f"got {phase_bin_index!r}"
        )
    if phase_bin_index[0] < 0 or phase_bin_index[1] < 0:
        raise ValueError(
            "phase_bin_index elements must be >= 0, "
            f"got {phase_bin_index!r}"
        )

    band = _classify_risk(risk_score)
    frequency = _FREQUENCY_MAP[band]
    bit_depth = _BIT_DEPTH_MAP[band]

    # Symbolic trace: PB(i,j)-D<drift>-F<freq>-B<depth>-<route>
    trace = (
        f"PB({phase_bin_index[0]},{phase_bin_index[1]})"
        f"-D{spectral_drift:.2f}"
        f"-F{int(frequency)}"
        f"-B{bit_depth}"
        f"-{governed_route}"
    )

    # Build signature without hash first, compute hash, then rebuild.
    proto = AuditoryPhaseSignature(
        phase_bin_index=phase_bin_index,
        instability_frequency_hz=frequency,
        amplitude_band=band,
        bit_depth_level=bit_depth,
        audio_symbolic_trace=trace,
        governed_route_hint=governed_route,
        stable_hash="",  # placeholder
    )
    stable_hash = _compute_signature_hash(proto)

    return AuditoryPhaseSignature(
        phase_bin_index=phase_bin_index,
        instability_frequency_hz=frequency,
        amplitude_band=band,
        bit_depth_level=bit_depth,
        audio_symbolic_trace=trace,
        governed_route_hint=governed_route,
        stable_hash=stable_hash,
    )


# ---------------------------------------------------------------------------
# Ledger
# ---------------------------------------------------------------------------


def build_auditory_phase_ledger(
    signatures: Any,
) -> AuditoryPhaseLedger:
    """Build an immutable auditory phase ledger from signatures.

    Parameters
    ----------
    signatures : iterable of AuditoryPhaseSignature
        Signatures to collect.  Normalised to a tuple internally.

    Returns
    -------
    AuditoryPhaseLedger
    """
    signatures = tuple(signatures)
    for i, s in enumerate(signatures):
        if not isinstance(s, AuditoryPhaseSignature):
            raise TypeError(
                f"signatures[{i}] must be AuditoryPhaseSignature, "
                f"got {type(s).__name__}"
            )
    ledger_hash = _compute_ledger_hash(signatures)
    return AuditoryPhaseLedger(
        signatures=signatures,
        signature_count=len(signatures),
        stable_hash=ledger_hash,
    )


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def export_auditory_phase_bundle(
    signature: AuditoryPhaseSignature,
) -> Dict[str, Any]:
    """Export a single signature as a canonical JSON-safe dict.

    Deterministic: same signature always produces byte-identical export.
    Shape is canonical_dict + layer + stable_hash, allowing hash
    recomputation directly from the exported artifact.
    """
    base = _signature_to_canonical_dict(signature)
    base["layer"] = "closed_loop_auditory_phase_control"
    base["stable_hash"] = signature.stable_hash
    return base


def export_auditory_phase_ledger(
    ledger: AuditoryPhaseLedger,
) -> Dict[str, Any]:
    """Export a ledger as a canonical JSON-safe dict.

    Deterministic: same ledger always produces byte-identical export.
    """
    return {
        "layer": "closed_loop_auditory_phase_control",
        "signature_count": ledger.signature_count,
        "signatures": [
            export_auditory_phase_bundle(s) for s in ledger.signatures
        ],
        "stable_hash": ledger.stable_hash,
        "version": AUDITORY_PHASE_VERSION,
    }
