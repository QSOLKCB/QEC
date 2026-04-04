"""
Cross-Domain Quantization Framework — v136.10.0

Unified continuous-to-discrete quantization infrastructure spanning:

    audio signal processing
    AI / model weight quantization
    control-state discretization
    risk-band quantization
    phase-space binning

Core primitive — canonical uniform mid-tread (round-to-nearest) quantization:

    Q(x) = Δ · floor(x / Δ + 1/2)

Zero is always a representable level.
All domains share this single mathematical law.

Layer: analysis (Layer 4) — additive mathematical infrastructure.
Never imports or mutates decoder internals.

All outputs are deterministic, frozen, and byte-identical on replay.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Version
# ---------------------------------------------------------------------------

CROSS_DOMAIN_QUANTIZATION_VERSION: str = "v136.10.0"

# Float precision for deterministic hashing
FLOAT_PRECISION: int = 12


# ---------------------------------------------------------------------------
# Risk-band labels (reuses v136.9.x semantics)
# ---------------------------------------------------------------------------

RISK_BANDS: Tuple[str, ...] = (
    "LOW",
    "WATCH",
    "WARNING",
    "CRITICAL",
    "COLLAPSE_IMMINENT",
)

RISK_THRESHOLDS: Tuple[float, ...] = (0.2, 0.4, 0.6, 0.8)


# ---------------------------------------------------------------------------
# Core quantization primitive
# ---------------------------------------------------------------------------

def uniform_quantize(x: np.ndarray, delta: float) -> np.ndarray:
    """Canonical uniform mid-tread (round-to-nearest) quantization.

    Q(x) = Δ · floor(x / Δ + 1/2)

    Zero is always a representable level.

    Parameters
    ----------
    x : np.ndarray
        Continuous input values.
    delta : float
        Quantization step size. Must be positive.

    Returns
    -------
    np.ndarray
        Quantized values on the Δ-lattice.

    Raises
    ------
    ValueError
        If delta is not positive.
    """
    if delta <= 0.0:
        raise ValueError(f"delta must be positive, got {delta}")
    x = np.asarray(x, dtype=np.float64)
    return delta * np.floor(x / delta + 0.5)


# ---------------------------------------------------------------------------
# Frozen dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class QuantizationDecision:
    """Immutable record of a single quantization operation."""
    domain: str
    parameters: Tuple[Tuple[str, object], ...]
    input_summary: Tuple[Tuple[str, float], ...]
    output_levels: int
    error_estimate: float
    stable_hash: str


@dataclass(frozen=True)
class QuantizationLedger:
    """Immutable ordered ledger of quantization decisions."""
    decisions: Tuple[QuantizationDecision, ...]
    stable_hash: str


# ---------------------------------------------------------------------------
# Hashing helpers
# ---------------------------------------------------------------------------

def _float_key(v: float) -> str:
    """Deterministic float-to-string for hashing."""
    return f"{v:.{FLOAT_PRECISION}e}"


def _canonical_value(v: object) -> str:
    """Deterministic serialization of a parameter value for hashing."""
    if isinstance(v, float):
        return _float_key(v)
    if isinstance(v, int):
        return str(v)
    if isinstance(v, tuple):
        return "[" + ",".join(_canonical_value(e) for e in v) + "]"
    return str(v)


def _hash_decision(
    domain: str,
    parameters: Tuple[Tuple[str, object], ...],
    input_summary: Tuple[Tuple[str, float], ...],
    output_levels: int,
    error_estimate: float,
) -> str:
    """Compute SHA-256 stable hash for a quantization decision."""
    payload = json.dumps(
        {
            "domain": domain,
            "parameters": [[k, _canonical_value(v)] for k, v in parameters],
            "input_summary": [[k, _float_key(v)] for k, v in input_summary],
            "output_levels": output_levels,
            "error_estimate": _float_key(error_estimate),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _hash_ledger(decisions: Tuple[QuantizationDecision, ...]) -> str:
    """Compute SHA-256 stable hash over entire ledger."""
    hashes = tuple(d.stable_hash for d in decisions)
    payload = json.dumps(hashes, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _input_summary(x: np.ndarray) -> Tuple[Tuple[str, float], ...]:
    """Deterministic summary statistics of an input array."""
    x = np.asarray(x, dtype=np.float64)
    return (
        ("count", float(x.size)),
        ("max", float(np.max(x))),
        ("mean", float(np.mean(x))),
        ("min", float(np.min(x))),
    )


# ---------------------------------------------------------------------------
# Audio domain
# ---------------------------------------------------------------------------

def sample_rate_quantize(
    signal: np.ndarray,
    sample_rate_hz: int,
    original_rate_hz: int = 96000,
) -> Tuple[np.ndarray, QuantizationDecision]:
    """Temporal quantization — resample along the time axis.

    Decimates or selects samples at the target rate from a signal
    assumed to be sampled at ``original_rate_hz``.

    Parameters
    ----------
    signal : np.ndarray
        1-D continuous-time signal samples.
    sample_rate_hz : int
        Target sample rate in Hz.
    original_rate_hz : int
        Original sample rate (default 96000 Hz).

    Returns
    -------
    resampled : np.ndarray
        Signal resampled at the target rate.
    decision : QuantizationDecision
        Immutable quantization record.
    """
    signal = np.asarray(signal, dtype=np.float64)
    if sample_rate_hz <= 0:
        raise ValueError(f"sample_rate_hz must be positive, got {sample_rate_hz}")
    if original_rate_hz <= 0:
        raise ValueError(f"original_rate_hz must be positive, got {original_rate_hz}")

    # Deterministic resampling via linear interpolation on uniform grid
    n_original = len(signal)
    # Integer arithmetic to avoid float off-by-one errors
    n_target = max(
        1,
        (n_original * sample_rate_hz + (original_rate_hz // 2))
        // original_rate_hz,
    )
    target_times = np.arange(n_target, dtype=np.float64) / sample_rate_hz
    original_times = np.arange(n_original, dtype=np.float64) / original_rate_hz
    resampled = np.interp(target_times, original_times, signal)

    error = float(np.mean((
        np.interp(original_times, target_times, resampled) - signal
    ) ** 2)) if n_target < n_original else 0.0

    params = (
        ("original_rate_hz", original_rate_hz),
        ("sample_rate_hz", sample_rate_hz),
    )
    summary = _input_summary(signal)
    h = _hash_decision("audio_sample_rate", params, summary, n_target, error)
    decision = QuantizationDecision(
        domain="audio_sample_rate",
        parameters=params,
        input_summary=summary,
        output_levels=n_target,
        error_estimate=error,
        stable_hash=h,
    )
    return resampled, decision


def bit_depth_quantize(
    samples: np.ndarray,
    bit_depth: int,
) -> Tuple[np.ndarray, float, QuantizationDecision]:
    """Amplitude quantization — reduce bit depth.

    Maps continuous samples in [-1, 1] onto a mid-tread lattice with
    step Δ = 2 / 2^n. The actual representable level count is 2^n + 1
    because zero is always included (mid-tread property).

    Parameters
    ----------
    samples : np.ndarray
        Audio samples normalized to [-1, 1].
    bit_depth : int
        Target bit depth (e.g. 24, 16, 8, 4).

    Returns
    -------
    quantized : np.ndarray
        Quantized samples.
    noise_estimate : float
        Mean squared quantization error.
    decision : QuantizationDecision
        Immutable quantization record.
    """
    if bit_depth < 1:
        raise ValueError(f"bit_depth must be >= 1, got {bit_depth}")
    samples = np.asarray(samples, dtype=np.float64)

    n_steps = 2 ** bit_depth
    delta = 2.0 / n_steps  # full range [-1, 1] = width 2
    representable_levels = n_steps + 1  # mid-tread: zero is a level
    quantized = uniform_quantize(samples, delta)
    # Clamp to valid range
    quantized = np.clip(quantized, -1.0, 1.0)

    noise = float(np.mean((quantized - samples) ** 2))

    params = (("bit_depth", bit_depth), ("levels", representable_levels))
    summary = _input_summary(samples)
    h = _hash_decision("audio_bit_depth", params, summary, representable_levels, noise)
    decision = QuantizationDecision(
        domain="audio_bit_depth",
        parameters=params,
        input_summary=summary,
        output_levels=representable_levels,
        error_estimate=noise,
        stable_hash=h,
    )
    return quantized, noise, decision


# ---------------------------------------------------------------------------
# AI / weight quantization domain
# ---------------------------------------------------------------------------

def weight_quantize(
    weights: np.ndarray,
    bits: int,
) -> Tuple[np.ndarray, float, float, QuantizationDecision]:
    """Symmetric uniform weight quantization.

    Maps floating-point weights to a mid-tread symmetric lattice with
    step Δ = 2·w_max / 2^bits. Representable levels = 2^bits + 1
    because zero is always included.

    Parameters
    ----------
    weights : np.ndarray
        Model weights (arbitrary range).
    bits : int
        Target bit width (e.g. 16, 8, 4).

    Returns
    -------
    quantized : np.ndarray
        Quantized weights.
    max_abs_error : float
        Maximum absolute quantization error.
    mse : float
        Mean squared quantization error.
    decision : QuantizationDecision
        Immutable quantization record.
    """
    if bits < 1:
        raise ValueError(f"bits must be >= 1, got {bits}")
    weights = np.asarray(weights, dtype=np.float64)

    w_max = float(np.max(np.abs(weights)))
    n_steps = 2 ** bits
    representable_levels = n_steps + 1  # mid-tread: zero is a level
    if w_max == 0.0:
        quantized = np.zeros_like(weights)
        params = (("bits", bits), ("levels", representable_levels), ("w_max", 0.0))
        summary = _input_summary(weights)
        h = _hash_decision("ai_weight", params, summary, representable_levels, 0.0)
        decision = QuantizationDecision(
            domain="ai_weight",
            parameters=params,
            input_summary=summary,
            output_levels=representable_levels,
            error_estimate=0.0,
            stable_hash=h,
        )
        return quantized, 0.0, 0.0, decision

    delta = (2.0 * w_max) / n_steps
    quantized = uniform_quantize(weights, delta)
    quantized = np.clip(quantized, -w_max, w_max)

    errors = np.abs(quantized - weights)
    max_abs_error = float(np.max(errors))
    mse = float(np.mean(errors ** 2))

    params = (
        ("bits", bits),
        ("levels", representable_levels),
        ("w_max", round(w_max, FLOAT_PRECISION)),
    )
    summary = _input_summary(weights)
    h = _hash_decision("ai_weight", params, summary, representable_levels, mse)
    decision = QuantizationDecision(
        domain="ai_weight",
        parameters=params,
        input_summary=summary,
        output_levels=representable_levels,
        error_estimate=mse,
        stable_hash=h,
    )
    return quantized, max_abs_error, mse, decision


# ---------------------------------------------------------------------------
# Control / risk-band domain
# ---------------------------------------------------------------------------

def risk_band_quantize(score: float) -> Tuple[str, QuantizationDecision]:
    """Map a continuous [0, 1] risk score to a symbolic risk band.

    Thresholds:
        [0.0, 0.2) -> LOW
        [0.2, 0.4) -> WATCH
        [0.4, 0.6) -> WARNING
        [0.6, 0.8) -> CRITICAL
        [0.8, 1.0] -> COLLAPSE_IMMINENT

    Parameters
    ----------
    score : float
        Risk score in [0, 1].

    Returns
    -------
    band : str
        Symbolic risk band label.
    decision : QuantizationDecision
        Immutable quantization record.
    """
    if not (0.0 <= score <= 1.0):
        raise ValueError(f"score must be in [0, 1], got {score}")

    band_index = 0
    for i, t in enumerate(RISK_THRESHOLDS):
        if score >= t:
            band_index = i + 1
    band = RISK_BANDS[band_index]

    params = (("thresholds", RISK_THRESHOLDS),)
    summary = (("score", float(score)),)
    h = _hash_decision(
        "control_risk_band", params, summary, len(RISK_BANDS), 0.0,
    )
    decision = QuantizationDecision(
        domain="control_risk_band",
        parameters=params,
        input_summary=summary,
        output_levels=len(RISK_BANDS),
        error_estimate=0.0,
        stable_hash=h,
    )
    return band, decision


# ---------------------------------------------------------------------------
# Phase-space domain
# ---------------------------------------------------------------------------

def phase_space_quantize(
    q: float,
    p: float,
    bin_width: float,
) -> Tuple[float, float, Tuple[int, int], QuantizationDecision]:
    """Quantize phase-space coordinates onto a uniform grid.

    Parameters
    ----------
    q : float
        Position coordinate.
    p : float
        Momentum coordinate.
    bin_width : float
        Bin width (same for both axes). Must be positive.

    Returns
    -------
    quantized_q : float
        Quantized position.
    quantized_p : float
        Quantized momentum.
    phase_bin_index : tuple of (int, int)
        Integer bin indices (iq, ip).
    decision : QuantizationDecision
        Immutable quantization record.
    """
    if bin_width <= 0.0:
        raise ValueError(f"bin_width must be positive, got {bin_width}")

    q_arr = np.array([q], dtype=np.float64)
    p_arr = np.array([p], dtype=np.float64)
    qQ = float(uniform_quantize(q_arr, bin_width)[0])
    pQ = float(uniform_quantize(p_arr, bin_width)[0])
    iq = int(np.floor(q / bin_width + 0.5))
    ip = int(np.floor(p / bin_width + 0.5))

    error = math.sqrt((qQ - q) ** 2 + (pQ - p) ** 2)

    params = (("bin_width", bin_width),)
    summary = (("p", float(p)), ("q", float(q)))
    h = _hash_decision("phase_space", params, summary, 0, error)
    decision = QuantizationDecision(
        domain="phase_space",
        parameters=params,
        input_summary=summary,
        output_levels=0,
        error_estimate=error,
        stable_hash=h,
    )
    return qQ, pQ, (iq, ip), decision


# ---------------------------------------------------------------------------
# Ledger construction
# ---------------------------------------------------------------------------

def build_ledger(
    decisions: Tuple[QuantizationDecision, ...],
) -> QuantizationLedger:
    """Build an immutable quantization ledger from decisions.

    Parameters
    ----------
    decisions : tuple of QuantizationDecision
        Ordered quantization records.

    Returns
    -------
    QuantizationLedger
        Frozen ledger with stable hash.
    """
    decisions = tuple(decisions)
    h = _hash_ledger(decisions)
    return QuantizationLedger(decisions=decisions, stable_hash=h)


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export_quantization_bundle(
    ledger: QuantizationLedger,
) -> str:
    """Export ledger as canonical JSON string.

    Deterministic serialization — sorted keys, minimal separators,
    fixed float precision. Byte-identical on replay.

    Parameters
    ----------
    ledger : QuantizationLedger
        Immutable quantization ledger.

    Returns
    -------
    str
        Canonical JSON string.
    """
    entries = []
    for d in ledger.decisions:
        entries.append({
            "domain": d.domain,
            "error_estimate": _float_key(d.error_estimate),
            "input_summary": [[k, _float_key(v)] for k, v in d.input_summary],
            "output_levels": d.output_levels,
            "parameters": [[k, _canonical_value(v)] for k, v in d.parameters],
            "stable_hash": d.stable_hash,
        })
    bundle = {
        "decisions": entries,
        "stable_hash": ledger.stable_hash,
        "version": CROSS_DOMAIN_QUANTIZATION_VERSION,
    }
    return json.dumps(bundle, sort_keys=True, separators=(",", ":"))
