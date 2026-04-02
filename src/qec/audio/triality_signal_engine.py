"""
QEC Triality Signal Engine — Deterministic Audio Signature Rendering (v136.8.3).

Triality Law
------------
Audio Signature =
    Carrier(Code Family)
  + Modulation(Error Type)
  + Overlay(Topology State)

Design invariants
-----------------
* frozen dataclasses only
* deterministic — same input always produces identical waveform
* no hidden randomness
* no new dependencies (numpy + scipy + hashlib only)
* decoder untouched
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np
from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SAMPLE_RATE: int = 8000
DURATION: float = 0.25  # seconds
NUM_SAMPLES: int = int(SAMPLE_RATE * DURATION)

# Carrier frequency band: [200, 2000] Hz
CARRIER_FREQ_MIN: float = 200.0
CARRIER_FREQ_MAX: float = 2000.0

# Modulation frequency band: [1, 20] Hz
MOD_FREQ_MIN: float = 1.0
MOD_FREQ_MAX: float = 20.0

# Overlay harmonic count range
OVERLAY_HARMONICS_MIN: int = 1
OVERLAY_HARMONICS_MAX: int = 8


# ---------------------------------------------------------------------------
# Frozen dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrialityParams:
    """Immutable triality signal parameters derived from QEC state."""

    carrier_freq: float
    mod_freq: float
    mod_depth: float
    overlay_harmonics: int
    overlay_base_freq: float
    state_hash: str


# ---------------------------------------------------------------------------
# Deterministic hash-to-float mapping
# ---------------------------------------------------------------------------


def _hash_to_float(data: str, salt: str = "") -> float:
    """Map a string deterministically to [0.0, 1.0) via SHA-256."""
    digest = hashlib.sha256(f"{salt}:{data}".encode("utf-8")).hexdigest()
    # Use first 8 hex chars (32 bits) for uniform float in [0.0, 1.0)
    return int(digest[:8], 16) / (2**32)


def _hash_to_int(data: str, low: int, high: int, salt: str = "") -> int:
    """Map a string deterministically to [low, high] integer range."""
    t = _hash_to_float(data, salt)
    return low + int(t * (high - low + 1)) % (high - low + 1)


# ---------------------------------------------------------------------------
# Triality parameter derivation
# ---------------------------------------------------------------------------


def derive_triality_params(
    code_family: str,
    error_type: str,
    topology_state: str,
    state_hash: str,
) -> TrialityParams:
    """Derive deterministic triality signal parameters from QEC state.

    Carrier frequency is derived from code_family.
    Modulation frequency and depth from error_type.
    Overlay harmonics from topology_state.

    Deterministic: same inputs always produce identical parameters.
    """
    # Carrier from code family
    carrier_t = _hash_to_float(code_family, salt="carrier")
    carrier_freq = CARRIER_FREQ_MIN + carrier_t * (CARRIER_FREQ_MAX - CARRIER_FREQ_MIN)

    # Modulation from error type
    mod_t = _hash_to_float(error_type, salt="mod_freq")
    mod_freq = MOD_FREQ_MIN + mod_t * (MOD_FREQ_MAX - MOD_FREQ_MIN)
    mod_depth = _hash_to_float(error_type, salt="mod_depth") * 0.8 + 0.1  # [0.1, 0.9]

    # Overlay from topology state
    overlay_harmonics = _hash_to_int(
        topology_state, OVERLAY_HARMONICS_MIN, OVERLAY_HARMONICS_MAX, salt="harmonics"
    )
    overlay_base_t = _hash_to_float(topology_state, salt="overlay_base")
    overlay_base_freq = 50.0 + overlay_base_t * 150.0  # [50, 200] Hz

    return TrialityParams(
        carrier_freq=carrier_freq,
        mod_freq=mod_freq,
        mod_depth=mod_depth,
        overlay_harmonics=overlay_harmonics,
        overlay_base_freq=overlay_base_freq,
        state_hash=state_hash,
    )


# ---------------------------------------------------------------------------
# Waveform synthesis
# ---------------------------------------------------------------------------


def _generate_carrier(freq: float, n_samples: int, sample_rate: int) -> NDArray:
    """Generate a pure sine carrier tone."""
    t = np.arange(n_samples, dtype=np.float64) / sample_rate
    return np.sin(2.0 * math.pi * freq * t)


def _apply_am_modulation(
    carrier: NDArray, mod_freq: float, mod_depth: float, sample_rate: int,
) -> NDArray:
    """Apply amplitude modulation to carrier."""
    n_samples = len(carrier)
    t = np.arange(n_samples, dtype=np.float64) / sample_rate
    envelope = 1.0 - mod_depth * (0.5 + 0.5 * np.sin(2.0 * math.pi * mod_freq * t))
    return carrier * envelope


def _generate_overlay(
    base_freq: float, n_harmonics: int, n_samples: int, sample_rate: int,
) -> NDArray:
    """Generate additive harmonic overlay."""
    t = np.arange(n_samples, dtype=np.float64) / sample_rate
    overlay = np.zeros(n_samples, dtype=np.float64)
    for h in range(1, n_harmonics + 1):
        amplitude = 0.3 / h  # harmonic rolloff
        overlay += amplitude * np.sin(2.0 * math.pi * base_freq * h * t)
    return overlay


def synthesize_triality_waveform(params: TrialityParams) -> NDArray:
    """Synthesize a deterministic triality waveform from parameters.

    Triality Law:
        Signal = Carrier(code_family) + Modulation(error_type) + Overlay(topology)

    Returns a 1-D float64 numpy array of length NUM_SAMPLES.
    """
    # Layer 1: Carrier
    carrier = _generate_carrier(params.carrier_freq, NUM_SAMPLES, SAMPLE_RATE)

    # Layer 2: AM modulation
    modulated = _apply_am_modulation(
        carrier, params.mod_freq, params.mod_depth, SAMPLE_RATE,
    )

    # Layer 3: Harmonic overlay
    overlay = _generate_overlay(
        params.overlay_base_freq, params.overlay_harmonics, NUM_SAMPLES, SAMPLE_RATE,
    )

    # Combine: modulated carrier + overlay
    signal = modulated + overlay

    # Normalize to [-1, 1]
    peak = np.max(np.abs(signal))
    if peak > 0.0:
        signal = signal / peak

    return signal


# ---------------------------------------------------------------------------
# PSD computation
# ---------------------------------------------------------------------------


def compute_psd(signal: NDArray) -> NDArray:
    """Compute the one-sided power spectral density via real FFT.

    Returns a 1-D float64 array (PSD magnitudes).
    Deterministic: same signal always produces identical PSD.
    """
    n = len(signal)
    fft_vals = np.fft.rfft(signal)
    psd = (np.abs(fft_vals) ** 2) / n
    # Double all except DC and Nyquist
    psd[1:-1] *= 2.0
    return psd


def compute_psd_hash(psd: NDArray) -> str:
    """Compute a deterministic SHA-256 hash of a PSD array.

    Uses canonical byte representation for reproducibility.
    """
    # Use tobytes on float64 for exact binary identity
    return hashlib.sha256(psd.tobytes()).hexdigest()


# ---------------------------------------------------------------------------
# Spectral features
# ---------------------------------------------------------------------------


def compute_spectral_centroid(psd: NDArray) -> float:
    """Compute the spectral centroid from PSD."""
    freqs = np.arange(len(psd), dtype=np.float64)
    total_power = np.sum(psd)
    if total_power == 0.0:
        return 0.0
    return float(np.sum(freqs * psd) / total_power)


def compute_spectral_rolloff(psd: NDArray, threshold: float = 0.85) -> float:
    """Compute the spectral rolloff frequency bin."""
    total_power = np.sum(psd)
    if total_power == 0.0:
        return 0.0
    cumsum = np.cumsum(psd)
    rolloff_idx = np.searchsorted(cumsum, threshold * total_power)
    return float(rolloff_idx)


def compute_peak_bins(psd: NDArray, n_peaks: int = 5) -> Tuple[int, ...]:
    """Return the top-n PSD peak bin indices, sorted ascending."""
    n_peaks = min(n_peaks, len(psd))
    indices = np.argsort(psd, kind="stable")[-n_peaks:]
    return tuple(sorted(int(i) for i in indices))
