"""
v74.6.0 — Invariant Sonification Engine (Math → Sound Layer).

Converts invariant structure, stability metrics, and phase classification
into deterministic audio signals.  Structure-preserving mathematical
sonification — not aesthetic audio.

Mapping design:
- Feature identity  → frequency (pitch)
- Stability score   → amplitude
- Mean drift        → modulation rate
- Invariant strength→ harmonic structure
- Phase class       → envelope shape

Layer 5 — Experiments.
Does not modify decoder internals.  Fully deterministic.  Read-only on inputs.
"""

from __future__ import annotations

import copy
import math
import os
import struct
import wave
from typing import Any, Dict, List, Optional, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Constants — Feature → Frequency Mapping
# ---------------------------------------------------------------------------

_FEATURE_FREQ: Dict[str, float] = {
    "energy": 220.0,
    "centroid": 330.0,
    "spread": 440.0,
    "zcr": 550.0,
}

_FEATURES = ("energy", "centroid", "spread", "zcr")

# ---------------------------------------------------------------------------
# Constants — Phase → Envelope Parameters
# ---------------------------------------------------------------------------

_PHASE_ENVELOPE: Dict[str, str] = {
    "stable_region": "sustained",
    "near_boundary": "pulsing",
    "unstable_region": "decaying",
    "chaotic_transition": "irregular",
}


# ---------------------------------------------------------------------------
# Step 1 — Amplitude from stability score
# ---------------------------------------------------------------------------

def _amplitude_from_stability(stability_score: float) -> float:
    """Map stability score to amplitude.  Stable → louder, unstable → quieter."""
    return 1.0 / (1.0 + stability_score)


# ---------------------------------------------------------------------------
# Step 2 — Modulation rate from drift
# ---------------------------------------------------------------------------

def _modulation_rate(drift: float) -> float:
    """Map mean drift to tremolo modulation rate (Hz).

    Low drift  → 0 Hz (smooth tone).
    High drift → up to ~20 Hz rapid modulation.
    """
    return 20.0 * min(abs(drift), 1.0)


# ---------------------------------------------------------------------------
# Step 3 — Harmonic structure from invariant strength
# ---------------------------------------------------------------------------

def _harmonic_mix(feature: str, invariants: dict) -> List[float]:
    """Return harmonic amplitudes [fundamental, 2nd, 3rd] based on invariant class.

    Strong invariant → pure sine [1, 0, 0].
    Weak invariant   → sine + small harmonic [1, 0.15, 0].
    Non-invariant    → noisy multi-harmonic [1, 0.3, 0.2].
    """
    if feature in invariants.get("strong_invariants", []):
        return [1.0, 0.0, 0.0]
    if feature in invariants.get("weak_invariants", []):
        return [1.0, 0.15, 0.0]
    return [1.0, 0.3, 0.2]


# ---------------------------------------------------------------------------
# Step 4 — Envelope from phase classification
# ---------------------------------------------------------------------------

def _envelope(n_samples: int, phase: str, sr: int = 44100) -> np.ndarray:
    """Generate an amplitude envelope based on phase classification.

    Parameters
    ----------
    n_samples : int
        Number of samples.
    phase : str
        Phase classification string.
    sr : int
        Sample rate.

    Returns
    -------
    np.ndarray
        Envelope array of shape ``(n_samples,)`` in [0, 1].
    """
    t = np.linspace(0.0, n_samples / sr, n_samples, endpoint=False)
    envelope_type = _PHASE_ENVELOPE.get(phase, "sustained")

    if envelope_type == "sustained":
        return np.ones(n_samples)

    if envelope_type == "pulsing":
        # 4 Hz pulsing
        return 0.5 + 0.5 * np.cos(2.0 * np.pi * 4.0 * t)

    if envelope_type == "decaying":
        # Exponential decay with time constant = duration / 3
        duration = n_samples / sr
        tau = max(duration / 3.0, 1e-9)
        return np.exp(-t / tau)

    # irregular — deterministic pseudo-irregular bursts via sum of incommensurate cosines
    return np.clip(
        0.5 + 0.3 * np.cos(2.0 * np.pi * 5.0 * t)
        + 0.2 * np.cos(2.0 * np.pi * 7.7 * t),
        0.0, 1.0,
    )


# ---------------------------------------------------------------------------
# Step 5 — Single-Feature Tone Generator
# ---------------------------------------------------------------------------

def _generate_feature_tone(
    feature: str,
    analysis: dict,
    duration: float,
    sr: int,
) -> np.ndarray:
    """Generate a tone for a single feature based on analysis results.

    Parameters
    ----------
    feature : str
        Feature name (energy, centroid, spread, zcr).
    analysis : dict
        Output from ``run_invariant_analysis``.
    duration : float
        Duration in seconds.
    sr : int
        Sample rate.

    Returns
    -------
    np.ndarray
        Signal array of shape ``(n_samples,)``.
    """
    n_samples = int(duration * sr)
    t = np.linspace(0.0, duration, n_samples, endpoint=False)

    freq = _FEATURE_FREQ[feature]
    stability_score = analysis.get("stability_score", 0.0)
    amplitude = _amplitude_from_stability(stability_score)

    # Mean drift for this feature.
    mean_drift = analysis.get("mean_drift", {})
    drift = abs(mean_drift.get(feature, 0.0))
    mod_rate = _modulation_rate(drift)

    # Harmonic structure.
    invariants = analysis.get("invariants", {})
    harmonics = _harmonic_mix(feature, invariants)

    # Build signal: sum of harmonics.
    signal = np.zeros(n_samples)
    for idx, h_amp in enumerate(harmonics):
        if h_amp > 0.0:
            harmonic_num = idx + 1
            signal += h_amp * np.sin(
                2.0 * np.pi * freq * harmonic_num * t
            )

    # Apply modulation (tremolo).
    if mod_rate > 0.0:
        modulation = 1.0 - 0.5 * (1.0 - np.cos(2.0 * np.pi * mod_rate * t))
        signal *= modulation

    # Apply amplitude.
    signal *= amplitude

    return signal


# ---------------------------------------------------------------------------
# Step 6 — Main Signal Generator
# ---------------------------------------------------------------------------

def generate_invariant_signal(
    analysis: dict,
    duration: float = 2.0,
    sr: int = 44100,
) -> np.ndarray:
    """Generate a combined sonification signal from invariant analysis output.

    Parameters
    ----------
    analysis : dict
        Output from ``run_invariant_analysis``.  Not mutated.
    duration : float
        Duration in seconds (default 2.0).
    sr : int
        Sample rate (default 44100).

    Returns
    -------
    np.ndarray
        Normalized signal array of shape ``(n_samples,)`` in [-1, 1].
    """
    analysis = copy.deepcopy(analysis)
    n_samples = int(duration * sr)

    if n_samples == 0:
        return np.array([], dtype=np.float64)

    # Sum feature tones.
    combined = np.zeros(n_samples)
    for feature in _FEATURES:
        combined += _generate_feature_tone(feature, analysis, duration, sr)

    # Apply phase envelope.
    phase = analysis.get("phase", "stable_region")
    env = _envelope(n_samples, phase, sr)
    combined *= env

    # Normalize to [-1, 1].
    peak = np.max(np.abs(combined))
    if peak > 0.0:
        combined /= peak

    return combined


# ---------------------------------------------------------------------------
# Step 7 — Multi-Frame Sequence Mode
# ---------------------------------------------------------------------------

def generate_sequence_sound(
    analyses: Sequence[dict],
    duration_per_frame: float = 2.0,
    sr: int = 44100,
) -> np.ndarray:
    """Concatenate sonification signals for a sequence of analyses.

    Parameters
    ----------
    analyses : sequence of dict
        List of analysis outputs.  Not mutated.
    duration_per_frame : float
        Duration per frame in seconds.
    sr : int
        Sample rate.

    Returns
    -------
    np.ndarray
        Concatenated signal array.
    """
    if not analyses:
        return np.array([], dtype=np.float64)

    frames = []
    for a in analyses:
        frames.append(generate_invariant_signal(a, duration=duration_per_frame, sr=sr))

    return np.concatenate(frames)


# ---------------------------------------------------------------------------
# Step 8 — WAV Output
# ---------------------------------------------------------------------------

def write_wav(signal: np.ndarray, path: str, sr: int = 44100) -> str:
    """Write a signal array to a 16-bit mono WAV file.

    Parameters
    ----------
    signal : np.ndarray
        Signal in [-1, 1].
    path : str
        Output file path.
    sr : int
        Sample rate.

    Returns
    -------
    str
        Absolute path to the written file.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    # Clip and convert to 16-bit PCM.
    clipped = np.clip(signal, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype(np.int16)

    with wave.open(path, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())

    return os.path.abspath(path)
