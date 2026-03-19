"""Multidimensional deterministic sonification engine (v72.2.0).

Extends the E8 baseline into stereo multi-channel output with:
    - Qutrit (base-3) frequency mapping
    - Ququart (base-4) triangle-wave mapping
    - Complexity-driven energy distribution across channels
    - Invariant-preserving silence gates applied post-mix

Output: stereo int16 numpy array of shape (n_samples, 2).

Invariants:
    - Deterministic: same input -> identical bytes
    - No mutation of input dict
    - No NaN / inf in output
    - Bounded: abs(sample) <= 32767
    - Silence gates applied after full signal composition
"""

from __future__ import annotations

import copy
import math
import struct
import wave
from typing import List, Optional, Tuple

import numpy as np

SAMPLE_RATE: int = 44100
DURATION: float = 5.0
BASE_FREQ: float = 110.0
CUSTOM_PI: float = (1.0 + math.sqrt(5.0)) / 4.0
PHI: float = (1.0 + math.sqrt(5.0)) / 2.0

E8_ROOT_RATIOS: List[float] = [
    1.0, 1.125, 1.2, 1.333, 1.5, 1.6, 1.875, 2.0,
    2.25, 2.4, 2.666, 3.0, 3.2, 3.75, 4.0, 4.5,
]


def _validate_input(result: dict) -> None:
    """Validate the input dict strictly. Raises ValueError on bad input."""
    if not isinstance(result, dict):
        raise ValueError("result must be a dict")

    required_keys = {"columns", "errorRate", "complexity", "invariants"}
    missing = required_keys - set(result.keys())
    if missing:
        raise ValueError(f"Missing required keys: {missing}")

    columns = result["columns"]
    if not isinstance(columns, list) or not all(isinstance(c, int) for c in columns):
        raise ValueError("columns must be a list of int")

    error_rate = result["errorRate"]
    if not isinstance(error_rate, (int, float)):
        raise ValueError("errorRate must be numeric")

    complexity = result["complexity"]
    if not isinstance(complexity, (int, float)):
        raise ValueError("complexity must be numeric")

    invariants = result["invariants"]
    if not isinstance(invariants, list):
        raise ValueError("invariants must be a list")
    for item in invariants:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError("Each invariant must be a (start, end) pair")
        if not all(isinstance(v, (int, float)) for v in item):
            raise ValueError("Invariant bounds must be numeric")


def _triangle_wave(phase: np.ndarray) -> np.ndarray:
    """Deterministic triangle wave from phase array (0-based, period 2*pi)."""
    # Normalize phase to [0, 1) period
    p = (phase / (2.0 * np.pi)) % 1.0
    # Triangle: rises 0->1 over first half, falls 1->-1 over second half
    return np.where(p < 0.5, 4.0 * p - 1.0, 3.0 - 4.0 * p)


def sonify_e8_multidim(
    result: dict,
    output_path: Optional[str] = None,
) -> np.ndarray:
    """Generate a multidimensional deterministic sonification.

    Produces stereo output:
        Channel 0 = baseline-dominant (structural / E8)
        Channel 1 = multidimensional encoding (qutrit + ququart)

    Args:
        result: Dict with keys columns, errorRate, complexity, invariants.
        output_path: Optional path to write a stereo 16-bit WAV file.

    Returns:
        Stereo int16 numpy array of shape (n_samples, 2).
    """
    _validate_input(result)

    # Deep copy to guarantee no mutation of input
    data = copy.deepcopy(result)

    columns: List[int] = data["columns"]
    error_rate: float = float(data["errorRate"])
    complexity: float = float(data["complexity"])
    invariants: List[Tuple[float, float]] = [
        (float(s), float(e)) for s, e in data["invariants"]
    ]

    num_samples = int(SAMPLE_RATE * DURATION)
    t = np.linspace(0.0, DURATION, num_samples, endpoint=False, dtype=np.float64)

    # Clamp complexity to [0, 1]
    complexity = max(0.0, min(1.0, complexity))

    # Energy distribution across channels
    baseline_energy = 1.0 - complexity
    multidim_energy = complexity

    # --- Channel 0: Baseline (E8 root ratios, sine voices) ---
    n_voices = len(columns)
    if n_voices == 0:
        ch0 = np.zeros(num_samples, dtype=np.float64)
    else:
        detune = math.log(error_rate + 1.0) * 1200.0
        amp = baseline_energy / float(n_voices)
        ch0 = np.zeros(num_samples, dtype=np.float64)
        for i, col in enumerate(columns):
            ratio = E8_ROOT_RATIOS[col % len(E8_ROOT_RATIOS)]
            freq = BASE_FREQ * ratio
            freq_mod = freq * (1.0 + detune * float(i + 1) * 1e-4)
            ch0 += amp * np.sin(2.0 * np.pi * freq_mod * t)

    # --- Channel 1: Multidimensional layers ---
    ch1 = np.zeros(num_samples, dtype=np.float64)

    if n_voices > 0:
        detune = math.log(error_rate + 1.0) * 1200.0

        # Qutrit layer (base-3): short pulses evenly spaced
        n_qutrit_pulses = max(1, n_voices)
        pulse_duration = DURATION / float(n_qutrit_pulses)
        for i, col in enumerate(columns):
            qutrit_val = col % 3  # {0, 1, 2}
            freq = BASE_FREQ * (CUSTOM_PI ** qutrit_val)
            freq_mod = freq * (1.0 + detune * float(i + 1) * 1e-4)
            # Pulse window: evenly spaced
            pulse_start = i * pulse_duration
            pulse_end = pulse_start + pulse_duration
            pulse_mask = (t >= pulse_start) & (t < pulse_end)
            qutrit_amp = multidim_energy * 0.5 / float(n_voices)
            signal = qutrit_amp * np.sin(2.0 * np.pi * freq_mod * t)
            ch1 += signal * pulse_mask

        # Ququart layer (base-4): triangle waveform, lower amplitude
        for i, col in enumerate(columns):
            ququart_val = col % 4  # {0, 1, 2, 3}
            freq = (BASE_FREQ / PHI) * float(ququart_val + 1)
            freq_mod = freq * (1.0 + detune * float(i + 1) * 1e-4)
            ququart_amp = multidim_energy * 0.3 / float(n_voices)
            phase = 2.0 * np.pi * freq_mod * t
            ch1 += ququart_amp * _triangle_wave(phase)

    # --- Assemble stereo output ---
    out = np.zeros((num_samples, 2), dtype=np.float64)
    out[:, 0] = ch0
    out[:, 1] = ch1

    # --- Invariant mask: FINAL STEP (after full signal composition) ---
    for start, end in invariants:
        mask = (t >= start) & (t <= end)
        out[mask, :] = 0.0

    # --- Single normalization pass ---
    max_val = np.max(np.abs(out))
    if max_val > 0.0:
        out = out / max_val

    # Convert to int16
    audio_int16 = np.clip(out * 32767.0, -32767.0, 32767.0).astype(np.int16)

    # Safety checks
    assert not np.any(np.isnan(out)), "NaN detected in audio"
    assert not np.any(np.isinf(out)), "Inf detected in audio"

    # Optionally write WAV
    if output_path is not None:
        _write_wav_stereo(output_path, audio_int16)

    return audio_int16


def _write_wav_stereo(filename: str, audio_int16: np.ndarray) -> None:
    """Write stereo int16 audio to a WAV file using stdlib only."""
    with wave.open(filename, "w") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        for frame in audio_int16:
            wf.writeframes(struct.pack("<hh", int(frame[0]), int(frame[1])))
