"""Deterministic E8 baseline sonification engine.

Sonification spec:
    - SAMPLE_RATE = 44100 Hz
    - DURATION = 5.0 s
    - BASE_FREQ = 110.0 Hz
    - 16 E8 root ratios mapped to column indices
    - Error rate drives pitch modulation via log detune
    - Complexity controls per-voice amplitude
    - Invariant intervals impose hard silence gates
    - Output: mono int16 numpy array, optionally written as WAV

Invariants:
    - Deterministic: same input -> identical bytes
    - No mutation of input dict
    - No NaN / inf in output
    - Bounded: abs(sample) <= 32767
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


def sonify_e8_baseline(
    result: dict,
    output_path: Optional[str] = None,
) -> np.ndarray:
    """Generate a deterministic E8 baseline sonification from a result dict.

    Args:
        result: Dict with keys columns, errorRate, complexity, invariants.
        output_path: Optional path to write a mono 16-bit WAV file.

    Returns:
        Mono int16 numpy array of the sonified signal.
    """
    _validate_input(result)

    # Work on a deep copy to guarantee no mutation of input
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

    n_voices = len(columns)
    if n_voices == 0:
        # No columns: silent output
        audio = np.zeros(num_samples, dtype=np.float64)
    else:
        # Detune factor from error rate
        detune = math.log(error_rate + 1.0) * 1200.0

        # Per-voice amplitude
        amp = (1.0 - complexity) / float(n_voices)

        # Sum sine voices
        audio = np.zeros(num_samples, dtype=np.float64)
        for i, col in enumerate(columns):
            ratio = E8_ROOT_RATIOS[col % len(E8_ROOT_RATIOS)]
            freq = BASE_FREQ * ratio
            freq_mod = freq * (1.0 + detune * float(i + 1) * 1e-4)
            audio += amp * np.sin(2.0 * np.pi * freq_mod * t)

    # Apply silence gates from invariants (hard mask, after summation)
    for start, end in invariants:
        mask = (t >= start) & (t <= end)
        audio[mask] = 0.0

    # Normalize to prevent clipping, preserve relative structure
    peak = np.max(np.abs(audio))
    if peak > 0.0:
        audio = audio / peak

    # Convert to int16
    audio_int16 = np.clip(audio * 32767.0, -32767.0, 32767.0).astype(np.int16)

    # Safety checks
    assert not np.any(np.isnan(audio)), "NaN detected in audio"
    assert not np.any(np.isinf(audio)), "Inf detected in audio"

    # Optionally write WAV
    if output_path is not None:
        _write_wav(output_path, audio_int16)

    return audio_int16


def _write_wav(filename: str, audio_int16: np.ndarray) -> None:
    """Write mono int16 audio to a WAV file using stdlib only."""
    with wave.open(filename, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        for sample in audio_int16:
            wf.writeframes(struct.pack("<h", int(sample)))
