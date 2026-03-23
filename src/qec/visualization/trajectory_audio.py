"""v86.2.0 — Deterministic sonification of spectral trajectories.

Maps trajectory dynamics to audio events using only numpy + stdlib
``wave`` / ``struct`` modules.  Same input always produces identical
WAV output (16-bit PCM, 44100 Hz).
"""

from __future__ import annotations

import struct
import wave
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np


# ── constants ────────────────────────────────────────────────────────

SAMPLE_RATE: int = 44100
STEP_DURATION: float = 0.2  # seconds per trajectory step
_BASE_FREQ: float = 220.0
_CLICK_FREQ: float = 1760.0
_CLICK_DURATION: float = 0.01  # seconds


# ── helpers ──────────────────────────────────────────────────────────


def _normalize(values: List[float]) -> List[float]:
    """Min-max normalize *values* to [0, 1].  Constant → all 0.5."""
    if not values:
        return []
    lo = min(values)
    hi = max(values)
    if hi == lo:
        return [0.5] * len(values)
    return [(v - lo) / (hi - lo) for v in values]


def _sine_segment(freq: float, duration: float, amplitude: float) -> np.ndarray:
    """Return a deterministic sine-wave segment (float64, [-1, 1])."""
    n_samples = int(SAMPLE_RATE * duration)
    t = np.arange(n_samples, dtype=np.float64) / SAMPLE_RATE
    return amplitude * np.sin(2.0 * np.pi * freq * t)


def _ending_motif(trajectory_type: str) -> np.ndarray:
    """Return a short closing motif determined by trajectory type."""
    dur = 0.15
    if trajectory_type == "convergent":
        # rising 2-tone
        return np.concatenate([
            _sine_segment(330.0, dur, 0.6),
            _sine_segment(440.0, dur, 0.6),
        ])
    if trajectory_type == "oscillatory":
        # alternating tones
        return np.concatenate([
            _sine_segment(350.0, dur, 0.5),
            _sine_segment(280.0, dur, 0.5),
        ])
    if trajectory_type == "divergent":
        # descending tone
        return np.concatenate([
            _sine_segment(440.0, dur, 0.6),
            _sine_segment(330.0, dur, 0.6),
        ])
    # undetermined → flat tone
    return _sine_segment(300.0, dur * 2, 0.4)


# ── main entry point ────────────────────────────────────────────────


def sonify_spectral_trajectory(
    traj: Dict[str, Any],
    output_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Sonify a spectral trajectory to a 16-bit PCM WAV file.

    Parameters
    ----------
    traj : dict
        Output of ``run_phase_trajectory_analysis``.
    output_path : str or Path, optional
        Destination WAV path.  When *None* a default temp path is used.

    Returns
    -------
    dict
        ``{"output_path": str, "duration": float}``
    """
    lambda_max: List[float] = traj.get("lambda_max", [])
    drift: List[float] = traj.get("drift", [])
    transitions: List[Dict[str, Any]] = traj.get("temporal_transitions", [])
    trajectory_type: str = traj.get("trajectory_type", "undetermined")

    norm_lm = _normalize(lambda_max)
    transition_set = {tr["time_index"] for tr in transitions}

    # Derive per-step amplitude scale from drift.
    # drift has len(lambda_max) - 1 entries; prepend 0 for step 0.
    drift_padded = [0.0] + list(drift)
    max_drift = max(drift_padded) if drift_padded else 1.0
    scale_factor = 1.0 / max_drift if max_drift > 0 else 1.0

    segments: List[np.ndarray] = []
    n_samples_per_step = int(SAMPLE_RATE * STEP_DURATION)

    for t, nlm in enumerate(norm_lm):
        freq = _BASE_FREQ + _BASE_FREQ * nlm
        amp = min(1.0, drift_padded[t] * scale_factor) if t < len(drift_padded) else 0.5
        amp = max(0.05, amp)  # ensure audibility

        seg = _sine_segment(freq, STEP_DURATION, amp)

        # insert click at transition boundaries
        if t in transition_set:
            click_samples = int(SAMPLE_RATE * _CLICK_DURATION)
            click = _sine_segment(_CLICK_FREQ, _CLICK_DURATION, 0.9)
            seg[:click_samples] = click[:len(seg[:click_samples])]

        segments.append(seg)

    # append ending motif
    segments.append(_ending_motif(trajectory_type))

    if segments:
        waveform = np.concatenate(segments)
    else:
        waveform = np.zeros(int(SAMPLE_RATE * 0.1), dtype=np.float64)

    # clamp to [-1, 1]
    waveform = np.clip(waveform, -1.0, 1.0)

    duration = float(len(waveform)) / SAMPLE_RATE

    # ── write WAV ────────────────────────────────────────────────────
    if output_path is None:
        import tempfile
        output_path = Path(tempfile.gettempdir()) / "trajectory_audio.wav"

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    int16_data = (waveform * 32767).astype(np.int16)
    raw_bytes = struct.pack(f"<{len(int16_data)}h", *int16_data)

    with wave.open(str(out), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(raw_bytes)

    return {
        "output_path": str(out),
        "duration": duration,
    }
