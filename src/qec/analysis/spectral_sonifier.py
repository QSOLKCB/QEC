"""Deterministic spectral sonification utilities."""

from __future__ import annotations

import struct
import wave

import numpy as np


def eigenvalue_to_frequency(ev: float, base: float = 220.0, scale: float = 120.0) -> float:
    """Map a spectral value to an audible frequency using a deterministic linear mapping."""
    return base + scale * float(ev)


def ipr_to_amplitude(ipr: float, max_amp: float = 0.9) -> float:
    """Map localization metric (IPR) to amplitude."""
    ipr_value = float(ipr)
    return max_amp * min(max(ipr_value, 0.0), 1.0)


def sine_wave(freq: float, duration: float, sr: int = 44100, amplitude: float = 0.5) -> np.ndarray:
    """Generate a deterministic sine wave."""
    t = np.linspace(0.0, duration, int(sr * duration), endpoint=False, dtype=np.float64)
    return amplitude * np.sin(2.0 * np.pi * freq * t)


def sonify_spectrum(
    eigenvalues: list[float] | np.ndarray,
    ipr: list[float] | np.ndarray | None = None,
    duration: float = 0.2,
    sr: int = 44100,
) -> np.ndarray:
    """Convert a sequence of spectral eigenvalues into a waveform."""
    audio_segments: list[np.ndarray] = []

    for index, ev in enumerate(eigenvalues):
        freq = eigenvalue_to_frequency(ev)

        amp = 0.5
        if ipr is not None and index < len(ipr):
            amp = ipr_to_amplitude(ipr[index])

        tone = sine_wave(freq, duration, sr, amp)
        audio_segments.append(tone)

    if not audio_segments:
        return np.zeros(1, dtype=np.float64)

    return np.concatenate(audio_segments)


def write_wav(filename: str, audio: np.ndarray, sr: int = 44100) -> None:
    """Write waveform to a mono 16-bit WAV file."""
    audio_array = np.asarray(audio, dtype=np.float64)
    normalized = audio_array / max(np.max(np.abs(audio_array)), 1e-12)

    with wave.open(filename, "w") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sr)

        for sample in normalized:
            wav_file.writeframes(struct.pack("<h", int(sample * 32767)))


def sonify_experiment_artifact(artifact: dict, output_file: str) -> None:
    """Sonify spectral data stored in an experiment artifact."""
    eigs = artifact.get("nb_eigenvalues", [])
    ipr = artifact.get("ipr_values", None)

    audio = sonify_spectrum(eigs, ipr)
    write_wav(output_file, audio)
