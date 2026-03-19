"""
v74.0.0 — Deterministic sonic artifact spectral analysis.

Loads audio files (WAV required, MP3 if environment supports it),
computes spectral and structural features, and generates visual + JSON
artifacts.  Treats audio data as read-only.

Layer 5 — Experiments.
Does not modify decoder internals.  Fully deterministic.
"""

from __future__ import annotations

import io
import json
import math
import os
import struct
import wave
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Audio Loading
# ---------------------------------------------------------------------------

def load_audio(path: str) -> Tuple[np.ndarray, int]:
    """Load an audio file and return (signal, sample_rate).

    Parameters
    ----------
    path : str
        Path to a WAV or MP3 file.

    Returns
    -------
    signal : np.ndarray
        1-D float64 array normalised to [-1, 1].
    sr : int
        Sample rate in Hz.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    RuntimeError
        If the format is unsupported or decoding fails.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    ext = p.suffix.lower()
    if ext == ".wav":
        return _load_wav(path)
    elif ext == ".mp3":
        return _load_mp3(path)
    else:
        raise RuntimeError(f"Unsupported audio format: {ext}")


def _load_wav(path: str) -> Tuple[np.ndarray, int]:
    """Load WAV via stdlib wave module."""
    with wave.open(path, "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        sr = wf.getframerate()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    if sampwidth == 1:
        dtype = np.uint8
    elif sampwidth == 2:
        dtype = np.int16
    elif sampwidth == 4:
        dtype = np.int32
    else:
        raise RuntimeError(f"Unsupported sample width: {sampwidth}")

    data = np.frombuffer(raw, dtype=dtype).astype(np.float64)

    # Normalise to [-1, 1]
    if sampwidth == 1:
        data = (data - 128.0) / 128.0
    elif sampwidth == 2:
        data = data / 32768.0
    elif sampwidth == 4:
        data = data / 2147483648.0

    # Mix to mono by averaging channels
    if n_channels > 1:
        data = data.reshape(-1, n_channels).mean(axis=1)

    return data, sr


def _load_mp3(path: str) -> Tuple[np.ndarray, int]:
    """Load MP3 using available backend.

    Tries in order:
    1. scipy.io.wavfile via subprocess ffmpeg decode
    2. Raises clear error if no decoder available
    """
    # Try ffmpeg subprocess decode to WAV in memory
    import subprocess
    import tempfile

    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        result = subprocess.run(
            [
                "ffmpeg", "-y", "-i", path,
                "-ac", "1",        # mono
                "-ar", "44100",    # standard sample rate
                "-sample_fmt", "s16",
                "-f", "wav",
                tmp_path,
            ],
            capture_output=True,
            timeout=120,
        )
        if result.returncode == 0:
            signal, sr = _load_wav(tmp_path)
            return signal, sr
    except FileNotFoundError:
        pass  # ffmpeg not installed
    except Exception:
        pass
    finally:
        try:
            os.unlink(tmp_path)
        except (OSError, UnboundLocalError):
            pass

    raise RuntimeError(
        "MP3 decoding is not available in this environment. "
        "Install ffmpeg or convert to WAV format."
    )


# ---------------------------------------------------------------------------
# Feature Extraction
# ---------------------------------------------------------------------------

def compute_features(signal: np.ndarray, sr: int) -> Dict[str, Any]:
    """Compute deterministic spectral and structural features.

    Parameters
    ----------
    signal : np.ndarray
        1-D float64 mono audio.
    sr : int
        Sample rate in Hz.

    Returns
    -------
    dict
        Feature dictionary with all computed metrics.
    """
    n = len(signal)
    duration = float(n) / float(sr)

    # RMS energy
    rms_energy = float(np.sqrt(np.mean(signal ** 2)))

    # Peak amplitude
    peak_amplitude = float(np.max(np.abs(signal)))

    # Zero-crossing rate
    zcr = float(np.sum(np.abs(np.diff(np.sign(signal))) > 0)) / max(n - 1, 1)

    # FFT (positive frequencies only)
    fft_vals = np.fft.rfft(signal)
    fft_mag = np.abs(fft_vals)
    freqs = np.fft.rfftfreq(n, d=1.0 / sr)

    # Spectral centroid
    mag_sum = np.sum(fft_mag)
    if mag_sum > 0:
        spectral_centroid = float(np.sum(freqs * fft_mag) / mag_sum)
    else:
        spectral_centroid = 0.0

    # Spectral spread (standard deviation around centroid)
    if mag_sum > 0:
        spectral_spread = float(
            np.sqrt(np.sum(((freqs - spectral_centroid) ** 2) * fft_mag) / mag_sum)
        )
    else:
        spectral_spread = 0.0

    # FFT magnitude profile summary (top-20 peaks by magnitude)
    n_peaks = min(20, len(fft_mag))
    top_indices = np.argsort(fft_mag)[-n_peaks:][::-1]
    fft_profile = [
        {"frequency_hz": float(freqs[i]), "magnitude": float(fft_mag[i])}
        for i in top_indices
    ]

    return {
        "duration_seconds": duration,
        "sample_rate": sr,
        "n_samples": n,
        "rms_energy": rms_energy,
        "peak_amplitude": peak_amplitude,
        "zero_crossing_rate": zcr,
        "spectral_centroid_hz": spectral_centroid,
        "spectral_spread_hz": spectral_spread,
        "fft_top_peaks": fft_profile,
    }


# ---------------------------------------------------------------------------
# STFT Spectrogram (numpy-only)
# ---------------------------------------------------------------------------

def compute_spectrogram(
    signal: np.ndarray,
    sr: int,
    *,
    n_fft: int = 2048,
    hop_length: int = 512,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute STFT magnitude spectrogram.

    Returns
    -------
    S_db : np.ndarray
        Magnitude in dB, shape (n_freq, n_frames).
    freqs : np.ndarray
        Frequency bin centres.
    times : np.ndarray
        Frame centre times.
    """
    window = np.hanning(n_fft)
    n_frames = 1 + (len(signal) - n_fft) // hop_length
    if n_frames < 1:
        # Signal shorter than one FFT window — zero-pad
        padded = np.zeros(n_fft, dtype=signal.dtype)
        padded[: len(signal)] = signal
        n_frames = 1
        frames = padded.reshape(1, -1)
    else:
        frames = np.lib.stride_tricks.as_strided(
            signal,
            shape=(n_frames, n_fft),
            strides=(signal.strides[0] * hop_length, signal.strides[0]),
        ).copy()  # copy to avoid mutation via strides

    windowed = frames * window
    fft_result = np.fft.rfft(windowed, n=n_fft, axis=1)
    magnitude = np.abs(fft_result).T  # shape (n_freq, n_frames)

    # Convert to dB with floor at -80 dB
    ref = np.max(magnitude)
    if ref > 0:
        S_db = 20.0 * np.log10(np.maximum(magnitude, ref * 1e-4) / ref)
    else:
        S_db = np.full_like(magnitude, -80.0)

    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    times = np.arange(n_frames) * hop_length / sr

    return S_db, freqs, times


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def save_waveform_plot(
    signal: np.ndarray,
    sr: int,
    path: str,
) -> None:
    """Save a time-domain waveform plot to *path*."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    times = np.arange(len(signal)) / sr
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(times, signal, linewidth=0.3, color="#2c3e50")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Waveform")
    ax.set_xlim(0, times[-1] if len(times) > 0 else 1)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_spectrogram_plot(
    S_db: np.ndarray,
    freqs: np.ndarray,
    times: np.ndarray,
    path: str,
) -> None:
    """Save a spectrogram image to *path*."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.pcolormesh(
        times,
        freqs,
        S_db,
        shading="auto",
        cmap="inferno",
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title("Spectrogram (dB)")
    ax.set_ylim(0, min(freqs[-1], 8000))
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Per-File Analysis Pipeline
# ---------------------------------------------------------------------------

def analyse_file(
    path: str,
    output_dir: str,
) -> Dict[str, Any]:
    """Run full analysis on a single audio file.

    Parameters
    ----------
    path : str
        Path to the audio file.
    output_dir : str
        Directory to write artifacts into.

    Returns
    -------
    dict
        The analysis results (same content as written to analysis.json).
    """
    signal, sr = load_audio(path)

    features = compute_features(signal, sr)
    features["source_file"] = os.path.basename(path)

    os.makedirs(output_dir, exist_ok=True)

    # Spectrogram
    S_db, freqs, times = compute_spectrogram(signal, sr)
    save_spectrogram_plot(S_db, freqs, times, os.path.join(output_dir, "spectrogram.png"))

    # Waveform
    save_waveform_plot(signal, sr, os.path.join(output_dir, "waveform.png"))

    # JSON
    json_path = os.path.join(output_dir, "analysis.json")
    with open(json_path, "w") as f:
        json.dump(features, f, indent=2, sort_keys=True)

    return features
