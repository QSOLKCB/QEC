"""Quantum Audio Logic Lab — deterministic spectral analysis of audio artifacts.

Provides frozen, replay-safe spectral analysis of MP3 files using raw byte
content.  No external audio-decoding libraries are required; spectral features
are derived deterministically from the file's byte stream interpreted as a
pseudo-waveform signal via numpy/scipy.

All outputs are frozen dataclasses with tuple-only collections.  Given the
same input file, results are byte-identical across runs.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
from scipy import signal as scipy_signal


# ---------------------------------------------------------------------------
# Report dataclass — frozen, tuple-only, replay-safe
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class QuantumAudioLogicReport:
    """Immutable spectral analysis report for a single audio artifact."""

    filename: str
    file_sha256: str
    file_size_bytes: int
    duration_seconds: float
    sample_rate: int
    dominant_frequency_hz: float
    spectral_centroid_hz: float
    spectral_entropy: float
    harmonic_density: float
    subharmonic_energy_ratio: float
    coherence_score: float
    stability_label: str
    mapping_2d: Tuple[float, float]
    cluster_points: Tuple[Tuple[float, float], ...]


# ---------------------------------------------------------------------------
# Coherence law weights
# ---------------------------------------------------------------------------

_W1_RESONANCE = 0.30
_W2_DENSITY = 0.25
_W3_SUBHARMONIC = 0.25
_WQ_INTERACTION = 0.15
_W4_ENTROPY = 0.05


def _coherence_score(
    resonance: float,
    density: float,
    subharmonic: float,
    entropy_norm: float,
) -> float:
    """Compute coherence score C = w1*R + w2*D + w3*S + wq*(R*S) - w4*H.

    All inputs are expected in [0, 1].  Result is clamped to [0, 1].
    """
    c = (
        _W1_RESONANCE * resonance
        + _W2_DENSITY * density
        + _W3_SUBHARMONIC * subharmonic
        + _WQ_INTERACTION * (resonance * subharmonic)
        - _W4_ENTROPY * entropy_norm
    )
    return float(max(0.0, min(1.0, c)))


# ---------------------------------------------------------------------------
# Stability classification
# ---------------------------------------------------------------------------

def _stability_label(coherence: float) -> str:
    if coherence >= 0.70:
        return "highly_coherent"
    if coherence >= 0.45:
        return "moderately_coherent"
    if coherence >= 0.25:
        return "weakly_coherent"
    return "incoherent"


# ---------------------------------------------------------------------------
# Deterministic cluster generation
# ---------------------------------------------------------------------------

def _deterministic_clusters(
    entropy: float,
    resonance: float,
    density: float,
    file_hash: str,
    n_points: int = 8,
) -> Tuple[Tuple[float, float], ...]:
    """Generate deterministic 2D cluster points from spectral features.

    Uses SHA-256 sub-seeds derived from the file hash to produce
    reproducible scatter without any RNG state.
    """
    points = []
    cx = resonance
    cy = density
    spread = entropy / 20.0  # entropy-derived variance

    for i in range(n_points):
        sub = hashlib.sha256(f"{file_hash}:cluster:{i}".encode()).digest()
        dx = (sub[0] / 255.0 - 0.5) * 2.0 * spread
        dy = (sub[1] / 255.0 - 0.5) * 2.0 * spread
        points.append((round(cx + dx, 8), round(cy + dy, 8)))

    return tuple(sorted(points))


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

_ASSUMED_SAMPLE_RATE = 44100
_TARGET_RESONANCE_HZ = 432.0
_SUBHARMONIC_HZ = 216.0
_HARMONIC_SERIES = (432.0, 864.0, 1296.0, 1728.0, 2160.0)
_BAND_HALF_WIDTH = 20.0  # Hz


def _bytes_to_signal(raw: bytes) -> np.ndarray:
    """Interpret raw file bytes as an int16 pseudo-waveform, normalised."""
    n = (len(raw) // 2) * 2
    samples = np.frombuffer(raw[:n], dtype=np.int16).astype(np.float64)
    mx = np.max(np.abs(samples))
    if mx > 0:
        samples = samples / mx
    return samples


def _welch_psd(
    samples: np.ndarray,
    sr: int,
    nperseg: int = 4096,
) -> Tuple[np.ndarray, np.ndarray]:
    """Welch power spectral density — deterministic, no overlap jitter."""
    seg = min(nperseg, len(samples))
    freqs, psd = scipy_signal.welch(
        samples,
        fs=sr,
        nperseg=seg,
        noverlap=seg // 2,
        window="hann",
        detrend=False,
        scaling="spectrum",
    )
    return freqs, psd


def _band_energy(
    freqs: np.ndarray,
    psd: np.ndarray,
    centre: float,
    half_width: float,
) -> float:
    mask = (freqs >= centre - half_width) & (freqs <= centre + half_width)
    return float(np.sum(psd[mask]))


def analyze_quantum_audio_file(path: str) -> QuantumAudioLogicReport:
    """Analyse an audio artifact and return a frozen spectral report.

    Parameters
    ----------
    path : str
        Filesystem path to the MP3 (or any binary audio) file.

    Returns
    -------
    QuantumAudioLogicReport
        Frozen, replay-safe analysis report derived deterministically
        from the file's byte content.
    """
    p = Path(path)
    raw = p.read_bytes()
    file_size = len(raw)
    file_hash = hashlib.sha256(raw).hexdigest()

    samples = _bytes_to_signal(raw)
    sr = _ASSUMED_SAMPLE_RATE
    duration = len(samples) / sr

    # --- Welch PSD (smoothed, deterministic) ---
    freqs, psd = _welch_psd(samples, sr)

    # Skip DC component
    freqs_ac = freqs[1:]
    psd_ac = psd[1:]

    total_energy = float(np.sum(psd_ac))
    if total_energy == 0:
        total_energy = 1.0  # guard

    # Dominant frequency
    peak_idx = int(np.argmax(psd_ac))
    dominant_freq = float(freqs_ac[peak_idx])

    # Spectral centroid
    centroid = float(np.sum(freqs_ac * psd_ac) / np.sum(psd_ac))

    # Spectral entropy (normalised probability over PSD)
    p_dist = psd_ac / np.sum(psd_ac)
    p_pos = p_dist[p_dist > 0]
    entropy = float(-np.sum(p_pos * np.log2(p_pos)))

    # Harmonic density — fraction of energy near 432 Hz harmonic series
    harmonic_energy = sum(
        _band_energy(freqs, psd, hf, _BAND_HALF_WIDTH)
        for hf in _HARMONIC_SERIES
    )
    harmonic_density = harmonic_energy / total_energy

    # Subharmonic energy ratio — 216 Hz region
    sub_energy = _band_energy(freqs, psd, _SUBHARMONIC_HZ, _BAND_HALF_WIDTH)
    subharmonic_ratio = sub_energy / total_energy

    # 432 Hz resonance closeness (energy fraction)
    res_energy = _band_energy(freqs, psd, _TARGET_RESONANCE_HZ, _BAND_HALF_WIDTH)
    resonance = res_energy / total_energy

    # --- Normalised inputs for coherence law ---
    # Resonance: already a ratio [0, ~1]
    # Scale density and subharmonic into [0,1] via sigmoid-like mapping
    R = float(min(1.0, resonance * 100.0))
    D = float(min(1.0, harmonic_density * 50.0))
    S = float(min(1.0, subharmonic_ratio * 200.0))
    max_entropy = math.log2(len(p_pos)) if len(p_pos) > 1 else 1.0
    H = float(min(1.0, entropy / max_entropy)) if max_entropy > 0 else 0.0

    coherence = _coherence_score(R, D, S, H)
    label = _stability_label(coherence)

    # --- Topology map ---
    mapping_x = round(R, 8)
    mapping_y = round(D, 8)
    mapping_2d = (mapping_x, mapping_y)

    clusters = _deterministic_clusters(entropy, R, D, file_hash)

    return QuantumAudioLogicReport(
        filename=p.name,
        file_sha256=file_hash,
        file_size_bytes=file_size,
        duration_seconds=round(duration, 6),
        sample_rate=sr,
        dominant_frequency_hz=round(dominant_freq, 4),
        spectral_centroid_hz=round(centroid, 4),
        spectral_entropy=round(entropy, 6),
        harmonic_density=round(harmonic_density, 8),
        subharmonic_energy_ratio=round(subharmonic_ratio, 8),
        coherence_score=round(coherence, 6),
        stability_label=label,
        mapping_2d=mapping_2d,
        cluster_points=clusters,
    )


# ---------------------------------------------------------------------------
# Comparative analysis
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ComparativeAnalysisResult:
    """Frozen comparison of two QuantumAudioLogicReports."""

    report_a: QuantumAudioLogicReport
    report_b: QuantumAudioLogicReport
    coherence_delta: float
    entropy_delta: float
    resonance_shift: float
    subharmonic_delta: float
    cluster_tightness_a: float
    cluster_tightness_b: float
    cluster_tightness_delta: float
    best_version: str


def _cluster_tightness(points: Tuple[Tuple[float, float], ...]) -> float:
    """Mean Euclidean distance of cluster points from their centroid."""
    if len(points) == 0:
        return 0.0
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    cx = sum(xs) / len(xs)
    cy = sum(ys) / len(ys)
    dists = [math.sqrt((x - cx) ** 2 + (y - cy) ** 2) for x, y in points]
    return sum(dists) / len(dists)


def compare_reports(
    a: QuantumAudioLogicReport,
    b: QuantumAudioLogicReport,
) -> ComparativeAnalysisResult:
    """Deterministic comparison of two spectral analysis reports."""
    tight_a = _cluster_tightness(a.cluster_points)
    tight_b = _cluster_tightness(b.cluster_points)

    coherence_delta = b.coherence_score - a.coherence_score
    entropy_delta = b.spectral_entropy - a.spectral_entropy

    # Resonance shift: difference in 432 Hz energy contribution
    # Derived from mapping_2d x-coordinate (normalised resonance)
    resonance_shift = b.mapping_2d[0] - a.mapping_2d[0]

    subharmonic_delta = b.subharmonic_energy_ratio - a.subharmonic_energy_ratio

    # Best version: higher coherence wins
    if a.coherence_score > b.coherence_score:
        best = a.filename
    elif b.coherence_score > a.coherence_score:
        best = b.filename
    else:
        best = "tie"

    return ComparativeAnalysisResult(
        report_a=a,
        report_b=b,
        coherence_delta=round(coherence_delta, 6),
        entropy_delta=round(entropy_delta, 6),
        resonance_shift=round(resonance_shift, 8),
        subharmonic_delta=round(subharmonic_delta, 8),
        cluster_tightness_a=round(tight_a, 8),
        cluster_tightness_b=round(tight_b, 8),
        cluster_tightness_delta=round(tight_b - tight_a, 8),
        best_version=best,
    )
