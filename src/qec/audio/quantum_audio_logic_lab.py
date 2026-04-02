"""Quantum Audio Logic Lab — deterministic spectral analysis of audio artifacts.

Provides frozen, replay-safe spectral analysis of audio files.  Attempts
real waveform decode via available libraries (scipy.io.wavfile, soundfile,
audioread) before falling back to deterministic byte-window pseudo-spectrum.

MP3 frame headers are parsed natively (no external dependency) to extract
true sample rate, duration, bitrate, and channel count regardless of
decode path.

All outputs are frozen dataclasses with tuple-only collections.  Given the
same input file, results are byte-identical across runs.

Decode priority order:
  1. scipy.io.wavfile / soundfile / audioread  (real waveform)
  2. Deterministic byte-window pseudo-spectrum  (fallback)
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

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
    decode_path: str


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
# MP3 frame header parser — pure stdlib, no external dependency
# ---------------------------------------------------------------------------

# Bitrate tables: [version][layer][index] -> kbps (0 = free/bad)
_BITRATE_TABLE = {
    # MPEG1
    1: {
        1: (0, 32, 64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384, 416, 448, 0),
        2: (0, 32, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 384, 0),
        3: (0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 0),
    },
    # MPEG2 / MPEG2.5
    2: {
        1: (0, 32, 48, 56, 64, 80, 96, 112, 128, 144, 160, 176, 192, 224, 256, 0),
        2: (0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160, 0),
        3: (0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160, 0),
    },
}

_SAMPLE_RATE_TABLE = {
    1: (44100, 48000, 32000),
    2: (22050, 24000, 16000),
    2.5: (11025, 12000, 8000),
}

_VERSION_MAP = {0: 2.5, 2: 2, 3: 1}
_LAYER_MAP = {1: 3, 2: 2, 3: 1}


@dataclass(frozen=True)
class _Mp3FrameHeader:
    version: float
    layer: int
    sample_rate: int
    bitrate: int
    frame_length: int
    channels: int
    padding: int


def _parse_mp3_frame_header(data: bytes, offset: int) -> Optional[_Mp3FrameHeader]:
    """Parse a single MP3 frame header at *offset*.  Returns None on failure."""
    if offset + 4 > len(data):
        return None
    header = struct.unpack(">I", data[offset : offset + 4])[0]

    # Frame sync: 11 bits of 1s
    if (header >> 21) != 0x7FF:
        return None

    version_bits = (header >> 19) & 0x3
    layer_bits = (header >> 17) & 0x3
    bitrate_idx = (header >> 12) & 0xF
    sr_idx = (header >> 10) & 0x3
    padding = (header >> 9) & 0x1
    channel_mode = (header >> 6) & 0x3

    version = _VERSION_MAP.get(version_bits)
    layer = _LAYER_MAP.get(layer_bits)
    if version is None or layer is None:
        return None

    sr_row = _SAMPLE_RATE_TABLE.get(version)
    if sr_row is None or sr_idx >= 3:
        return None
    sample_rate = sr_row[sr_idx]

    br_ver = 1 if version == 1 else 2
    br_row = _BITRATE_TABLE.get(br_ver, {}).get(layer)
    if br_row is None or bitrate_idx >= len(br_row):
        return None
    bitrate = br_row[bitrate_idx] * 1000
    if bitrate == 0:
        return None

    # Frame length calculation
    if layer == 1:
        frame_length = (12 * bitrate // sample_rate + padding) * 4
    else:
        slot = 144 if version == 1 else 72
        frame_length = slot * bitrate // sample_rate + padding

    if frame_length <= 0:
        return None

    channels = 1 if channel_mode == 3 else 2

    return _Mp3FrameHeader(
        version=version,
        layer=layer,
        sample_rate=sample_rate,
        bitrate=bitrate,
        frame_length=frame_length,
        channels=channels,
        padding=padding,
    )


@dataclass(frozen=True)
class _Mp3Metadata:
    sample_rate: int
    bitrate: int
    channels: int
    frame_count: int
    total_samples: int
    duration_seconds: float
    audio_data_offset: int


def _parse_mp3_metadata(data: bytes) -> Optional[_Mp3Metadata]:
    """Scan MP3 byte stream for frame headers and extract real metadata."""
    offset = 0

    # Skip ID3v2 tag if present
    if len(data) >= 10 and data[:3] == b"ID3":
        tag_size = (
            (data[6] & 0x7F) << 21
            | (data[7] & 0x7F) << 14
            | (data[8] & 0x7F) << 7
            | (data[9] & 0x7F)
        )
        offset = 10 + tag_size

    audio_data_offset = offset
    frame_count = 0
    first_frame: Optional[_Mp3FrameHeader] = None

    # Scan for valid frames — require 3 consecutive valid frames
    # to confirm we've found the audio stream
    scan_limit = min(len(data), offset + 65536)
    while offset < scan_limit:
        frame = _parse_mp3_frame_header(data, offset)
        if frame is not None and frame.frame_length > 0:
            second = _parse_mp3_frame_header(data, offset + frame.frame_length)
            if second is not None and second.frame_length > 0:
                third = _parse_mp3_frame_header(
                    data, offset + frame.frame_length + second.frame_length,
                )
                if third is not None:
                    first_frame = frame
                    audio_data_offset = offset
                    break
        offset += 1

    if first_frame is None:
        return None

    # Count all frames from the confirmed start
    offset = audio_data_offset
    while offset < len(data) - 4:
        frame = _parse_mp3_frame_header(data, offset)
        if frame is not None and frame.frame_length > 0:
            frame_count += 1
            offset += frame.frame_length
        else:
            break  # end of frame stream

    if frame_count == 0:
        return None

    samples_per_frame = 1152 if first_frame.version == 1 else 576
    total_samples = frame_count * samples_per_frame
    duration = total_samples / first_frame.sample_rate

    return _Mp3Metadata(
        sample_rate=first_frame.sample_rate,
        bitrate=first_frame.bitrate,
        channels=first_frame.channels,
        frame_count=frame_count,
        total_samples=total_samples,
        duration_seconds=duration,
        audio_data_offset=audio_data_offset,
    )


# ---------------------------------------------------------------------------
# Spectral constants
# ---------------------------------------------------------------------------

_DEFAULT_SAMPLE_RATE = 44100
_TARGET_RESONANCE_HZ = 432.0
_SUBHARMONIC_HZ = 216.0
_HARMONIC_SERIES = (432.0, 864.0, 1296.0, 1728.0, 2160.0)
_BAND_HALF_WIDTH = 20.0  # Hz


# ---------------------------------------------------------------------------
# Decode path 1 — real waveform decode
# ---------------------------------------------------------------------------

def _try_decode_audio(path: str) -> Optional[Tuple[np.ndarray, int, str]]:
    """Attempt to decode audio file to PCM waveform.

    Returns (samples_float64_mono, sample_rate, decoder_name) or None.
    Tries: scipy.io.wavfile, soundfile, audioread — in that order.
    """
    p = Path(path)

    # --- scipy.io.wavfile (WAV files only) ---
    if p.suffix.lower() in (".wav", ".wave"):
        try:
            from scipy.io import wavfile
            sr, data = wavfile.read(path)
            samples = data.astype(np.float64)
            if samples.ndim > 1:
                samples = samples.mean(axis=1)
            mx = np.max(np.abs(samples))
            if mx > 0:
                samples = samples / mx
            return samples, sr, "scipy.io.wavfile"
        except Exception:
            pass

    # --- soundfile (many formats including MP3 via libsndfile) ---
    try:
        import soundfile as sf
        data, sr = sf.read(path, dtype="float64", always_2d=True)
        samples = data.mean(axis=1)
        mx = np.max(np.abs(samples))
        if mx > 0:
            samples = samples / mx
        return samples, sr, "soundfile"
    except Exception:
        pass

    # --- audioread (MP3 via system backends) ---
    try:
        import audioread
        with audioread.audio_open(path) as f:
            sr = f.samplerate
            channels = f.channels
            blocks = []
            for block in f:
                blocks.append(block)
        buf = b"".join(blocks)
        n = (len(buf) // 2) * 2
        samples = np.frombuffer(buf[:n], dtype=np.int16).astype(np.float64)
        if channels > 1:
            samples = samples.reshape(-1, channels).mean(axis=1)
        mx = np.max(np.abs(samples))
        if mx > 0:
            samples = samples / mx
        return samples, sr, "audioread"
    except Exception:
        pass

    return None


# ---------------------------------------------------------------------------
# Decode path 2 — byte-window pseudo-spectrum (fallback)
# ---------------------------------------------------------------------------

def _compute_byte_pseudospectrum(
    raw_bytes: bytes,
    sample_rate: int,
    audio_offset: int = 0,
) -> Tuple[np.ndarray, int]:
    """Interpret raw audio bytes as int16 pseudo-waveform.

    Uses audio_offset to skip non-audio headers (e.g. ID3 tags) so that
    the pseudo-waveform is derived from actual encoded audio frames.
    """
    audio_data = raw_bytes[audio_offset:]
    n = (len(audio_data) // 2) * 2
    if n < 2:
        audio_data = raw_bytes
        n = (len(audio_data) // 2) * 2
    samples = np.frombuffer(audio_data[:n], dtype=np.int16).astype(np.float64)
    mx = np.max(np.abs(samples))
    if mx > 0:
        samples = samples / mx
    return samples, sample_rate


# ---------------------------------------------------------------------------
# Shared spectral computation
# ---------------------------------------------------------------------------

def _welch_psd(
    samples: np.ndarray,
    sr: int,
    nperseg: int = 4096,
) -> Tuple[np.ndarray, np.ndarray]:
    """Welch power spectral density — deterministic, no overlap jitter."""
    if len(samples) == 0:
        return np.array([0.0]), np.array([0.0])
    seg = max(1, min(nperseg, len(samples)))
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


def _compute_waveform_spectrum(
    samples: np.ndarray,
    sr: int,
) -> dict:
    """Compute all spectral features from a waveform signal.

    Returns a dict of raw spectral measurements (before normalisation).
    """
    freqs, psd = _welch_psd(samples, sr)

    # Skip DC component
    freqs_ac = freqs[1:]
    psd_ac = psd[1:]

    total_energy = float(np.sum(psd_ac))
    if total_energy == 0:
        total_energy = 1.0

    # Dominant frequency
    peak_idx = int(np.argmax(psd_ac))
    dominant_freq = float(freqs_ac[peak_idx])

    # Spectral centroid
    sum_psd = float(np.sum(psd_ac))
    centroid = float(np.sum(freqs_ac * psd_ac) / sum_psd) if sum_psd > 0 else 0.0

    # Spectral entropy (normalised probability over PSD)
    p_dist = psd_ac / sum_psd if sum_psd > 0 else np.ones_like(psd_ac) / len(psd_ac)
    p_pos = p_dist[p_dist > 0]
    entropy = float(-np.sum(p_pos * np.log2(p_pos)))
    max_entropy = math.log2(len(p_pos)) if len(p_pos) > 1 else 1.0

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

    return {
        "dominant_freq": dominant_freq,
        "centroid": centroid,
        "entropy": entropy,
        "max_entropy": max_entropy,
        "harmonic_density": harmonic_density,
        "subharmonic_ratio": subharmonic_ratio,
        "resonance": resonance,
    }


# ---------------------------------------------------------------------------
# Public analysis entry point
# ---------------------------------------------------------------------------

def analyze_quantum_audio_file(path: str) -> QuantumAudioLogicReport:
    """Analyse an audio artifact and return a frozen spectral report.

    Attempts real waveform decode first (Priority 1), then falls back to
    deterministic byte-window pseudo-spectrum (Priority 3).  MP3 metadata
    (sample rate, duration) is always extracted from frame headers when
    the file is MP3-formatted.

    Parameters
    ----------
    path : str
        Filesystem path to the audio file.

    Returns
    -------
    QuantumAudioLogicReport
        Frozen, replay-safe analysis report derived deterministically
        from the file's content.
    """
    p = Path(path)
    raw = p.read_bytes()
    file_size = len(raw)
    file_hash = hashlib.sha256(raw).hexdigest()

    # --- Extract MP3 metadata if available ---
    mp3_meta = _parse_mp3_metadata(raw)

    # --- Attempt real waveform decode (Priority 1) ---
    decode_result = _try_decode_audio(path)

    if decode_result is not None:
        samples, sr, decoder_name = decode_result
        decode_path_label = f"waveform:{decoder_name}"
        # Trust the decoder's sample rate for spectral analysis — the
        # decoder may have resampled.  Use MP3 metadata only for
        # authoritative duration (frame-counted, more accurate than
        # len(samples)/sr for decoded streams with padding).
        duration = len(samples) / sr
        if mp3_meta is not None:
            duration = mp3_meta.duration_seconds
    else:
        # --- Fallback: byte-window pseudo-spectrum (Priority 3) ---
        if mp3_meta is not None:
            sr = mp3_meta.sample_rate
            duration = mp3_meta.duration_seconds
            audio_offset = mp3_meta.audio_data_offset
        else:
            sr = _DEFAULT_SAMPLE_RATE
            audio_offset = 0
            duration = 0.0  # computed below

        samples, sr = _compute_byte_pseudospectrum(raw, sr, audio_offset)
        decode_path_label = "byte_pseudospectrum"

        if mp3_meta is None:
            duration = len(samples) / sr

    # --- Compute spectral features ---
    spectral = _compute_waveform_spectrum(samples, sr)

    # --- Normalised inputs for coherence law ---
    R = float(min(1.0, spectral["resonance"] * 100.0))
    D = float(min(1.0, spectral["harmonic_density"] * 50.0))
    S = float(min(1.0, spectral["subharmonic_ratio"] * 200.0))
    max_ent = spectral["max_entropy"]
    H = float(min(1.0, spectral["entropy"] / max_ent)) if max_ent > 0 else 0.0

    coherence = _coherence_score(R, D, S, H)
    label = _stability_label(coherence)

    # --- Topology map ---
    mapping_x = round(R, 8)
    mapping_y = round(D, 8)
    mapping_2d = (mapping_x, mapping_y)

    clusters = _deterministic_clusters(spectral["entropy"], R, D, file_hash)

    return QuantumAudioLogicReport(
        filename=p.name,
        file_sha256=file_hash,
        file_size_bytes=file_size,
        duration_seconds=round(duration, 6),
        sample_rate=sr,
        dominant_frequency_hz=round(spectral["dominant_freq"], 4),
        spectral_centroid_hz=round(spectral["centroid"], 4),
        spectral_entropy=round(spectral["entropy"], 6),
        harmonic_density=round(spectral["harmonic_density"], 8),
        subharmonic_energy_ratio=round(spectral["subharmonic_ratio"], 8),
        coherence_score=round(coherence, 6),
        stability_label=label,
        mapping_2d=mapping_2d,
        cluster_points=clusters,
        decode_path=decode_path_label,
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

    resonance_shift = b.mapping_2d[0] - a.mapping_2d[0]

    subharmonic_delta = b.subharmonic_energy_ratio - a.subharmonic_energy_ratio

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


# ---------------------------------------------------------------------------
# Legacy sequence constants
# ---------------------------------------------------------------------------

_LEGACY_STATE_NAMES: Tuple[str, ...] = (
    "Stable",
    "Instability",
    "Transition",
    "Collapse",
    "Recovery",
)

_LEGACY_STATE_FILES: Tuple[str, ...] = (
    "QSOL 1_ Stable Invariant State.mp3",
    "QSOL 2_ Near-Boundary Instability.mp3",
    "QSOL 3_ Structural Transition.mp3",
    "QSOL 4_ Constraint Failure _ Collapse.mp3",
    "QSOL 5_ Recovery _ Re-stabilization.mp3",
)

_LEGACY_TRANSITION_CLASSES: Tuple[str, ...] = (
    "divergent",
    "transition",
    "collapse",
    "recovery",
)


# ---------------------------------------------------------------------------
# v136.6.1 — Legacy report dataclasses (frozen, tuple-only)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LegacySequenceTransitionReport:
    """Frozen report for a single legacy state-to-state transition."""

    from_state: str
    to_state: str
    classification: str
    centroid_delta: float
    entropy_delta: float
    harmonic_density_delta: float
    subharmonic_delta: float
    coherence_delta: float
    cluster_tightness_delta: float
    psd_similarity: float
    legacy_centroid_delta: float
    centroid_drift: float


@dataclass(frozen=True)
class LegacyAudioSequenceReport:
    """Frozen report for the full 5-state legacy audio sequence."""

    n_states: int
    state_reports: Tuple[QuantumAudioLogicReport, ...]
    transition_reports: Tuple[LegacySequenceTransitionReport, ...]
    legacy_similarity_score: float
    topology_drift_score: float
    best_recovery_match: str
    dependency_health: Tuple[Tuple[str, str], ...]
    stability_verdict: str


@dataclass(frozen=True)
class CrossEraComparisonReport:
    """Frozen comparison of legacy chain against v136.6 artifacts."""

    best_legacy_analog_v1: str
    best_legacy_analog_v2: str
    collapse_similarity: float
    recovery_similarity: float
    topology_alignment_score: float
    v2_resembles_recovery: bool


@dataclass(frozen=True)
class AudioStackHealthReport:
    """Frozen dependency and decode-path health report."""

    numpy_version: str
    scipy_version: str
    scipy_signal_available: bool
    scipy_io_wavfile_available: bool
    soundfile_available: bool
    soundfile_version: str
    audioread_available: bool
    audioread_version: str
    active_decode_path: str
    fallback_path: str
    pseudo_spectrum_ready: bool
    waveform_ready: bool
    operating_mode: str


# ---------------------------------------------------------------------------
# PSD cosine similarity
# ---------------------------------------------------------------------------

def _psd_similarity(
    samples_a: np.ndarray,
    sr_a: int,
    samples_b: np.ndarray,
    sr_b: int,
) -> float:
    """Cosine similarity between two Welch PSD vectors.

    When sample rates differ the PSDs are interpolated onto a shared
    frequency grid spanning the overlapping range so that each index
    corresponds to the same physical frequency.
    """
    freqs_a, psd_a = _welch_psd(samples_a, sr_a)
    freqs_b, psd_b = _welch_psd(samples_b, sr_b)

    if len(psd_a) == 0 or len(psd_b) == 0:
        return 0.0

    # Build shared frequency grid over the overlapping range
    f_min = max(float(freqs_a[0]), float(freqs_b[0]))
    f_max = min(float(freqs_a[-1]), float(freqs_b[-1]))
    if f_max <= f_min:
        return 0.0

    n_bins = min(len(freqs_a), len(freqs_b))
    shared_freqs = np.linspace(f_min, f_max, n_bins)

    # Linearly interpolate both PSDs onto the shared grid
    va = np.interp(shared_freqs, freqs_a, psd_a)
    vb = np.interp(shared_freqs, freqs_b, psd_b)

    norm_a = float(np.linalg.norm(va))
    norm_b = float(np.linalg.norm(vb))
    if norm_a == 0 or norm_b == 0:
        return 0.0

    return float(np.dot(va, vb) / (norm_a * norm_b))


# ---------------------------------------------------------------------------
# Internal: load samples for a file
# ---------------------------------------------------------------------------

def _load_samples(path: str) -> Tuple[np.ndarray, int]:
    """Load audio samples from a file (waveform or pseudo-spectrum).

    Attempts real waveform decode first; only reads raw bytes for
    pseudo-spectrum fallback to avoid redundant I/O.
    """
    decode_result = _try_decode_audio(path)
    if decode_result is not None:
        samples, sr, _ = decode_result
        return samples, sr

    # Fallback: read raw bytes for pseudo-spectrum
    p = Path(path)
    raw = p.read_bytes()
    mp3_meta = _parse_mp3_metadata(raw)

    if mp3_meta is not None:
        sr = mp3_meta.sample_rate
        offset = mp3_meta.audio_data_offset
    else:
        sr = _DEFAULT_SAMPLE_RATE
        offset = 0

    samples, sr = _compute_byte_pseudospectrum(raw, sr, offset)
    return samples, sr


# ---------------------------------------------------------------------------
# PART 1 — Legacy sequence regression
# ---------------------------------------------------------------------------

def analyze_legacy_audio_sequence(
    base_dir: str,
    baseline_json_path: Optional[str] = None,
) -> LegacyAudioSequenceReport:
    """Analyse the 5-state legacy audio sequence and compare to baseline.

    Parameters
    ----------
    base_dir : str
        Directory containing the 5 legacy QSOL MP3 files.
    baseline_json_path : str, optional
        Path to ``sequence_analysis.json``.  If None, only current
        spectral analysis is performed (no drift computation).

    Returns
    -------
    LegacyAudioSequenceReport
        Frozen, replay-safe sequence analysis with transition deltas.
    """
    # Analyse each state
    state_reports = []
    state_samples = []
    for fname in _LEGACY_STATE_FILES:
        fpath = str(Path(base_dir) / fname)
        report = analyze_quantum_audio_file(fpath)
        state_reports.append(report)
        samples, sr = _load_samples(fpath)
        state_samples.append((samples, sr))

    # Load legacy baseline if available
    legacy_transitions = None
    if baseline_json_path is not None:
        with open(baseline_json_path, "r") as f:
            legacy_data = json.load(f)
        legacy_transitions = legacy_data.get("transitions", [])

    # Compute transitions
    transition_reports = []
    centroid_drifts = []
    for i in range(len(state_reports) - 1):
        ra = state_reports[i]
        rb = state_reports[i + 1]

        centroid_delta = rb.spectral_centroid_hz - ra.spectral_centroid_hz
        entropy_delta = rb.spectral_entropy - ra.spectral_entropy
        hd_delta = rb.harmonic_density - ra.harmonic_density
        sub_delta = rb.subharmonic_energy_ratio - ra.subharmonic_energy_ratio
        coh_delta = rb.coherence_score - ra.coherence_score
        tight_a = _cluster_tightness(ra.cluster_points)
        tight_b = _cluster_tightness(rb.cluster_points)
        ct_delta = tight_b - tight_a

        sa, sra = state_samples[i]
        sb, srb = state_samples[i + 1]
        psd_sim = _psd_similarity(sa, sra, sb, srb)

        # Legacy baseline drift
        legacy_cd = 0.0
        cd_drift = 0.0
        if legacy_transitions is not None and i < len(legacy_transitions):
            lt = legacy_transitions[i]
            legacy_cd = lt["metrics"]["centroid_delta"]
            cd_drift = centroid_delta - legacy_cd

        centroid_drifts.append(abs(cd_drift))

        classification = _LEGACY_TRANSITION_CLASSES[i]

        transition_reports.append(LegacySequenceTransitionReport(
            from_state=_LEGACY_STATE_NAMES[i],
            to_state=_LEGACY_STATE_NAMES[i + 1],
            classification=classification,
            centroid_delta=round(centroid_delta, 6),
            entropy_delta=round(entropy_delta, 6),
            harmonic_density_delta=round(hd_delta, 8),
            subharmonic_delta=round(sub_delta, 8),
            coherence_delta=round(coh_delta, 6),
            cluster_tightness_delta=round(ct_delta, 8),
            psd_similarity=round(psd_sim, 6),
            legacy_centroid_delta=round(legacy_cd, 6),
            centroid_drift=round(cd_drift, 6),
        ))

    # Topology drift: mean absolute centroid drift from legacy
    topology_drift = sum(centroid_drifts) / len(centroid_drifts) if centroid_drifts else 0.0

    # Legacy similarity: average PSD similarity across transitions
    psd_sims = [t.psd_similarity for t in transition_reports]
    legacy_sim = sum(psd_sims) / len(psd_sims) if psd_sims else 0.0

    # Best recovery match: the state with highest coherence
    best_recovery = max(state_reports, key=lambda r: r.coherence_score)

    # Dependency health snapshot
    dep_health = _get_dependency_tuples()

    # Stability verdict
    coherences = [r.coherence_score for r in state_reports]
    mean_coh = sum(coherences) / len(coherences)
    if mean_coh >= 0.45 and topology_drift < 500.0:
        verdict = "stable_regression"
    elif mean_coh >= 0.25:
        verdict = "partial_regression"
    else:
        verdict = "degraded_regression"

    return LegacyAudioSequenceReport(
        n_states=len(state_reports),
        state_reports=tuple(state_reports),
        transition_reports=tuple(transition_reports),
        legacy_similarity_score=round(legacy_sim, 6),
        topology_drift_score=round(topology_drift, 6),
        best_recovery_match=best_recovery.filename,
        dependency_health=dep_health,
        stability_verdict=verdict,
    )


# ---------------------------------------------------------------------------
# PART 3 — Cross-generation comparison
# ---------------------------------------------------------------------------

def compare_legacy_vs_v1366(
    legacy_report: LegacyAudioSequenceReport,
    v1_path: str,
    v2_path: str,
) -> CrossEraComparisonReport:
    """Compare legacy 5-state chain against v136.6 artifacts.

    Parameters
    ----------
    legacy_report : LegacyAudioSequenceReport
        Result from ``analyze_legacy_audio_sequence``.
    v1_path : str
        Path to Quantum Coherence Threshold (v1).mp3
    v2_path : str
        Path to Quantum Coherence Threshold (v2).mp3

    Returns
    -------
    CrossEraComparisonReport
        Frozen cross-era comparison with topology alignment.
    """
    v1_report = analyze_quantum_audio_file(v1_path)
    v2_report = analyze_quantum_audio_file(v2_path)

    states = legacy_report.state_reports
    n_expected = len(_LEGACY_STATE_NAMES)
    if len(states) != n_expected:
        raise ValueError(
            f"legacy_report has {len(states)} states, expected {n_expected}"
        )

    # Named indices from the canonical state chain
    _COLLAPSE_IDX = _LEGACY_STATE_NAMES.index("Collapse")
    _RECOVERY_IDX = _LEGACY_STATE_NAMES.index("Recovery")

    # Find best legacy analog for v1 and v2 by coherence proximity
    def _best_analog(target: QuantumAudioLogicReport) -> str:
        best_name = states[0].filename
        best_dist = abs(target.coherence_score - states[0].coherence_score)
        for s in states[1:]:
            d = abs(target.coherence_score - s.coherence_score)
            if d < best_dist:
                best_dist = d
                best_name = s.filename
        return best_name

    best_v1 = _best_analog(v1_report)
    best_v2 = _best_analog(v2_report)

    # Collapse similarity: compare v1 against Collapse state
    collapse_state = states[_COLLAPSE_IDX]
    collapse_sim = 1.0 - min(1.0, abs(
        v1_report.coherence_score - collapse_state.coherence_score
    ))

    # Recovery similarity: compare v2 against Recovery state
    recovery_state = states[_RECOVERY_IDX]
    recovery_sim = 1.0 - min(1.0, abs(
        v2_report.coherence_score - recovery_state.coherence_score
    ))

    # Topology alignment: average mapping_2d distance across all states
    v2_x, v2_y = v2_report.mapping_2d
    dists = []
    for s in states:
        sx, sy = s.mapping_2d
        dists.append(math.sqrt((v2_x - sx) ** 2 + (v2_y - sy) ** 2))
    # Closest state alignment (lower distance = higher alignment)
    min_dist = min(dists) if dists else 1.0
    topology_alignment = round(max(0.0, 1.0 - min_dist), 6)

    # Does v2 resemble recovery/re-stabilization?
    v2_resembles = (
        recovery_sim > 0.5
        and v2_report.coherence_score >= recovery_state.coherence_score * 0.8
    )

    return CrossEraComparisonReport(
        best_legacy_analog_v1=best_v1,
        best_legacy_analog_v2=best_v2,
        collapse_similarity=round(collapse_sim, 6),
        recovery_similarity=round(recovery_sim, 6),
        topology_alignment_score=topology_alignment,
        v2_resembles_recovery=v2_resembles,
    )


# ---------------------------------------------------------------------------
# PART 4 — Dependency hardening
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _DependencyProbe:
    """Internal: cached result of probing all audio-stack dependencies."""

    numpy_version: str
    scipy_version: str
    scipy_signal_available: bool
    scipy_io_wavfile_available: bool
    soundfile_available: bool
    soundfile_version: str
    audioread_available: bool
    audioread_version: str


def _probe_dependencies() -> _DependencyProbe:
    """Probe the installed audio stack once (shared by public helpers)."""
    # numpy
    try:
        import numpy as _np
        np_ver = str(_np.__version__)
    except ImportError:
        np_ver = "not_installed"

    # scipy
    try:
        import scipy as _sp
        scipy_ver = str(_sp.__version__)
    except ImportError:
        scipy_ver = "not_installed"

    # scipy.signal
    try:
        from scipy import signal as _sig  # noqa: F401
        sig_avail = True
    except ImportError:
        sig_avail = False

    # scipy.io.wavfile
    try:
        from scipy.io import wavfile as _wf  # noqa: F401
        wavfile_avail = True
    except ImportError:
        wavfile_avail = False

    # soundfile
    sf_avail = False
    sf_ver = "not_installed"
    try:
        import soundfile as _sf
        sf_avail = True
        sf_ver = str(getattr(_sf, "__version__", "unknown"))
    except ImportError:
        pass

    # audioread
    ar_avail = False
    ar_ver = "not_installed"
    try:
        import audioread as _ar
        ar_avail = True
        ar_ver = str(getattr(_ar, "__version__", "unknown"))
    except ImportError:
        pass

    return _DependencyProbe(
        numpy_version=np_ver,
        scipy_version=scipy_ver,
        scipy_signal_available=sig_avail,
        scipy_io_wavfile_available=wavfile_avail,
        soundfile_available=sf_avail,
        soundfile_version=sf_ver,
        audioread_available=ar_avail,
        audioread_version=ar_ver,
    )


def _get_dependency_tuples() -> Tuple[Tuple[str, str], ...]:
    """Return sorted tuple of (package, version) for audio stack."""
    probe = _probe_dependencies()
    deps = [
        ("audioread", probe.audioread_version),
        ("numpy", probe.numpy_version),
        ("scipy", probe.scipy_version),
        ("soundfile", probe.soundfile_version),
    ]
    return tuple(sorted(deps))


def verify_audio_stack_health() -> AudioStackHealthReport:
    """Verify installed audio stack and report decode-path readiness.

    Returns
    -------
    AudioStackHealthReport
        Frozen report of dependency versions and decode capabilities.
    """
    probe = _probe_dependencies()

    # Determine active decode path
    waveform_ready = (
        probe.soundfile_available
        or probe.audioread_available
        or probe.scipy_io_wavfile_available
    )
    if probe.soundfile_available:
        active_path = "soundfile"
        fallback = "byte_pseudospectrum"
    elif probe.audioread_available:
        active_path = "audioread"
        fallback = "byte_pseudospectrum"
    elif probe.scipy_io_wavfile_available:
        active_path = "scipy.io.wavfile"
        fallback = "byte_pseudospectrum"
    else:
        active_path = "byte_pseudospectrum"
        fallback = "none"

    mode = "true_waveform_decode" if waveform_ready else "pseudo_spectrum"

    return AudioStackHealthReport(
        numpy_version=probe.numpy_version,
        scipy_version=probe.scipy_version,
        scipy_signal_available=probe.scipy_signal_available,
        scipy_io_wavfile_available=probe.scipy_io_wavfile_available,
        soundfile_available=probe.soundfile_available,
        soundfile_version=probe.soundfile_version,
        audioread_available=probe.audioread_available,
        audioread_version=probe.audioread_version,
        active_decode_path=active_path,
        fallback_path=fallback,
        pseudo_spectrum_ready=True,  # always available
        waveform_ready=waveform_ready,
        operating_mode=mode,
    )
