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

    # Scan for valid frames — require at least 3 consecutive valid frames
    # at the start to confirm we've found the audio stream
    scan_limit = min(len(data), offset + 8192)
    while offset < scan_limit:
        frame = _parse_mp3_frame_header(data, offset)
        if frame is not None:
            # Verify next frame follows immediately
            next_frame = _parse_mp3_frame_header(data, offset + frame.frame_length)
            if next_frame is not None:
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
            buf = b""
            for block in f:
                buf += block
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
        duration = len(samples) / sr
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

    # Use MP3 metadata for authoritative sample rate / duration when available
    if mp3_meta is not None:
        sr = mp3_meta.sample_rate
        duration = mp3_meta.duration_seconds

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
