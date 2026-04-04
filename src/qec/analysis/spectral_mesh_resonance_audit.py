"""v137.0.15 — Spectral Mesh Resonance Audit.

Theory-coupled spectral audit absorbing real semantics from:

  Deterministic Traversal Mesh.mp3
  E8 Triality Topology.mp3

Core theory constructs absorbed:

  1. PHI_LOCK — phi-ratio dominance in spectral peak adjacency
     Adjacent dominant peak magnitude ratios are compared against the
     golden ratio PHI = 1.618.  Bounded [0, 1].

  2. E8_TRIALITY_SPECTRAL_MESH — triality recurrence via mod-5 grouping
     Peak indices grouped by (index % 5) measure uniformity of the
     five triality axis classes.  Bounded [0, 1].

  3. OUROBOROS_AUDIO_LOOPBACK — cyclic self-similarity between early
     and late spectral windows.  Cosine similarity of first-half vs
     second-half peak magnitudes.  Bounded [0, 1].

  4. SID-style lightweight metrics — spectral_instability_score,
     spectral_drift_score, attractor_lock_score derived from peak
     spread, variance, and recurrence.  Each bounded [0, 1].

Pipeline law:

  raw file bytes
  -> deterministic numeric vector
  -> FFT magnitude spectrum
  -> peak extraction
  -> theory-coupled audit scores
  -> canonical JSON + SHA-256 ledger

Layer 4 -- Analysis.
Does not import or modify decoder internals.
Fully deterministic: no randomness, no global state, no input mutation.

Theory Upgrade Source:
- file: Deterministic Traversal Mesh.mp3, E8 Triality Topology.mp3
- concept: phi-lock spectral dominance, E8 triality recurrence, ouroboros loopback
- implementation: spectral peak audit with phi/triality/loopback scores
- invariant tested: PHI_LOCK, E8_TRIALITY_SPECTRAL_MESH, OUROBOROS_AUDIO_LOOPBACK
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Version
# ---------------------------------------------------------------------------

SPECTRAL_MESH_RESONANCE_VERSION: str = "v137.0.15"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PHI: float = 1.618033988749895
_BLOCK_SIZE: int = 4096  # FFT window size
_TOP_K_PEAKS: int = 32   # number of dominant peaks to extract


# ---------------------------------------------------------------------------
# Frozen dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SpectralResonancePeak:
    """A single spectral peak extracted from raw file bytes."""
    peak_index: int
    frequency_ratio: float
    magnitude: float
    stable_hash: str
    version: str = SPECTRAL_MESH_RESONANCE_VERSION


@dataclass(frozen=True)
class SpectralMeshAuditDecision:
    """Theory-coupled audit decision for one source file."""
    source_name: str
    peak_count: int
    phi_lock_score: float
    triality_recurrence_score: float
    loopback_cycle_score: float
    spectral_instability_score: float
    spectral_drift_score: float
    attractor_lock_score: float
    symbolic_trace: str
    stable_hash: str
    version: str = SPECTRAL_MESH_RESONANCE_VERSION


@dataclass(frozen=True)
class SpectralMeshAuditLedger:
    """Ledger aggregating multiple audit decisions."""
    decisions: Tuple[SpectralMeshAuditDecision, ...]
    decision_count: int
    stable_hash: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _stable_hash_dict(d: Dict[str, Any]) -> str:
    """SHA-256 of canonical JSON."""
    raw = json.dumps(d, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _bytes_to_float_vector(data: bytes) -> np.ndarray:
    """Convert raw bytes to a deterministic float64 vector.

    Each byte becomes a float64 value in [0, 255].  This is intentionally
    simple — the goal is deterministic invariant audit, not audio fidelity.
    """
    return np.frombuffer(data, dtype=np.uint8).astype(np.float64)


def _compute_fft_magnitudes(vec: np.ndarray, block_size: int) -> np.ndarray:
    """Compute averaged FFT magnitudes over non-overlapping windows."""
    n = len(vec)
    if n == 0:
        return np.zeros(block_size // 2 + 1, dtype=np.float64)
    if n < block_size:
        padded = np.zeros(block_size, dtype=np.float64)
        padded[:n] = vec
        trimmed = padded.reshape(1, block_size)
    else:
        n_blocks = n // block_size
        trimmed = vec[: n_blocks * block_size].reshape(n_blocks, block_size)
    # Real FFT per block — take only positive frequencies
    spectra = np.abs(np.fft.rfft(trimmed, axis=1))
    # Average across blocks for stable spectrum
    return np.mean(spectra, axis=0)


def _find_top_peaks(magnitudes: np.ndarray, top_k: int) -> np.ndarray:
    """Return indices of top-k magnitude peaks, sorted by index."""
    if len(magnitudes) == 0 or top_k <= 0:
        return np.array([], dtype=np.int64)
    k = min(top_k, len(magnitudes))
    # Stable tie-breaking: lexsort by (index ASC, magnitude ASC)
    # so equal magnitudes resolve by lowest index first
    ordered = np.lexsort((np.arange(len(magnitudes)), magnitudes))
    indices = ordered[-k:]
    return np.sort(indices)


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def extract_spectral_peaks(
    audio_path: str,
) -> Tuple[SpectralResonancePeak, ...]:
    """Extract spectral peaks from raw file bytes.

    Reads the file as raw bytes, converts to float vector, computes
    windowed FFT magnitudes, and returns the top-k peaks as frozen
    dataclasses.
    """
    with open(audio_path, "rb") as f:
        data = f.read()

    vec = _bytes_to_float_vector(data)
    magnitudes = _compute_fft_magnitudes(vec, _BLOCK_SIZE)
    peak_indices = _find_top_peaks(magnitudes, _TOP_K_PEAKS)

    max_mag = float(np.max(magnitudes)) if len(magnitudes) > 0 else 1.0
    if max_mag == 0.0:
        max_mag = 1.0

    nyquist = len(magnitudes) - 1 if len(magnitudes) > 1 else 1

    peaks = []
    for idx in peak_indices:
        idx_int = int(idx)
        mag = float(magnitudes[idx_int])
        freq_ratio = float(idx_int) / float(nyquist) if nyquist > 0 else 0.0
        peak_dict = {
            "peak_index": idx_int,
            "frequency_ratio": freq_ratio,
            "magnitude": mag,
            "version": SPECTRAL_MESH_RESONANCE_VERSION,
            "block_size": _BLOCK_SIZE,
            "top_k": _TOP_K_PEAKS,
        }
        h = _stable_hash_dict(peak_dict)
        peaks.append(SpectralResonancePeak(
            peak_index=idx_int,
            frequency_ratio=freq_ratio,
            magnitude=mag,
            stable_hash=h,
        ))

    return tuple(peaks)


def _compute_phi_lock_score(peaks: Tuple[SpectralResonancePeak, ...]) -> float:
    """PHI_LOCK: compare adjacent peak magnitude ratios to golden ratio.

    For each pair of adjacent peaks (sorted by index), compute the ratio
    of the larger magnitude to the smaller.  Measure closeness to PHI.
    Return bounded [0, 1] where 1.0 = perfect phi-lock.
    """
    if len(peaks) < 2:
        return 0.0

    scores = []
    for i in range(len(peaks) - 1):
        a = peaks[i].magnitude
        b = peaks[i + 1].magnitude
        hi = max(abs(a), abs(b))
        lo = min(abs(a), abs(b))
        ratio = hi / lo if lo > 1e-12 else 0.0
        # Closeness to PHI: 1 - |ratio - PHI| / PHI, clamped
        closeness = 1.0 - abs(ratio - PHI) / PHI
        scores.append(max(0.0, min(1.0, closeness)))

    return float(np.mean(scores))


def _compute_triality_recurrence_score(
    peaks: Tuple[SpectralResonancePeak, ...],
) -> float:
    """E8_TRIALITY_SPECTRAL_MESH: triality via mod-5 grouping.

    Group peak indices by (index % 5).  Measure how uniformly peaks
    distribute across the 5 triality axis classes.  Perfect uniformity
    scores 1.0.
    """
    if len(peaks) == 0:
        return 0.0

    counts = [0] * 5
    for p in peaks:
        counts[p.peak_index % 5] += 1

    n = len(peaks)
    expected = n / 5.0
    # Normalized deviation from uniform
    max_dev = expected * 4.0  # worst case: all in one bin
    if max_dev < 1e-12:
        return 1.0
    actual_dev = sum(abs(c - expected) for c in counts)
    score = 1.0 - actual_dev / max_dev
    return max(0.0, min(1.0, score))


def _compute_loopback_cycle_score(
    peaks: Tuple[SpectralResonancePeak, ...],
) -> float:
    """OUROBOROS_AUDIO_LOOPBACK: cyclic self-similarity.

    Compare first-half peak magnitudes with second-half peak magnitudes
    via cosine similarity.  High similarity = strong ouroboros loopback.
    """
    if len(peaks) < 4:
        return 0.0

    mags = np.array([p.magnitude for p in peaks], dtype=np.float64)
    mid = len(mags) // 2
    first_half = mags[:mid]
    second_half = mags[mid: mid + len(first_half)]

    if len(second_half) == 0:
        return 0.0

    dot = float(np.dot(first_half, second_half))
    norm_a = float(np.linalg.norm(first_half))
    norm_b = float(np.linalg.norm(second_half))

    if norm_a < 1e-12 or norm_b < 1e-12:
        return 0.0

    cosine = dot / (norm_a * norm_b)
    # Map [-1, 1] to [0, 1]
    return max(0.0, min(1.0, (cosine + 1.0) / 2.0))


def _compute_spectral_instability_score(
    peaks: Tuple[SpectralResonancePeak, ...],
) -> float:
    """SID-style: instability from peak magnitude spread."""
    if len(peaks) < 2:
        return 0.0
    mags = np.array([p.magnitude for p in peaks], dtype=np.float64)
    spread = float(np.max(mags) - np.min(mags))
    mean_mag = float(np.mean(mags))
    if mean_mag < 1e-12:
        return 0.0
    # Normalized spread as instability proxy
    raw = spread / mean_mag
    return max(0.0, min(1.0, raw / 4.0))


def _compute_spectral_drift_score(
    peaks: Tuple[SpectralResonancePeak, ...],
) -> float:
    """SID-style: drift from frequency-ratio variance."""
    if len(peaks) < 2:
        return 0.0
    ratios = np.array([p.frequency_ratio for p in peaks], dtype=np.float64)
    var = float(np.var(ratios))
    # Normalize: variance of uniform [0,1] is ~0.083
    return max(0.0, min(1.0, var / 0.1))


def _compute_attractor_lock_score(
    peaks: Tuple[SpectralResonancePeak, ...],
) -> float:
    """SID-style: attractor lock from peak index recurrence."""
    if len(peaks) < 2:
        return 0.0
    indices = np.array([p.peak_index for p in peaks], dtype=np.float64)
    diffs = np.diff(indices)
    if len(diffs) == 0:
        return 0.0
    mean_diff = float(np.mean(diffs))
    if mean_diff < 1e-12:
        return 1.0
    std_diff = float(np.std(diffs))
    # Low variance in spacing = strong attractor lock
    cv = std_diff / mean_diff  # coefficient of variation
    return max(0.0, min(1.0, 1.0 - cv / 2.0))


def build_spectral_mesh_audit(
    audio_path: str,
) -> SpectralMeshAuditDecision:
    """Build a complete spectral mesh audit decision for one file."""
    import os
    source_name = os.path.basename(audio_path)
    peaks = extract_spectral_peaks(audio_path)

    phi_lock = _compute_phi_lock_score(peaks)
    triality = _compute_triality_recurrence_score(peaks)
    loopback = _compute_loopback_cycle_score(peaks)
    instability = _compute_spectral_instability_score(peaks)
    drift = _compute_spectral_drift_score(peaks)
    attractor = _compute_attractor_lock_score(peaks)

    symbolic_trace = (
        f"PHI_LOCK={phi_lock:.6f}|"
        f"E8_TRIALITY={triality:.6f}|"
        f"OUROBOROS={loopback:.6f}|"
        f"SID_INST={instability:.6f}|"
        f"SID_DRIFT={drift:.6f}|"
        f"SID_ATTR={attractor:.6f}"
    )

    decision_dict = {
        "source_name": source_name,
        "peak_count": len(peaks),
        "phi_lock_score": phi_lock,
        "triality_recurrence_score": triality,
        "loopback_cycle_score": loopback,
        "spectral_instability_score": instability,
        "spectral_drift_score": drift,
        "attractor_lock_score": attractor,
        "symbolic_trace": symbolic_trace,
        "version": SPECTRAL_MESH_RESONANCE_VERSION,
    }
    h = _stable_hash_dict(decision_dict)

    return SpectralMeshAuditDecision(
        source_name=source_name,
        peak_count=len(peaks),
        phi_lock_score=phi_lock,
        triality_recurrence_score=triality,
        loopback_cycle_score=loopback,
        spectral_instability_score=instability,
        spectral_drift_score=drift,
        attractor_lock_score=attractor,
        symbolic_trace=symbolic_trace,
        stable_hash=h,
    )


def build_spectral_mesh_audit_ledger(
    decisions: Tuple[SpectralMeshAuditDecision, ...],
) -> SpectralMeshAuditLedger:
    """Aggregate multiple audit decisions into a ledger."""
    ledger_dict = {
        "decision_hashes": [d.stable_hash for d in decisions],
        "decision_count": len(decisions),
    }
    h = _stable_hash_dict(ledger_dict)
    return SpectralMeshAuditLedger(
        decisions=decisions,
        decision_count=len(decisions),
        stable_hash=h,
    )


def export_spectral_mesh_audit_bundle(
    decision: SpectralMeshAuditDecision,
) -> Dict[str, Any]:
    """Export a single audit decision as canonical JSON-safe dict."""
    return {
        "source_name": decision.source_name,
        "peak_count": decision.peak_count,
        "phi_lock_score": decision.phi_lock_score,
        "triality_recurrence_score": decision.triality_recurrence_score,
        "loopback_cycle_score": decision.loopback_cycle_score,
        "spectral_instability_score": decision.spectral_instability_score,
        "spectral_drift_score": decision.spectral_drift_score,
        "attractor_lock_score": decision.attractor_lock_score,
        "symbolic_trace": decision.symbolic_trace,
        "stable_hash": decision.stable_hash,
        "version": decision.version,
    }


def export_spectral_mesh_audit_ledger(
    ledger: SpectralMeshAuditLedger,
) -> Dict[str, Any]:
    """Export the full ledger as canonical JSON-safe dict."""
    return {
        "decisions": [
            export_spectral_mesh_audit_bundle(d) for d in ledger.decisions
        ],
        "decision_count": ledger.decision_count,
        "stable_hash": ledger.stable_hash,
    }
