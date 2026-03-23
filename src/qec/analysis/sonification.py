"""Deterministic sonification of hierarchical correction & invariant dynamics (v96.1.0).

Converts hierarchical correction stages, invariant overlays, and stability
dynamics into reproducible audio signals via structured data-to-sound mapping.

This is NOT decorative audio.  Every parameter maps to a measurable property:
    - correction stage  → base frequency
    - projection distance → amplitude envelope (attack)
    - stability efficiency → sustain duration
    - active invariants → harmonic overtones

Layer 6 — Analysis.
Does not import from experiments or decoder.  Fully deterministic.
Uses only stdlib: wave, math, struct.  No external audio libraries.
"""

from __future__ import annotations

import math
import os
import struct
import wave
from typing import Any, Dict, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# PART 1 — AUDIO CONFIG
# ---------------------------------------------------------------------------

SAMPLE_RATE = 44100
BASE_FREQ = 220.0  # A3
DURATION_PER_STAGE = 0.25  # seconds per stage
AMPLITUDE = 0.3


# ---------------------------------------------------------------------------
# 1.2 — Deterministic frequency mapping: mode → frequency
# ---------------------------------------------------------------------------

_MODE_FREQ: Dict[str, float] = {
    "square": BASE_FREQ,
    "d4": BASE_FREQ * 4.0 / 3.0,
    "e8_like": BASE_FREQ * 3.0 / 2.0,
}


def map_mode_to_freq(mode: str) -> float:
    """Map a correction mode string to a deterministic base frequency.

    Single stages map to fixed frequencies.
    Multi-stage modes (e.g. "square>d4") sum their component frequencies.

    Args:
        mode: Stage name or multi-stage string separated by ">".

    Returns:
        Frequency in Hz.
    """
    stages = mode.split(">")
    total = 0.0
    for stage in stages:
        freq = _MODE_FREQ.get(stage)
        if freq is None:
            raise ValueError(f"unknown correction stage: {stage!r}")
        total += freq
    return total


# ---------------------------------------------------------------------------
# PART 2 — STAGE SONIFICATION
# ---------------------------------------------------------------------------

# 2.1 — Generate sine wave

def generate_tone(freq: float, duration: float, amplitude: float) -> List[float]:
    """Generate a pure sine tone as a list of float samples in [-1, 1].

    Args:
        freq: Frequency in Hz.
        duration: Duration in seconds.
        amplitude: Peak amplitude in [0, 1].

    Returns:
        List of float samples.
    """
    n_samples = int(SAMPLE_RATE * duration)
    samples: List[float] = []
    for i in range(n_samples):
        t = i / SAMPLE_RATE
        value = amplitude * math.sin(2.0 * math.pi * freq * t)
        samples.append(value)
    return samples


# 2.2 — Map projection distance → amplitude envelope

def amplitude_from_projection(dist: float) -> float:
    """Map projection distance to attack amplitude.

    Larger projection → louder attack.
    Clamped deterministically to [0.05, 1.0].

    Args:
        dist: Projection distance (non-negative).

    Returns:
        Amplitude scaling factor.
    """
    # Sigmoid-like mapping: tanh maps [0, inf) → [0, 1)
    raw = math.tanh(dist)
    # Scale to [0.05, 1.0] so silence is never total.
    return 0.05 + 0.95 * raw


# 2.3 — Stability → sustain

def sustain_from_stability(stability_eff: float) -> float:
    """Map stability efficiency to sustain duration multiplier.

    High stability → longer sustain (up to 2x base duration).
    Low stability  → quick decay (down to 0.3x base duration).

    Args:
        stability_eff: Stability efficiency in [0, 1].

    Returns:
        Duration multiplier.
    """
    clamped = max(0.0, min(1.0, stability_eff))
    return 0.3 + 1.7 * clamped


# ---------------------------------------------------------------------------
# PART 3 — INVARIANT SONIFICATION
# ---------------------------------------------------------------------------

# 3.1 — Invariant → harmonic ratio

INVARIANT_HARMONICS: Dict[str, float] = {
    "local_stability_constraint": 1.25,
    "equivalence_class_constraint": 1.5,
    "geometry_alignment_constraint": 1.333,
    "explicit_allowed_state_constraint": 1.2,
    "bounded_projection_constraint": 1.75,
}


# 3.2 — Apply harmonics

def apply_invariant_harmonics(
    base_freq: float,
    invariants: Sequence[str],
) -> List[float]:
    """Compute harmonic frequency set from base tone + active invariants.

    Args:
        base_freq: Base frequency in Hz.
        invariants: List of active invariant names.

    Returns:
        List of frequencies: [base_freq, harmonic1, harmonic2, ...].
    """
    freqs = [base_freq]
    for inv_name in invariants:
        ratio = INVARIANT_HARMONICS.get(inv_name)
        if ratio is not None:
            freqs.append(base_freq * ratio)
    return freqs


# ---------------------------------------------------------------------------
# PART 4 — HIERARCHICAL AUDIO CONSTRUCTION
# ---------------------------------------------------------------------------

def _mix_samples(
    base: List[float],
    overlay: List[float],
    overlay_amp: float,
) -> List[float]:
    """Mix an overlay signal into a base signal (additive).

    If overlay is shorter, it is zero-padded.
    If overlay is longer, it is truncated.
    """
    n = len(base)
    result: List[float] = []
    for i in range(n):
        b = base[i]
        o = overlay[i] * overlay_amp if i < len(overlay) else 0.0
        result.append(b + o)
    return result


def _apply_sustain_envelope(
    samples: List[float],
    sustain_mult: float,
) -> List[float]:
    """Apply a sustain/decay envelope to a sample buffer.

    The tone sustains for sustain_mult fraction of its length,
    then decays linearly to zero over the remainder.
    """
    n = len(samples)
    if n == 0:
        return []
    sustain_end = int(n * min(sustain_mult / 2.0, 1.0))
    result: List[float] = []
    for i in range(n):
        if i < sustain_end:
            result.append(samples[i])
        else:
            # Linear decay from sustain_end to end.
            remaining = n - sustain_end
            if remaining > 0:
                decay = 1.0 - (i - sustain_end) / remaining
            else:
                decay = 0.0
            result.append(samples[i] * max(decay, 0.0))
    return result


# 4.1 — Convert stages to audio

def sonify_stages(
    stages: List[str],
    projection_distances: List[float],
    stability: float,
    invariants: List[str],
) -> List[float]:
    """Convert hierarchical correction stages to a sequential waveform.

    For each stage:
      - determine base frequency from mode
      - determine amplitude from projection distance
      - determine sustain from stability efficiency
      - apply invariant harmonics as additive overtones
      - append sequentially to output

    Args:
        stages: Ordered list of correction stage names.
        projection_distances: Per-stage projection distances.
        stability: Stability efficiency in [0, 1].
        invariants: List of active invariant names.

    Returns:
        Raw sample list (mono, float in approximately [-1, 1]).
    """
    if len(projection_distances) < len(stages):
        # Pad with zeros if fewer distances than stages.
        projection_distances = list(projection_distances) + [
            0.0
        ] * (len(stages) - len(projection_distances))

    sustain_mult = sustain_from_stability(stability)
    all_samples: List[float] = []

    for i, stage in enumerate(stages):
        base_freq = _MODE_FREQ.get(stage)
        if base_freq is None:
            raise ValueError(f"unknown correction stage: {stage!r}")

        dist = projection_distances[i]
        amp = amplitude_from_projection(dist) * AMPLITUDE
        duration = DURATION_PER_STAGE * sustain_mult

        # Generate base tone.
        tone = generate_tone(base_freq, duration, amp)

        # Apply invariant harmonics as quieter overtones.
        harmonic_freqs = apply_invariant_harmonics(base_freq, invariants)
        for h_freq in harmonic_freqs[1:]:  # skip base (already generated)
            harmonic_tone = generate_tone(h_freq, duration, amp * 0.3)
            tone = _mix_samples(tone, harmonic_tone, 1.0)

        # Apply sustain envelope.
        tone = _apply_sustain_envelope(tone, sustain_mult)

        all_samples.extend(tone)

    return all_samples


# 4.2 — Normalize waveform

def _normalize(samples: List[float]) -> List[float]:
    """Normalize sample list to [-1, 1]."""
    if not samples:
        return []
    peak = max(abs(s) for s in samples)
    if peak > 0.0:
        return [s / peak for s in samples]
    return list(samples)


# ---------------------------------------------------------------------------
# PART 5 — WAV EXPORT
# ---------------------------------------------------------------------------

def write_wav(filename: str, samples: List[float]) -> str:
    """Write a mono 16-bit WAV file from float samples in [-1, 1].

    Args:
        filename: Output file path.
        samples: Float samples in [-1, 1].

    Returns:
        Absolute path to written file.
    """
    os.makedirs(os.path.dirname(os.path.abspath(filename)) or ".", exist_ok=True)

    with wave.open(filename, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(SAMPLE_RATE)

        frame_data = b""
        for s in samples:
            clamped = max(-1.0, min(1.0, s))
            pcm_val = int(clamped * 32767.0)
            frame_data += struct.pack("<h", pcm_val)

        wf.writeframes(frame_data)

    return os.path.abspath(filename)


# ---------------------------------------------------------------------------
# PART 6 — FULL PIPELINE
# ---------------------------------------------------------------------------

def run_sonification(
    data: List[Dict[str, Any]],
    filename_prefix: str = "qec",
    output_dir: str = ".",
) -> List[Dict[str, Any]]:
    """Entry point: sonify hierarchical correction results.

    Expects a list of result dicts from hierarchical correction runs.
    Each dict should contain:
      - "mode": str (e.g. "square>d4>e8_like")
      - "stages": list of str
      - "projection_distances": list of float
      - "stability_efficiency": float
      - invariants (optional): list of invariant name strings

    For each system, generates a waveform and saves a .wav file.

    Args:
        data: List of hierarchical correction result dicts.
        filename_prefix: Prefix for output filenames.
        output_dir: Directory for output files.

    Returns:
        List of report dicts with dfa_type, n, file, duration.
    """
    reports: List[Dict[str, Any]] = []

    for entry in data:
        mode = entry.get("mode", "square")
        stages = entry.get("stages", mode.split(">"))
        projection_distances = entry.get("projection_distances", [])
        stability = entry.get("stability_efficiency", 0.5)
        invariants = entry.get("invariants", [])
        dfa_type = entry.get("dfa_type", "unknown")
        n = entry.get("n", 0)

        # Generate waveform.
        raw_samples = sonify_stages(
            stages, projection_distances, stability, invariants,
        )
        samples = _normalize(raw_samples)

        # Filename.
        safe_type = dfa_type.replace(" ", "_")
        fname = f"{filename_prefix}_{safe_type}_{n}.wav"
        fpath = os.path.join(output_dir, fname)

        write_wav(fpath, samples)

        duration = len(samples) / SAMPLE_RATE
        reports.append({
            "dfa_type": dfa_type,
            "n": n,
            "file": fname,
            "duration": round(duration, 4),
        })

    return reports


# ---------------------------------------------------------------------------
# PART 7 — COMPARISON MODE (A/B stereo)
# ---------------------------------------------------------------------------

def sonify_comparison(
    before: Dict[str, Any],
    after: Dict[str, Any],
    filename: str = "comparison.wav",
) -> str:
    """Generate a stereo WAV comparing before (left) and after (right).

    Left channel:  baseline (before hierarchical + invariants)
    Right channel: hierarchical + invariants

    Both channels are deterministic.

    Args:
        before: Correction result dict (baseline).
        after: Correction result dict (hierarchical + invariants).
        filename: Output file path.

    Returns:
        Absolute path to written file.
    """
    def _make_samples(entry: Dict[str, Any]) -> List[float]:
        mode = entry.get("mode", "square")
        stages = entry.get("stages", mode.split(">"))
        projection_distances = entry.get("projection_distances", [])
        stability = entry.get("stability_efficiency", 0.5)
        invariants = entry.get("invariants", [])
        raw = sonify_stages(stages, projection_distances, stability, invariants)
        return _normalize(raw)

    left = _make_samples(before)
    right = _make_samples(after)

    # Pad shorter channel to match length.
    max_len = max(len(left), len(right))
    left.extend([0.0] * (max_len - len(left)))
    right.extend([0.0] * (max_len - len(right)))

    os.makedirs(os.path.dirname(os.path.abspath(filename)) or ".", exist_ok=True)

    with wave.open(filename, "w") as wf:
        wf.setnchannels(2)  # stereo
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)

        frame_data = b""
        for i in range(max_len):
            l_val = max(-1.0, min(1.0, left[i]))
            r_val = max(-1.0, min(1.0, right[i]))
            frame_data += struct.pack("<h", int(l_val * 32767.0))
            frame_data += struct.pack("<h", int(r_val * 32767.0))

        wf.writeframes(frame_data)

    return os.path.abspath(filename)


# ---------------------------------------------------------------------------
# PART 8 — PRINT LAYER
# ---------------------------------------------------------------------------

def print_sonification_summary(report: List[Dict[str, Any]]) -> str:
    """Format sonification report as human-readable text.

    Args:
        report: List of dicts from run_sonification.

    Returns:
        Deterministic text summary.
    """
    lines: List[str] = []
    lines.append("=== Sonification Summary ===")
    lines.append("")

    for entry in report:
        dfa_type = entry.get("dfa_type", "unknown")
        n = entry.get("n", 0)
        fname = entry.get("file", "")
        duration = entry.get("duration", 0.0)
        stages = entry.get("stages", "")
        invariants = entry.get("invariants", "")

        lines.append(f"DFA: {dfa_type} (n={n})")
        lines.append(f"  file: {fname}")
        lines.append(f"  duration: {duration}s")
        if stages:
            lines.append(f"  stages: {stages}")
        if invariants:
            lines.append(f"  invariants: {invariants}")
        lines.append("")

    return "\n".join(lines)
