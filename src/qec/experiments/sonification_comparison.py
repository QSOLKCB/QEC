"""Comparative sonification experiment (v72.3.0).

Runs baseline and multidimensional sonification on identical inputs
and computes quantitative comparison metrics. Produces reproducible
JSON artifacts and optional WAV files.

Invariants:
    - Deterministic: same input -> identical metrics
    - No mutation of input dict
    - No hidden dependencies (numpy + stdlib only)
    - Does not modify sonification modules
"""

from __future__ import annotations

import copy
import json
import os
import struct
import wave
from typing import Optional

import numpy as np

from qec.modules.sonification.e8_baseline import sonify_e8_baseline, SAMPLE_RATE, DURATION
from qec.modules.sonification.e8_multidim import sonify_e8_multidim


def _build_invariant_mask(invariants: list, num_samples: int) -> np.ndarray:
    """Build boolean mask matching the silence gate logic in sonification modules."""
    t = np.linspace(0.0, DURATION, num_samples, endpoint=False, dtype=np.float64)
    mask = np.zeros(num_samples, dtype=bool)
    for start, end in invariants:
        mask |= (t >= float(start)) & (t <= float(end))
    return mask


def _compute_silence_fidelity(signal_int16: np.ndarray, mask: np.ndarray) -> float:
    """Fraction of samples inside invariant windows that are exactly zero."""
    total = int(np.sum(mask))
    if total == 0:
        return 1.0
    masked = signal_int16[mask]
    zero_count = int(np.sum(masked == 0))
    return float(zero_count) / float(total)


def _compute_rms(signal: np.ndarray) -> float:
    """RMS energy of a 1-D signal (as float64)."""
    s = signal.astype(np.float64)
    return float(np.sqrt(np.mean(s ** 2)))


def _compute_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation between two 1-D arrays."""
    a_f = a.astype(np.float64)
    b_f = b.astype(np.float64)
    if np.std(a_f) == 0.0 or np.std(b_f) == 0.0:
        return 0.0
    corr_matrix = np.corrcoef(a_f, b_f)
    return float(corr_matrix[0, 1])


def _write_wav_mono(filename: str, audio_int16: np.ndarray) -> None:
    """Write mono int16 WAV using stdlib only."""
    with wave.open(filename, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        for sample in audio_int16:
            wf.writeframes(struct.pack("<h", int(sample)))


def _write_wav_stereo(filename: str, audio_int16: np.ndarray) -> None:
    """Write stereo int16 WAV using stdlib only."""
    with wave.open(filename, "w") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        for frame in audio_int16:
            wf.writeframes(struct.pack("<hh", int(frame[0]), int(frame[1])))


def run_sonification_comparison(
    result: dict,
    output_dir: Optional[str] = None,
) -> dict:
    """Run baseline and multidimensional sonification on the same input
    and compute comparison metrics.

    Args:
        result: Dict with keys columns, errorRate, complexity, invariants.
        output_dir: Optional directory for WAV and JSON artifacts.

    Returns:
        Dict containing comparison metrics and metadata.
    """
    # Deep copy to guarantee no mutation of caller's dict
    data = copy.deepcopy(result)

    # Step 1 — Generate signals
    baseline = sonify_e8_baseline(data)
    multidim = sonify_e8_multidim(data)

    num_samples = len(baseline)
    invariants = [(float(s), float(e)) for s, e in data["invariants"]]
    inv_mask = _build_invariant_mask(invariants, num_samples)

    # Step 2 — Align signals (baseline mono -> stereo for comparison)
    baseline_stereo = np.stack([baseline, baseline], axis=1)

    # Step 3 — Compute metrics

    # 3.1 Invariant silence fidelity
    baseline_silence = _compute_silence_fidelity(baseline, inv_mask)
    # Multidim: check both channels
    multidim_ch0_silence = _compute_silence_fidelity(multidim[:, 0], inv_mask)
    multidim_ch1_silence = _compute_silence_fidelity(multidim[:, 1], inv_mask)
    multidim_silence = min(multidim_ch0_silence, multidim_ch1_silence)

    # 3.2 Cross-channel leakage (multidim only)
    total_mask_samples = int(np.sum(inv_mask))
    if total_mask_samples == 0:
        leakage_rate = 0.0
    else:
        masked_multidim = multidim[inv_mask]
        nonzero_count = int(np.sum(masked_multidim != 0))
        leakage_rate = float(nonzero_count) / float(total_mask_samples * 2)

    # 3.3 Structural energy ratio (RMS)
    baseline_energy = _compute_rms(baseline)
    multidim_ch0_energy = _compute_rms(multidim[:, 0])
    multidim_ch1_energy = _compute_rms(multidim[:, 1])

    # 3.4 Channel separation (multidim correlation)
    channel_correlation = _compute_correlation(multidim[:, 0], multidim[:, 1])

    # 3.5 Spectral spread proxy (variance)
    baseline_variance = float(np.var(baseline.astype(np.float64)))
    multidim_variance = float(np.var(multidim.astype(np.float64)))

    metrics = {
        "version": "v72.3.0",
        "baseline_silence_fidelity": baseline_silence,
        "multidim_silence_fidelity": multidim_silence,
        "multidim_ch0_silence_fidelity": multidim_ch0_silence,
        "multidim_ch1_silence_fidelity": multidim_ch1_silence,
        "leakage_rate": leakage_rate,
        "baseline_energy": baseline_energy,
        "multidim_ch0_energy": multidim_ch0_energy,
        "multidim_ch1_energy": multidim_ch1_energy,
        "channel_correlation": channel_correlation,
        "baseline_variance": baseline_variance,
        "multidim_variance": multidim_variance,
        "num_samples": num_samples,
        "num_invariant_samples": total_mask_samples,
        "input_summary": {
            "n_columns": len(data["columns"]),
            "error_rate": float(data["errorRate"]),
            "complexity": float(data["complexity"]),
            "n_invariants": len(data["invariants"]),
        },
    }

    # Optional artifact outputs
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        _write_wav_mono(
            os.path.join(output_dir, "baseline.wav"), baseline
        )
        _write_wav_stereo(
            os.path.join(output_dir, "multidim.wav"), multidim
        )
        json_path = os.path.join(output_dir, "comparison.json")
        with open(json_path, "w") as f:
            json.dump(metrics, f, indent=2, sort_keys=True)

    return metrics
