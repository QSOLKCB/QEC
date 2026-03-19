"""
v74.1.0 — Deterministic pairwise sonic comparison and classification.

Compares two sonic analysis result dictionaries (as produced by
``sonic_analysis.analyse_file``) and computes spectral + structural deltas.
Classifies the relationship between the two signals.

Layer 5 — Experiments.
Does not modify decoder internals.  Fully deterministic.  Read-only on inputs.
"""

from __future__ import annotations

import copy
import math
from typing import Any, Dict, List


# ---------------------------------------------------------------------------
# Pairwise Feature Comparison
# ---------------------------------------------------------------------------

def compare_sonic_features(a: dict, b: dict) -> dict:
    """Compare two sonic analysis results and compute deltas.

    Parameters
    ----------
    a, b : dict
        Analysis result dictionaries from ``sonic_analysis.compute_features``
        or ``analyse_file``.  Must contain at minimum:
        ``rms_energy``, ``spectral_centroid_hz``, ``spectral_spread_hz``,
        ``zero_crossing_rate``, ``fft_top_peaks``.

    Returns
    -------
    dict
        Comparison metrics including deltas and FFT similarity score.
    """
    # Deep-copy to guarantee no mutation of caller data.
    a = copy.deepcopy(a)
    b = copy.deepcopy(b)

    energy_delta = b["rms_energy"] - a["rms_energy"]
    centroid_delta = b["spectral_centroid_hz"] - a["spectral_centroid_hz"]
    spread_delta = b["spectral_spread_hz"] - a["spectral_spread_hz"]
    zcr_delta = b["zero_crossing_rate"] - a["zero_crossing_rate"]
    fft_similarity = _fft_overlap_score(a["fft_top_peaks"], b["fft_top_peaks"])

    return {
        "energy_delta": float(energy_delta),
        "centroid_delta": float(centroid_delta),
        "spread_delta": float(spread_delta),
        "zcr_delta": float(zcr_delta),
        "fft_similarity": float(fft_similarity),
    }


# ---------------------------------------------------------------------------
# FFT Overlap Score
# ---------------------------------------------------------------------------

def _fft_overlap_score(
    peaks_a: List[Dict[str, float]],
    peaks_b: List[Dict[str, float]],
    *,
    tolerance_hz: float = 50.0,
) -> float:
    """Compute similarity ratio between two FFT peak profiles.

    For each peak in *a*, find the closest peak in *b* within
    ``tolerance_hz``.  The score is the fraction of peaks in *a*
    that have a match in *b*, weighted by relative magnitude overlap.

    Returns a value in [0.0, 1.0] where 1.0 means identical profiles.
    """
    if not peaks_a or not peaks_b:
        return 0.0

    # Normalise magnitudes within each set for relative comparison.
    max_a = max(p["magnitude"] for p in peaks_a) or 1.0
    max_b = max(p["magnitude"] for p in peaks_b) or 1.0

    n = len(peaks_a)
    total_score = 0.0

    for pa in peaks_a:
        norm_a = pa["magnitude"] / max_a

        best_overlap = 0.0
        for pb in peaks_b:
            if abs(pa["frequency_hz"] - pb["frequency_hz"]) <= tolerance_hz:
                norm_b = pb["magnitude"] / max_b
                # Similarity: 1 minus relative magnitude difference.
                mag_max = max(norm_a, norm_b)
                if mag_max > 0:
                    sim = 1.0 - abs(norm_a - norm_b) / mag_max
                else:
                    sim = 0.0
                best_overlap = max(best_overlap, sim)

        total_score += best_overlap

    return total_score / n


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

# Thresholds — simple deterministic logic, no ML.
_THRESHOLDS = {
    "energy_abs": 0.02,       # absolute energy delta for "minimal"
    "centroid_abs": 200.0,    # Hz
    "spread_abs": 150.0,      # Hz
    "zcr_abs": 0.02,
    "fft_sim_high": 0.7,
    "fft_sim_low": 0.3,
    "energy_drop": -0.03,     # collapse: energy drops significantly
    "energy_rise": 0.03,      # recovery: energy rises significantly
}


def classify_comparison(metrics: dict) -> str:
    """Classify the relationship between two compared signals.

    Parameters
    ----------
    metrics : dict
        Output of ``compare_sonic_features``.

    Returns
    -------
    str
        One of: ``"stable"``, ``"divergent"``, ``"transition"``,
        ``"collapse"``, ``"recovery"``.
    """
    t = _THRESHOLDS
    ed = metrics["energy_delta"]
    cd = abs(metrics["centroid_delta"])
    sd = abs(metrics["spread_delta"])
    zd = abs(metrics["zcr_delta"])
    fs = metrics["fft_similarity"]

    # Collapse: energy drops AND spectral simplification (spread narrows or
    # FFT similarity drops)
    if ed <= t["energy_drop"] and (sd > t["spread_abs"] or fs < t["fft_sim_low"]):
        return "collapse"

    # Recovery: energy rises AND structure returns (FFT similarity moderate+)
    if ed >= t["energy_rise"] and fs >= t["fft_sim_low"]:
        return "recovery"

    # Stable: all deltas small, FFT similar
    if (abs(ed) <= t["energy_abs"]
            and cd <= t["centroid_abs"]
            and sd <= t["spread_abs"]
            and zd <= t["zcr_abs"]
            and fs >= t["fft_sim_high"]):
        return "stable"

    # Transition: large structured change (multiple big deltas)
    big_count = sum([
        cd > t["centroid_abs"],
        sd > t["spread_abs"],
        zd > t["zcr_abs"],
        fs < t["fft_sim_high"],
    ])
    if big_count >= 2:
        return "transition"

    # Divergent: moderate change that doesn't fit above categories
    return "divergent"
