"""
v74.2.0 — Deterministic pairwise sonic comparison and directional classification.

Compares two sonic analysis result dictionaries (as produced by
``sonic_analysis.analyse_file``) and computes spectral + structural deltas
plus derived ratio metrics.  Classifies the *nature* of the change between
two signals using directional rules.

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

    # Derived ratio metrics (v74.2.0).
    energy_ratio = _safe_ratio(b["rms_energy"], a["rms_energy"])
    spread_ratio = _safe_ratio(b["spectral_spread_hz"], a["spectral_spread_hz"])
    fft_peak_count_change = (
        _count_significant_peaks(b["fft_top_peaks"])
        - _count_significant_peaks(a["fft_top_peaks"])
    )

    return {
        "energy_delta": float(energy_delta),
        "centroid_delta": float(centroid_delta),
        "spread_delta": float(spread_delta),
        "zcr_delta": float(zcr_delta),
        "fft_similarity": float(fft_similarity),
        "energy_ratio": float(energy_ratio),
        "spread_ratio": float(spread_ratio),
        "fft_peak_count_change": int(fft_peak_count_change),
    }


# ---------------------------------------------------------------------------
# Derived Metric Helpers
# ---------------------------------------------------------------------------

def _safe_ratio(numerator: float, denominator: float) -> float:
    """Return *numerator / denominator*, guarding against zero."""
    if denominator == 0.0:
        if numerator == 0.0:
            return 1.0
        return float("inf") if numerator > 0 else float("-inf")
    return numerator / denominator


def _count_significant_peaks(
    peaks: List[Dict[str, float]],
    *,
    rel_threshold: float = 0.1,
) -> int:
    """Count FFT peaks whose magnitude exceeds *rel_threshold* of the max."""
    if not peaks:
        return 0
    max_mag = max(p["magnitude"] for p in peaks)
    if max_mag <= 0.0:
        return 0
    cutoff = max_mag * rel_threshold
    return sum(1 for p in peaks if p["magnitude"] >= cutoff)


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
# Directional Classification (v74.2.0)
# ---------------------------------------------------------------------------

# Thresholds — simple deterministic logic, no ML.
_THRESHOLDS = {
    # Stable: all deltas must be within these bounds.
    "energy_abs": 0.02,
    "centroid_abs": 200.0,    # Hz
    "spread_abs": 150.0,      # Hz
    "zcr_abs": 0.02,
    "fft_sim_stable": 0.7,
    # Transition: structural reorganisation.
    "fft_sim_transition": 0.2,
    # Recovery: smooth re-concentration.
    "fft_sim_recovery": 0.5,
}


def classify_comparison(metrics: dict) -> str:
    """Classify the *nature* of the change between two compared signals.

    Uses directional (signed) deltas and ratio metrics to distinguish:

    * ``"stable"``    — minimal change across all metrics
    * ``"transition"`` — complete structural reorganisation (very low FFT
      similarity)
    * ``"collapse"``  — energy drops while spectrum diffuses (spread
      increases)
    * ``"recovery"``  — spectrum re-concentrates (spread decreases, centroid
      drops) with preserved FFT structure
    * ``"divergent"`` — spectrum expands without energy loss

    Parameters
    ----------
    metrics : dict
        Output of ``compare_sonic_features``.

    Returns
    -------
    str
        One of the five classification labels above.
    """
    t = _THRESHOLDS
    ed = metrics["energy_delta"]
    cd = metrics["centroid_delta"]       # signed
    sd = metrics["spread_delta"]         # signed
    zd = metrics["zcr_delta"]
    fs = metrics["fft_similarity"]

    # ------------------------------------------------------------------
    # 1. STABLE — very small deltas across all metrics
    # ------------------------------------------------------------------
    if (abs(ed) <= t["energy_abs"]
            and abs(cd) <= t["centroid_abs"]
            and abs(sd) <= t["spread_abs"]
            and abs(zd) <= t["zcr_abs"]
            and fs >= t["fft_sim_stable"]):
        return "stable"

    # ------------------------------------------------------------------
    # 2. TRANSITION — complete structural reorganisation
    #    FFT peaks are entirely different; no directional signature.
    # ------------------------------------------------------------------
    if fs < t["fft_sim_transition"]:
        return "transition"

    # ------------------------------------------------------------------
    # 3. COLLAPSE — energy drops + spectral diffusion
    #    Spread increases (signal loses coherence) while energy falls.
    # ------------------------------------------------------------------
    if ed < 0 and sd > t["spread_abs"]:
        return "collapse"

    # ------------------------------------------------------------------
    # 4. RECOVERY — spectral re-concentration
    #    Spread decreases and centroid drops (refocusing), with FFT
    #    structure preserved.
    # ------------------------------------------------------------------
    if (sd < -t["spread_abs"]
            and cd < 0
            and fs >= t["fft_sim_recovery"]):
        return "recovery"

    # ------------------------------------------------------------------
    # 5. DIVERGENT — spectrum expands without energy loss
    #    Spread increases while energy is stable or rising.
    # ------------------------------------------------------------------
    if sd > t["spread_abs"] and ed >= 0:
        return "divergent"

    # ------------------------------------------------------------------
    # Fallback — large change without clear directional signature.
    # ------------------------------------------------------------------
    return "transition"
