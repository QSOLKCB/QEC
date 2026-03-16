"""Deterministic difficulty-tier classification for spectral regions."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.qec.analysis.spectral_difficulty import estimate_spectral_difficulty


def _extract_region_spectra(memory: Any) -> np.ndarray:
    if memory is None:
        return np.zeros((0, 0), dtype=np.float64)
    if hasattr(memory, "centers") and callable(memory.centers):
        centers = np.asarray(memory.centers(), dtype=np.float64)
        if centers.ndim == 2:
            return centers
    if isinstance(memory, dict) and "region_spectra" in memory:
        regions = np.asarray(memory["region_spectra"], dtype=np.float64)
        if regions.ndim == 2:
            return regions
    return np.zeros((0, 0), dtype=np.float64)


def classify_region_difficulty(memory: Any) -> dict[str, list[int]]:
    """Classify regions into deterministic difficulty tiers via quantiles."""
    regions = _extract_region_spectra(memory)
    if regions.shape[0] == 0:
        return {
            "tier_0_easy": [],
            "tier_1_intermediate": [],
            "tier_2_hard": [],
            "tier_3_frontier": [],
        }

    scores = np.asarray(
        [estimate_spectral_difficulty(memory, regions[i]) for i in range(regions.shape[0])],
        dtype=np.float64,
    )

    q1, q2, q3 = np.quantile(scores, [0.25, 0.5, 0.75])
    idx = np.arange(regions.shape[0], dtype=np.int64)

    order = np.lexsort((idx, scores))

    tiers = {
        "tier_0_easy": [],
        "tier_1_intermediate": [],
        "tier_2_hard": [],
        "tier_3_frontier": [],
    }

    for oi in order:
        i = int(oi)
        score = float(scores[i])
        if score <= float(q1):
            tiers["tier_0_easy"].append(i)
        elif score <= float(q2):
            tiers["tier_1_intermediate"].append(i)
        elif score <= float(q3):
            tiers["tier_2_hard"].append(i)
        else:
            tiers["tier_3_frontier"].append(i)

    return tiers
