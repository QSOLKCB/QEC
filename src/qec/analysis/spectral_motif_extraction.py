"""Deterministic spectral motif extraction from discovery archive entries."""

from __future__ import annotations

from typing import Any

import numpy as np

_ROUND = 12


def _iter_archive_entries(archive: dict[str, Any] | list[dict[str, Any]]) -> list[dict[str, Any]]:
    if isinstance(archive, list):
        return sorted(list(archive), key=lambda e: str(e.get("candidate_id", "")))
    categories = archive.get("categories", {}) if isinstance(archive, dict) else {}
    seen: dict[str, dict[str, Any]] = {}
    for entries in categories.values():
        for entry in entries:
            cid = str(entry.get("candidate_id", ""))
            seen[cid] = entry
    return sorted(seen.values(), key=lambda e: str(e.get("candidate_id", "")))


def _spectral_signature(objectives: dict[str, Any]) -> np.ndarray:
    return np.asarray(
        [
            float(objectives.get("spectral_radius", 0.0)),
            float(objectives.get("bethe_margin", 0.0)),
            float(objectives.get("ipr_localization", 0.0)),
            float(objectives.get("entropy", 0.0)),
        ],
        dtype=np.float64,
    )


def extract_spectral_motifs(archive: dict[str, Any] | list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Extract recurring deterministic motifs from high-performing archive entries."""
    entries = _iter_archive_entries(archive)
    if not entries:
        return []

    composites = [float(e.get("objectives", {}).get("composite_score", np.inf)) for e in entries]
    finite = np.asarray([c for c in composites if np.isfinite(c)], dtype=np.float64)
    if finite.size == 0:
        return []

    threshold = float(np.percentile(finite, 25.0))
    selected = [
        e for e in entries
        if float(e.get("objectives", {}).get("composite_score", np.inf)) <= threshold
    ]
    if not selected:
        selected = entries[:1]

    grouped: dict[tuple[float, ...], list[np.ndarray]] = {}
    for entry in selected:
        sig = _spectral_signature(entry.get("objectives", {}))
        key = tuple(float(np.round(v, _ROUND)) for v in sig.tolist())
        grouped.setdefault(key, []).append(sig)

    motifs: list[dict[str, Any]] = []
    for idx, (_key, signatures) in enumerate(sorted(grouped.items(), key=lambda item: item[0])):
        arr = np.vstack(signatures).astype(np.float64, copy=False)
        centroid = np.mean(arr, axis=0, dtype=np.float64).astype(np.float64, copy=False)
        motifs.append({
            "motif_id": int(idx),
            "spectral_signature": np.round(centroid, _ROUND).astype(np.float64, copy=False),
            "frequency": int(arr.shape[0]),
        })
    return motifs
