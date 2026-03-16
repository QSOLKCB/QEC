"""Deterministic inter-agent spacing enforcement in spectral space."""

from __future__ import annotations

import numpy as np


def _distance_sq(a: np.ndarray, b: np.ndarray) -> float:
    diff = a - b
    return float(np.dot(diff, diff))


def enforce_agent_spacing(
    targets: dict[str, np.ndarray | None],
    min_distance: float = 0.3,
) -> dict[str, np.ndarray | None]:
    """Deterministically adjust close targets along their difference vectors."""
    adjusted = {
        aid: (None if targets[aid] is None else np.asarray(targets[aid], dtype=np.float64).copy())
        for aid in sorted(targets.keys())
    }

    keys = list(adjusted.keys())
    min_dist_sq = float(min_distance) * float(min_distance)

    for i in range(len(keys)):
        ai = keys[i]
        ti = adjusted[ai]
        if ti is None:
            continue
        for j in range(i + 1, len(keys)):
            aj = keys[j]
            tj = adjusted[aj]
            if tj is None:
                continue
            dist_sq = _distance_sq(ti, tj)
            if dist_sq >= min_dist_sq:
                continue

            diff = tj - ti
            norm_sq = float(np.dot(diff, diff))
            if norm_sq <= 1e-18:
                # deterministic fallback axis
                basis = np.zeros_like(tj, dtype=np.float64)
                if basis.size > 0:
                    basis[0] = 1.0
                direction = basis
            else:
                direction = diff / np.sqrt(norm_sq)

            needed = float(min_distance) - float(np.sqrt(dist_sq))
            step = max(needed, 0.0)
            tj = tj + direction * step
            adjusted[aj] = tj.astype(np.float64, copy=False)

    return adjusted
