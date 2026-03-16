"""Deterministic helpers for novelty-driven spectral phase discovery."""

from __future__ import annotations

import numpy as np


def _as_matrix(vectors: list[np.ndarray] | np.ndarray) -> np.ndarray:
    """Convert vectors to a deterministic float64 2D array."""
    arr = np.asarray(vectors, dtype=np.float64)
    if arr.size == 0:
        return np.zeros((0, 0), dtype=np.float64)
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    if arr.ndim == 2:
        return arr
    return arr.reshape(arr.shape[0], -1)


def compute_phase_novelty_score(
    vector: np.ndarray,
    known_phase_centroids: list[np.ndarray] | np.ndarray,
) -> float:
    """Return minimum Euclidean distance from vector to known centroids."""
    centroids = _as_matrix(known_phase_centroids)
    vector = np.asarray(vector, dtype=np.float64).reshape(-1)
    if centroids.size == 0:
        return float(np.float64(0.0))

    dims = int(min(vector.size, centroids.shape[1]))
    if dims <= 0:
        return float(np.float64(0.0))

    distances = np.linalg.norm(centroids[:, :dims] - vector[:dims], axis=1).astype(np.float64, copy=False)
    score = float(np.float64(np.min(distances)))
    return score


def select_novel_phase_target(
    candidate_vectors: list[np.ndarray] | np.ndarray,
    known_phase_centroids: list[np.ndarray] | np.ndarray,
) -> dict[str, np.ndarray]:
    """Select the candidate with maximal novelty, with deterministic tie-breaking."""
    candidates = _as_matrix(candidate_vectors)
    if candidates.size == 0 or candidates.shape[0] == 0:
        return {"novelty_vector": np.zeros((0,), dtype=np.float64)}

    scores = np.asarray(
        [compute_phase_novelty_score(row, known_phase_centroids) for row in candidates],
        dtype=np.float64,
    )
    order_keys: list[np.ndarray] = [np.arange(candidates.shape[0], dtype=np.int64)]
    for dim in range(candidates.shape[1] - 1, -1, -1):
        order_keys.append(candidates[:, dim])
    order_keys.append(-scores)
    order = np.lexsort(tuple(order_keys))
    return {"novelty_vector": candidates[int(order[0])].astype(np.float64, copy=True)}


def propose_phase_novelty_step(
    current_vector: np.ndarray,
    novelty_vector: np.ndarray,
) -> np.ndarray:
    """Take a deterministic normalized step from current vector toward novelty vector."""
    current = np.asarray(current_vector, dtype=np.float64).reshape(-1)
    target = np.asarray(novelty_vector, dtype=np.float64).reshape(-1)
    dims = int(min(current.size, target.size))
    if dims <= 0:
        return current.copy()

    out = current.copy()
    direction = target[:dims] - current[:dims]
    norm = float(np.float64(np.linalg.norm(direction)))
    if norm <= 0.0:
        return out.astype(np.float64, copy=False)

    unit_direction = direction / np.float64(norm)
    out[:dims] = current[:dims] + np.float64(0.25) * unit_direction
    return out.astype(np.float64, copy=False)


def detect_new_phase(
    vector: np.ndarray,
    known_phase_centroids: list[np.ndarray] | np.ndarray,
    threshold: float,
) -> dict[str, bool]:
    """Classify vector as potential new phase when all centroid distances exceed threshold."""
    score = compute_phase_novelty_score(vector, known_phase_centroids)
    return {"is_new_phase": bool(float(np.float64(score)) > float(np.float64(threshold)))}
