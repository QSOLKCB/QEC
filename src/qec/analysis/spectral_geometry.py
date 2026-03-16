"""Deterministic spectral-space geometry helpers."""

from __future__ import annotations

import numpy as np


def spectral_distance(a, b) -> float:
    """Euclidean distance between two spectral feature vectors."""
    a_arr = np.asarray(a, dtype=np.float64)
    b_arr = np.asarray(b, dtype=np.float64)
    return float(np.float64(np.linalg.norm(a_arr - b_arr)))


def trajectory_arc_length(points) -> float:
    """Deterministic arc length over an ordered spectral trajectory."""
    pts = np.asarray(points, dtype=np.float64)
    if pts.shape[0] < 2:
        return 0.0
    total = np.float64(0.0)
    for idx in range(1, int(pts.shape[0])):
        total = np.float64(total + np.float64(spectral_distance(pts[idx - 1], pts[idx])))
    return float(total)


def estimate_local_curvature(points) -> list[float]:
    """Estimate local turning angle (radians) for sliding 3-point windows."""
    pts = np.asarray(points, dtype=np.float64)
    if pts.shape[0] < 3:
        return []
    curvatures: list[float] = []
    for idx in range(1, int(pts.shape[0]) - 1):
        prev_seg = np.asarray(pts[idx] - pts[idx - 1], dtype=np.float64)
        next_seg = np.asarray(pts[idx + 1] - pts[idx], dtype=np.float64)
        prev_norm = np.float64(np.linalg.norm(prev_seg))
        next_norm = np.float64(np.linalg.norm(next_seg))
        if prev_norm <= 0.0 or next_norm <= 0.0:
            curvatures.append(0.0)
            continue
        cosine = np.float64(np.dot(prev_seg, next_seg) / (prev_norm * next_norm))
        angle = np.float64(np.arccos(np.clip(cosine, -1.0, 1.0)))
        curvatures.append(float(angle))
    return curvatures


def estimate_basin_geometry(points) -> dict[str, float]:
    """Compute deterministic spectral basin geometry descriptors."""
    pts = np.asarray(points, dtype=np.float64)
    if pts.shape[0] == 0:
        return {
            "basin_radius_estimate": 0.0,
            "mean_step_length": 0.0,
            "mean_curvature": 0.0,
            "spectral_dispersion": 0.0,
        }

    centroid = np.asarray(np.mean(pts, axis=0, dtype=np.float64), dtype=np.float64)
    centered = np.asarray(pts - centroid, dtype=np.float64)
    radial_distances = np.asarray(np.linalg.norm(centered, axis=1), dtype=np.float64)
    curvatures = np.asarray(estimate_local_curvature(pts), dtype=np.float64)

    if pts.shape[0] < 2:
        mean_step = np.float64(0.0)
    else:
        diffs = np.asarray(pts[1:] - pts[:-1], dtype=np.float64)
        step_lengths = np.asarray(np.linalg.norm(diffs, axis=1), dtype=np.float64)
        mean_step = np.float64(np.mean(step_lengths, dtype=np.float64))

    return {
        "basin_radius_estimate": float(np.float64(np.max(radial_distances))),
        "mean_step_length": float(mean_step),
        "mean_curvature": float(np.float64(np.mean(curvatures, dtype=np.float64))) if curvatures.size > 0 else 0.0,
        "spectral_dispersion": float(np.float64(np.mean(radial_distances, dtype=np.float64))),
    }


def spectral_entropy(eigs) -> float:
    """Shannon entropy of normalized spectral magnitudes."""
    eigs_arr = np.abs(np.asarray(eigs, dtype=np.float64))
    total = float(eigs_arr.sum())
    if total <= 0.0:
        return 0.0
    p = eigs_arr / total
    return float(-np.sum(p * np.log(p + 1e-12)))


def spectral_diversity(history) -> float:
    """Mean consecutive spectral distance across a trajectory."""
    if len(history) < 2:
        return 0.0
    dists = [spectral_distance(history[i], history[i - 1]) for i in range(1, len(history))]
    return float(np.mean(np.asarray(dists, dtype=np.float64)))
