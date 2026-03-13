from __future__ import annotations

import numpy as np


_ROUND = 12


def participation_entropy(v: np.ndarray) -> float:
    x = np.asarray(v, dtype=np.float64).ravel()
    s2 = float(np.sum(x * x))
    if s2 <= 1e-30:
        return 0.0
    p = (x * x) / s2
    p = p[p > 0.0]
    return round(float(-np.sum(p * np.log(p))), _ROUND)


def inverse_participation_ratio(v: np.ndarray) -> float:
    x = np.asarray(v, dtype=np.float64).ravel()
    s2 = float(np.sum(x * x))
    if s2 <= 1e-30:
        return 0.0
    s4 = float(np.sum((x * x) ** 2))
    return round(s4 / (s2 * s2), _ROUND)


def effective_support_size(v: np.ndarray) -> float:
    return round(float(np.exp(participation_entropy(v))), _ROUND)


def extract_support(v: np.ndarray, tau: float | None = None) -> list[int]:
    x = np.asarray(v, dtype=np.float64).ravel()
    if x.size == 0:
        return []
    abs_x = np.abs(x)
    if tau is None:
        tau = 0.1 * float(np.max(abs_x))
    thr = float(tau)
    idx = [int(i) for i, val in enumerate(abs_x) if float(val) >= thr and val > 0.0]
    idx.sort()
    return idx
