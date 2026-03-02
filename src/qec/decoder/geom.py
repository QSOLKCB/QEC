"""
Degree-aware message rebalancing — v3.8.0.

Provides precomputed per-check scaling factors based on check-node
degree for the ``geom_v1`` schedule.

Scaling rule
------------
For each check node j with degree d_c[j]:

    alpha_c(d_c[j]) = 1 / sqrt(d_c[j])

This is a purely deterministic function of graph topology.
Not adaptive.  Not URW.  Not global scalar scaling.

The scaling factors are precomputed once from the Tanner graph
and applied multiplicatively to check-to-variable messages during
the ``geom_v1`` BP schedule.
"""

from __future__ import annotations

import numpy as np


def compute_check_degree_scale(
    c2v: list[list[int]],
) -> np.ndarray:
    """Compute per-check scaling factors from check-node degrees.

    Parameters
    ----------
    c2v : list[list[int]]
        Check-to-variable adjacency list (length m).

    Returns
    -------
    alpha_c : np.ndarray
        Scaling factors of shape (m,), dtype float64.
        alpha_c[j] = 1.0 / sqrt(degree(j)).
        For degree-0 checks, alpha_c[j] = 1.0 (no-op).
    """
    m = len(c2v)
    alpha_c = np.ones(m, dtype=np.float64)
    for j in range(m):
        d = len(c2v[j])
        if d > 0:
            alpha_c[j] = 1.0 / np.sqrt(float(d))
    return alpha_c
