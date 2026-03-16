"""Deterministic motif-guided Tanner graph mutation."""

from __future__ import annotations

from typing import Any

import numpy as np


def apply_motif_mutation(graph: np.ndarray, motif: dict[str, Any] | None) -> np.ndarray:
    """Inject motif structure into a Tanner graph deterministically."""
    H = np.asarray(graph, dtype=np.float64).copy()
    if motif is None or H.ndim != 2 or H.size == 0:
        return H

    signature = np.asarray(motif.get("spectral_signature", []), dtype=np.float64).reshape(-1)
    if signature.size == 0:
        return H

    m, n = H.shape
    pivot = int(abs(int(np.round(signature[0] * 1e6))))
    src_row = int(pivot % max(1, m))
    src_col = int((pivot // max(1, m)) % max(1, n))
    dst_row = int((src_row + 1) % max(1, m))
    dst_col = int((src_col + 1) % max(1, n))

    if H[src_row, src_col] > 0.5 and H[dst_row, dst_col] < 0.5:
        H[src_row, src_col] = 0.0
        H[dst_row, dst_col] = 1.0
    elif H[dst_row, src_col] > 0.5 and H[src_row, dst_col] < 0.5:
        H[dst_row, src_col] = 0.0
        H[src_row, dst_col] = 1.0

    return H.astype(np.float64, copy=False)
