"""Deterministic first-order NB perturbation scorer (FOHPE-style)."""

from __future__ import annotations

from typing import Any

import numpy as np
import scipy.sparse

from src.qec.analysis.constants import MIN_EIGENVECTOR_NORM, MIN_FOHPE_DENOM
from src.qec.analysis.nonbacktracking_flow import NonBacktrackingFlowAnalyzer


_ROUND = 12


class NBPerturbationScorer:
    """Score deterministic swap perturbations with first-order NB proxies."""

    def __init__(self, *, power_iterations: int = 50, precision: int = _ROUND) -> None:
        self.precision = precision
        self._flow = NonBacktrackingFlowAnalyzer(power_iterations=power_iterations)

    def compute_nb_spectrum(
        self,
        H: np.ndarray | scipy.sparse.spmatrix,
    ) -> dict[str, Any]:
        """Compute deterministic NB edge-mode vectors used by first-order scoring."""
        H_arr = self._to_dense_copy(H)
        m, n = H_arr.shape
        flow = self._flow.compute_flow(H_arr)

        directed_edges = flow.get("directed_edges", [])
        index = {edge: i for i, edge in enumerate(directed_edges)}
        raw = np.asarray(flow.get("directed_edge_flow", np.zeros(0, dtype=np.float64)), dtype=np.float64)
        if m == 0 or n == 0 or len(directed_edges) == 0 or raw.size == 0:
            return {
                "valid_first_order": False,
                "directed_edges": directed_edges,
                "index": index,
                "u": np.zeros(0, dtype=np.float64),
                "v": np.zeros(0, dtype=np.float64),
                "fohpe_denominator": 0.0,
            }

        u, ok_u = self._l2_normalize(np.asarray(np.abs(raw), dtype=np.float64))
        v, ok_v = self._l2_normalize(np.asarray(np.abs(raw), dtype=np.float64))
        if not ok_u or not ok_v:
            return {
                "valid_first_order": False,
                "directed_edges": directed_edges,
                "index": index,
                "u": np.zeros(0, dtype=np.float64),
                "v": np.zeros(0, dtype=np.float64),
                "fohpe_denominator": 0.0,
            }

        denom = np.vdot(v, u)
        if not np.isfinite(float(np.abs(denom))) or float(np.abs(denom)) < MIN_FOHPE_DENOM:
            return {
                "valid_first_order": False,
                "directed_edges": directed_edges,
                "index": index,
                "u": np.zeros(0, dtype=np.float64),
                "v": np.zeros(0, dtype=np.float64),
                "fohpe_denominator": 0.0,
            }

        return {
            "valid_first_order": True,
            "directed_edges": directed_edges,
            "index": index,
            "u": u,
            "v": v,
            "fohpe_denominator": float(np.real(denom)),
        }

    def predict_swap_delta(
        self,
        H: np.ndarray | scipy.sparse.spmatrix,
        swap: tuple[int, int, int, int],
        spectrum: dict[str, Any],
    ) -> dict[str, float | bool] | None:
        """Predict first-order perturbation delta and pressure for one swap."""
        if not bool(spectrum.get("valid_first_order", False)):
            return None

        H_arr = self._to_dense_copy(H)
        m, n = H_arr.shape
        ci, vi, cj, vj = swap
        if (
            ci < 0
            or cj < 0
            or vi < 0
            or vj < 0
            or ci >= m
            or cj >= m
            or vi >= n
            or vj >= n
            or ci == cj
            or vi == vj
        ):
            return None

        if H_arr[ci, vi] == 0.0 or H_arr[cj, vj] == 0.0:
            return None
        if H_arr[ci, vj] != 0.0 or H_arr[cj, vi] != 0.0:
            return None

        index = spectrum.get("index", {})
        u = np.asarray(spectrum.get("u", np.zeros(0, dtype=np.float64)), dtype=np.float64)
        v = np.asarray(spectrum.get("v", np.zeros(0, dtype=np.float64)), dtype=np.float64)
        denom = float(spectrum.get("fohpe_denominator", 0.0))

        if u.size == 0 or v.size == 0 or len(u) != len(v):
            return None
        if not np.isfinite(denom) or abs(denom) < MIN_FOHPE_DENOM:
            return None

        removed = ((ci, vi), (cj, vj))
        added = ((ci, vj), (cj, vi))

        delta_terms: list[tuple[int, float]] = []
        for cix, vix in removed:
            j1 = index.get((vix, n + cix))
            j2 = index.get((n + cix, vix))
            if j1 is not None:
                delta_terms.append((int(j1), -1.0))
            if j2 is not None:
                delta_terms.append((int(j2), -1.0))

        for cix, vix in added:
            j1 = index.get((vix, n + cix))
            j2 = index.get((n + cix, vix))
            if j1 is not None:
                delta_terms.append((int(j1), 1.0))
            if j2 is not None:
                delta_terms.append((int(j2), 1.0))

        if not delta_terms:
            return None

        delta = 0.0
        for idx, coeff in delta_terms:
            if idx < 0 or idx >= len(u):
                return None
            delta += coeff * float(v[idx]) * float(u[idx])

        predicted_delta = delta / denom

        i = index.get((vi, n + ci))
        j = index.get((vj, n + cj))
        pressure = 0.0
        if i is not None and j is not None and i < len(u) and j < len(v):
            pressure = abs(float(u[int(i)])) * abs(float(v[int(j)]))

        return {
            "valid_first_order": True,
            "predicted_delta": round(float(predicted_delta), self.precision),
            "pressure": round(float(pressure), self.precision),
        }

    @staticmethod
    def _l2_normalize(vec: np.ndarray) -> tuple[np.ndarray, bool]:
        norm = float(np.linalg.norm(vec, ord=2))
        if not np.isfinite(norm) or norm <= MIN_EIGENVECTOR_NORM:
            return np.zeros(0, dtype=np.float64), False
        return np.asarray(vec / norm, dtype=np.float64), True

    @staticmethod
    def _to_dense_copy(H: np.ndarray | scipy.sparse.spmatrix) -> np.ndarray:
        if isinstance(H, np.ndarray) and H.dtype == np.float64:
            return H
        if scipy.sparse.issparse(H):
            return np.asarray(H.toarray(), dtype=np.float64)
        return np.asarray(H, dtype=np.float64)
