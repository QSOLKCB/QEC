"""Deterministic first-order NB perturbation scorer (FOHPE-style)."""

from __future__ import annotations

from typing import Any

import numpy as np
import scipy.sparse

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
        raw = np.asarray(flow.get("directed_edge_flow", np.zeros(0, dtype=np.float64)), dtype=np.float64)
        if m == 0 or n == 0 or len(directed_edges) == 0 or raw.size == 0:
            return {
                "valid_first_order": False,
                "directed_edges": directed_edges,
                "index": {edge: i for i, edge in enumerate(directed_edges)},
                "u": np.zeros(0, dtype=np.float64),
                "v": np.zeros(0, dtype=np.float64),
            }

        u = np.abs(raw).astype(np.float64, copy=False)
        v = np.abs(raw).astype(np.float64, copy=True)

        u, ok_u = self._l2_normalize(u)
        v, ok_v = self._l2_normalize(v)
        if not ok_u or not ok_v:
            return {
                "valid_first_order": False,
                "directed_edges": directed_edges,
                "index": {edge: i for i, edge in enumerate(directed_edges)},
                "u": np.zeros(0, dtype=np.float64),
                "v": np.zeros(0, dtype=np.float64),
            }

        return {
            "valid_first_order": True,
            "directed_edges": directed_edges,
            "index": {edge: i for i, edge in enumerate(directed_edges)},
            "u": u,
            "v": v,
        }

    def predict_swap_delta(
        self,
        H: np.ndarray | scipy.sparse.spmatrix,
        swap: tuple[int, int, int, int],
        spectrum: dict[str, Any],
    ) -> dict[str, float | bool]:
        """Predict first-order perturbation delta and pressure for one swap."""
        if not bool(spectrum.get("valid_first_order", False)):
            return {"valid_first_order": False, "predicted_delta": 0.0, "pressure": 0.0}

        H_arr = self._to_dense_copy(H)
        n = H_arr.shape[1]
        index = spectrum.get("index", {})
        u = np.asarray(spectrum.get("u", np.zeros(0, dtype=np.float64)), dtype=np.float64)
        v = np.asarray(spectrum.get("v", np.zeros(0, dtype=np.float64)), dtype=np.float64)

        if u.size == 0 or v.size == 0:
            return {"valid_first_order": False, "predicted_delta": 0.0, "pressure": 0.0}

        ci, vi, cj, vj = swap
        removed = [(ci, vi), (cj, vj)]
        added = [(ci, vj), (cj, vi)]

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
            return {"valid_first_order": False, "predicted_delta": 0.0, "pressure": 0.0}

        predicted_delta = 0.0
        for idx, coeff in delta_terms:
            if idx >= len(u) or idx >= len(v):
                return {"valid_first_order": False, "predicted_delta": 0.0, "pressure": 0.0}
            predicted_delta += coeff * float(v[idx]) * float(u[idx])

        i = index.get((vi, n + ci))
        j = index.get((vj, n + cj))
        pressure = 0.0
        if i is not None and j is not None and i < len(u) and j < len(v):
            pressure = abs(float(u[i])) * abs(float(v[j]))

        return {
            "valid_first_order": True,
            "predicted_delta": round(float(predicted_delta), self.precision),
            "pressure": round(float(pressure), self.precision),
        }

    @staticmethod
    def _l2_normalize(vec: np.ndarray) -> tuple[np.ndarray, bool]:
        norm = float(np.linalg.norm(vec, ord=2))
        if not np.isfinite(norm) or norm <= 1e-15:
            return np.zeros(0, dtype=np.float64), False
        return np.asarray(vec / norm, dtype=np.float64), True

    @staticmethod
    def _to_dense_copy(H: np.ndarray | scipy.sparse.spmatrix) -> np.ndarray:
        if scipy.sparse.issparse(H):
            return np.asarray(H.todense(), dtype=np.float64)
        return np.asarray(H, dtype=np.float64).copy()
