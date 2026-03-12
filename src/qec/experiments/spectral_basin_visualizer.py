"""Spectral basin-of-attraction trajectory utilities."""

from __future__ import annotations

from typing import Any

import numpy as np
import scipy.sparse

from src.qec.analysis.eigenvector_localization import EigenvectorLocalizationAnalyzer
from src.qec.analysis.nonbacktracking_flow import NonBacktrackingFlowAnalyzer
from src.qec.discovery.mutation_nb_gradient import NBGradientMutator


_ROUND = 12


class SpectralBasinVisualizer:
    """Track deterministic mutation trajectories in spectral phase space."""

    def __init__(self) -> None:
        self._flow_analyzer = NonBacktrackingFlowAnalyzer()
        self._mutator = NBGradientMutator(
            enabled=True,
            avoid_4cycles=True,
            flow_damping=True,
        )

    @staticmethod
    def _to_dense_copy(H: np.ndarray | scipy.sparse.spmatrix) -> np.ndarray:
        if scipy.sparse.issparse(H):
            return np.asarray(H.todense(), dtype=np.float64)
        return np.asarray(H, dtype=np.float64).copy()

    def _compute_point(self, H: np.ndarray, iteration: int) -> dict[str, Any]:
        flow = self._flow_analyzer.compute_flow(H)
        ipr_result = EigenvectorLocalizationAnalyzer.compute_ipr(
            np.asarray(flow["variable_flow"], dtype=np.float64),
        )
        directed_edge_flow = np.asarray(
            flow.get("directed_edge_flow", np.zeros(0, dtype=np.float64)),
            dtype=np.float64,
        )
        nb_energy = float(np.sum(directed_edge_flow ** 2))

        return {
            "iteration": int(iteration),
            "spectral_radius": round(float(flow.get("max_flow", 0.0)), _ROUND),
            "ipr": round(float(ipr_result["ipr"]), _ROUND),
            "nb_energy": round(nb_energy, _ROUND),
        }

    def trace_mutation_trajectory(
        self,
        H: np.ndarray | scipy.sparse.spmatrix,
        iterations: int = 10,
    ) -> list[dict[str, Any]]:
        """Trace spectral trajectory under deterministic NB gradient mutation."""
        H_current = self._to_dense_copy(H)
        trajectory = [self._compute_point(H_current, iteration=0)]

        if iterations <= 0:
            return trajectory

        for idx in range(1, iterations + 1):
            H_next, mutation_log = self._mutator.mutate(H_current, steps=1)
            if not mutation_log:
                trajectory.append(self._compute_point(H_current, iteration=idx))
                continue
            H_current = H_next
            trajectory.append(self._compute_point(H_current, iteration=idx))

        return trajectory
