"""v14.2.0 Deterministic non-backtracking eigenvector flow optimizer."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse

from src.qec.analysis.bethe_hessian import BetheHessianAnalyzer
from src.qec.analysis.nonbacktracking_flow import EdgeFlowField, NBFlowConfig, NonBacktrackingEigenvectorFlowAnalyzer
from src.qec.fitness.spectral_metrics import compute_girth_spectrum


@dataclass(frozen=True)
class NBFlowStepResult:
    step: int
    accepted: bool
    swap: tuple[int, int, int, int] | None
    delta_flow: float
    combined_score: float
    reason: str


@dataclass(frozen=True)
class NBFlowTrajectory:
    H_final: np.ndarray
    steps: tuple[NBFlowStepResult, ...]
    termination_reason: str


class NonBacktrackingEigenvectorFlowOptimizer:
    """Apply deterministic degree-preserving 2x2 rewires from NB edge flow."""

    def __init__(
        self,
        *,
        flow_config: NBFlowConfig | None = None,
        max_steps: int = 1000,
        flow_tol: float = 1e-8,
        swap_budget: int = 500,
        min_girth: int = 6,
        eta_bh: float = 0.1,
        use_bh_acceptance: bool = True,
    ) -> None:
        self.flow = NonBacktrackingEigenvectorFlowAnalyzer(config=flow_config or NBFlowConfig())
        self.max_steps = int(max_steps)
        self.flow_tol = float(flow_tol)
        self.swap_budget = int(swap_budget)
        self.min_girth = int(min_girth)
        self.eta_bh = float(eta_bh)
        self.use_bh_acceptance = bool(use_bh_acceptance)
        self._bh = BetheHessianAnalyzer()

    def optimize(self, H: np.ndarray | scipy.sparse.spmatrix) -> NBFlowTrajectory:
        H_curr = np.asarray(scipy.sparse.csr_matrix(H, dtype=np.float64).toarray(), dtype=np.float64)
        steps: list[NBFlowStepResult] = []

        for step in range(self.max_steps):
            field = self.flow.build_flow_field(H_curr)
            if len(field.selected_modes) == 0:
                return NBFlowTrajectory(H_final=H_curr, steps=tuple(steps), termination_reason='no_informative_nb_mode')

            swaps = self._enumerate_swaps(H_curr)
            if not swaps:
                return NBFlowTrajectory(H_final=H_curr, steps=tuple(steps), termination_reason='no_improving_swap')

            base_bh = self._bh_value(H_curr)
            ranked = []
            for swap in swaps[: self.swap_budget]:
                delta = self._delta_flow(swap, field)
                H_cand = self._apply_swap(H_curr, swap)
                if self.min_girth > 0 and compute_girth_spectrum(H_cand).get('girth', 0) < self.min_girth:
                    continue
                delta_bh = self._bh_value(H_cand) - base_bh
                score = float(delta + self.eta_bh * delta_bh)
                ranked.append(((round(score, 12),) + swap, swap, delta, score, delta_bh))

            if not ranked:
                return NBFlowTrajectory(H_final=H_curr, steps=tuple(steps), termination_reason='no_improving_swap')

            ranked.sort(key=lambda item: item[0])
            accepted = False
            rejected_swap: tuple[int, int, int, int] | None = None
            rejected_delta = 0.0
            rejected_score = 0.0

            for _, swap, delta, score, delta_bh in ranked:
                if delta >= -self.flow_tol:
                    continue
                if self.use_bh_acceptance and delta_bh > 0.0:
                    if rejected_swap is None:
                        rejected_swap = swap
                        rejected_delta = delta
                        rejected_score = score
                    continue

                H_curr = self._apply_swap(H_curr, swap)
                steps.append(NBFlowStepResult(step=step, accepted=True, swap=swap, delta_flow=round(float(delta), 12), combined_score=round(float(score), 12), reason='accepted'))
                accepted = True
                break

            if accepted:
                continue

            if rejected_swap is not None:
                steps.append(NBFlowStepResult(step=step, accepted=False, swap=rejected_swap, delta_flow=round(float(rejected_delta), 12), combined_score=round(float(rejected_score), 12), reason='bh_rejected'))
                return NBFlowTrajectory(H_final=H_curr, steps=tuple(steps), termination_reason='no_improving_swap')

            fallback_swap = ranked[0][1]
            fallback_delta = ranked[0][2]
            fallback_score = ranked[0][3]
            steps.append(NBFlowStepResult(step=step, accepted=False, swap=fallback_swap, delta_flow=round(float(fallback_delta), 12), combined_score=round(float(fallback_score), 12), reason='flow_converged'))
            return NBFlowTrajectory(H_final=H_curr, steps=tuple(steps), termination_reason='flow_converged')

        return NBFlowTrajectory(H_final=H_curr, steps=tuple(steps), termination_reason='max_steps_reached')

    @staticmethod
    def _enumerate_swaps(H: np.ndarray) -> list[tuple[int, int, int, int]]:
        m, n = H.shape
        check_neighbors = {ci: set(np.flatnonzero(H[ci] != 0).tolist()) for ci in range(m)}
        var_neighbors = {vi: set(np.flatnonzero(H[:, vi] != 0).tolist()) for vi in range(n)}
        swaps: list[tuple[int, int, int, int]] = []
        all_vars = tuple(range(n))
        var_set = set(all_vars)
        for ci in range(m):
            ci_neighbors = set(check_neighbors[ci])
            non_neighbor_vars = sorted(var_set - ci_neighbors)
            for vi in sorted(ci_neighbors):
                if len(ci_neighbors) <= 1 or len(var_neighbors[vi]) <= 1:
                    continue
                for vj in non_neighbor_vars:
                    if vj == vi:
                        continue
                    for cj in sorted(var_neighbors[vj]):
                        if cj == ci or H[cj, vi] != 0 or len(check_neighbors[cj]) <= 1:
                            continue
                        swaps.append((ci, vi, cj, vj))
        swaps.sort()
        return swaps

    @staticmethod
    def _apply_swap(H: np.ndarray, swap: tuple[int, int, int, int]) -> np.ndarray:
        ci, vi, cj, vj = swap
        out = H.copy()
        out[ci, vi] = 0.0
        out[cj, vj] = 0.0
        out[ci, vj] = 1.0
        out[cj, vi] = 1.0
        return out

    @staticmethod
    def _delta_flow(swap: tuple[int, int, int, int], field: EdgeFlowField) -> float:
        ci, vi, cj, vj = swap
        e = field.edge_pressure_map
        return float((e.get((ci, vj), 0.0) + e.get((cj, vi), 0.0)) - (e.get((ci, vi), 0.0) + e.get((cj, vj), 0.0)))

    def _bh_value(self, H: np.ndarray) -> float:
        if not self.use_bh_acceptance and self.eta_bh == 0.0:
            return 0.0
        return float(self._bh.compute_bethe_hessian_stability(H).get('bethe_hessian_min_eigenvalue', 0.0))
