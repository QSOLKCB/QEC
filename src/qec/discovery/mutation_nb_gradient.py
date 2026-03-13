"""
v12.6.0 — NB Instability Gradient Mutation Operator.

Applies deterministic degree-preserving Tanner-graph rewiring steps that
follow the instability gradient induced by the NB dominant eigenvector
field.

Layer 5 — Discovery.
Does not import or modify the decoder (Layer 1).
Fully deterministic: no randomness, no global state, no input mutation.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import scipy.sparse

from src.qec.analysis.nb_instability_gradient import NBInstabilityGradientAnalyzer
from src.qec.analysis.nb_spectral_basin_steering import NBSpectralBasinSteering
from src.qec.analysis.nb_trapping_set_predictor import NBTrappingSetPredictor
from src.qec.discovery.spectral_beam_search import plan_two_step_swap


_ROUND = 12


class NBGradientMutator:
    """Deterministic instability-gradient guided mutation operator.

    Parameters
    ----------
    flow_damping_alpha : float
        Convex damping weight in [0.0, 1.0] used when ``flow_damping=True``.
    """

    def __init__(
        self,
        *,
        enabled: bool = False,
        avoid_4cycles: bool = True,
        flow_damping: bool = False,
        flow_damping_alpha: float = 0.5,
        avoid_predicted_trapping_sets: bool = False,
        steer_spectral_basins: bool = False,
        enable_spectral_beam_search: bool = False,
        beam_width: int = 5,
        second_step_limit: int = 5,
        beam_activation_states: tuple[str, ...] = ("metastable_plateau", "localized_trap"),
        beam_score_weight: float = 1.0,
        precision: int = _ROUND,
    ) -> None:
        if not (0.0 <= flow_damping_alpha <= 1.0):
            raise ValueError("flow_damping_alpha must be between 0 and 1")

        self.enabled = enabled
        self.avoid_4cycles = avoid_4cycles
        self.flow_damping = flow_damping
        self.flow_damping_alpha = flow_damping_alpha
        self.avoid_predicted_trapping_sets = avoid_predicted_trapping_sets
        self.steer_spectral_basins = steer_spectral_basins
        self.enable_spectral_beam_search = enable_spectral_beam_search
        self.beam_width = max(1, int(beam_width))
        self.second_step_limit = max(1, int(second_step_limit))
        self.beam_activation_states = tuple(str(s) for s in beam_activation_states)
        self.beam_score_weight = float(beam_score_weight)
        self.precision = precision
        self._analyzer = NBInstabilityGradientAnalyzer()
        self._trapping_predictor = NBTrappingSetPredictor(precision=precision)
        self._basin_steering = NBSpectralBasinSteering(precision=precision)

    def mutate(
        self,
        H: np.ndarray | scipy.sparse.spmatrix,
        steps: int = 5,
    ) -> tuple[np.ndarray, list[dict[str, Any]]]:
        """Apply up to ``steps`` deterministic gradient rewiring swaps."""
        H_new = self._to_dense_copy(H)
        if not self.enabled or steps <= 0:
            return H_new, []

        mutations: list[dict[str, Any]] = []
        for _ in range(steps):
            gradient = self._analyzer.compute_gradient(H_new)
            step = self._apply_single_gradient_step(H_new, gradient)
            if step is None:
                break
            step.setdefault("beam_search_used", False)
            step.setdefault("beam_width", 0)
            step.setdefault("planned_sequence_score", None)
            mutations.append(step)

        return H_new, mutations

    def mutate_flow(
        self,
        H: np.ndarray | scipy.sparse.spmatrix,
        iterations: int = 10,
    ) -> tuple[np.ndarray, list[dict[str, Any]]]:
        """Simulate continuous topology evolution via minimal gradient steps."""
        H_new = self._to_dense_copy(H)
        if not self.enabled or iterations <= 0:
            return H_new, []

        mutations: list[dict[str, Any]] = []
        prev_direction: dict[tuple[int, int], float] = {}
        for _ in range(iterations):
            gradient = self._analyzer.compute_gradient(H_new)
            if self.flow_damping and prev_direction:
                damped_direction: dict[tuple[int, int], float] = {}
                for edge, value in gradient["gradient_direction"].items():
                    prev_val = prev_direction.get(edge, 0.0)
                    damped_direction[edge] = round(
                        self.flow_damping_alpha * float(value)
                        + (1.0 - self.flow_damping_alpha) * float(prev_val),
                        self.precision,
                    )
                gradient["gradient_direction"] = damped_direction

            if self.enable_spectral_beam_search:
                step = self._apply_single_gradient_step(
                    H_new,
                    gradient,
                    allow_beam=True,
                )
            else:
                step = self._apply_single_gradient_step(H_new, gradient)
            if step is None:
                break
            step.setdefault("beam_search_used", False)
            step.setdefault("beam_width", 0)
            step.setdefault("planned_sequence_score", None)
            mutations.append(step)
            prev_direction = dict(gradient["gradient_direction"])

        return H_new, mutations

    @staticmethod
    def _to_dense_copy(H: np.ndarray | scipy.sparse.spmatrix) -> np.ndarray:
        if scipy.sparse.issparse(H):
            return np.asarray(H.todense(), dtype=np.float64)
        return np.asarray(H, dtype=np.float64).copy()

    def _apply_single_gradient_step(
        self,
        H: np.ndarray,
        gradient: dict[str, Any],
        *,
        allow_beam: bool = False,
    ) -> dict[str, Any] | None:
        m, n = H.shape
        edge_scores = gradient["edge_scores"]
        node_instability = gradient["node_instability"]
        gradient_direction = gradient["gradient_direction"]

        predicted_node_instability: dict[int, float] = {}
        prediction: dict[str, Any] | None = None
        if self.avoid_predicted_trapping_sets:
            prediction = self._trapping_predictor.predict_trapping_regions(H)
            candidate_sets = prediction.get("candidate_sets", [])
            node_scores = prediction.get("node_scores", {})
            for candidate in candidate_sets:
                for node in candidate:
                    predicted_node_instability[int(node)] = float(node_scores.get(node, 0.0))

        check_neighbors, var_neighbors = self._build_neighbors(H)

        if self.avoid_predicted_trapping_sets:
            adjusted_edge_scores = {
                edge: float(score) / (1.0 + float(predicted_node_instability.get(edge[1], 0.0)))
                for edge, score in edge_scores.items()
            }
        else:
            adjusted_edge_scores = edge_scores

        if self.steer_spectral_basins:
            steering = self._basin_steering.compute_steering(H, prediction=prediction)
            steering_score = float(steering["steering_score"])
            adjusted_edge_scores = {
                edge: float(score) / (1.0 + steering_score)
                for edge, score in adjusted_edge_scores.items()
            }

        if allow_beam and self._is_beam_activation_state(H, prediction):
            plan = plan_two_step_swap(
                H,
                enumerate_candidates=lambda graph: self._enumerate_swap_candidates(
                    graph,
                    self._analyzer.compute_gradient(graph),
                ),
                apply_swap=self._apply_swap_to_copy,
                beam_width=self.beam_width,
                second_step_limit=self.second_step_limit,
                beam_score_weight=self.beam_score_weight,
            )
            if plan is not None:
                return self._commit_swap(H, plan["first_swap"], plan)

        candidates = self._enumerate_swap_candidates(
            H,
            gradient,
            adjusted_edge_scores=adjusted_edge_scores,
        )
        if not candidates:
            return None
        return self._commit_swap(H, candidates[0])

    def _is_beam_activation_state(
        self,
        H: np.ndarray,
        prediction: dict[str, Any] | None,
    ) -> bool:
        pred = prediction
        if pred is None:
            pred = self._trapping_predictor.predict_trapping_regions(H)
        ipr = float(pred.get("ipr", 0.0))
        risk = float(pred.get("risk_score", pred.get("trapping_risk", 0.0)))
        if ipr >= 0.2 and risk > 0.0:
            state = "localized_trap"
        elif ipr >= 0.1:
            state = "metastable_plateau"
        else:
            state = "free_descent"
        return state in self.beam_activation_states

    def _enumerate_swap_candidates(
        self,
        H: np.ndarray,
        gradient: dict[str, Any],
        *,
        adjusted_edge_scores: dict[tuple[int, int], float] | None = None,
    ) -> list[dict[str, Any]]:
        m, n = H.shape
        edge_scores = gradient["edge_scores"]
        node_instability = gradient["node_instability"]
        gradient_direction = gradient["gradient_direction"]
        if adjusted_edge_scores is None:
            adjusted_edge_scores = edge_scores
        check_neighbors, var_neighbors = self._build_neighbors(H)

        ranked_edges = sorted(adjusted_edge_scores, key=lambda e: (-adjusted_edge_scores[e], e[0], e[1]))

        all_candidates: list[dict[str, Any]] = []
        for ci, vi in ranked_edges:
            if H[ci, vi] == 0:
                continue

            base_grad = gradient_direction.get((ci, vi), 0.0)
            candidates: list[tuple[float, int, int]] = []
            for vj in range(n):
                if vj == vi or H[ci, vj] != 0:
                    continue

                grad_target = round(
                    float(node_instability[ci] - node_instability[m + vj]),
                    self.precision,
                )
                if not base_grad > grad_target:
                    continue

                partner = self._find_partner_check(
                    ci, vi, vj, check_neighbors, var_neighbors,
                )
                if partner is None:
                    continue

                candidates.append((grad_target, partner, vj))

            if not candidates:
                continue

            candidates.sort(key=lambda x: (x[0], x[1], x[2]))
            grad_target, cj, vj = candidates[0]

            candidate = {
                "removed_edge": (ci, vi),
                "added_edge": (ci, vj),
                "partner_removed": (cj, vj),
                "partner_added": (cj, vi),
                "source_gradient": round(float(base_grad), self.precision),
                "target_gradient": round(float(grad_target), self.precision),
                "score": round(float(grad_target - base_grad), self.precision),
                "swap_index": len(all_candidates),
                "remove": ((ci, vi), (cj, vj)),
                "add": ((ci, vj), (cj, vi)),
            }
            all_candidates.append(candidate)

        return all_candidates

    def _apply_swap_to_copy(self, H: np.ndarray, candidate: dict[str, Any]) -> np.ndarray:
        H_new = np.asarray(H, dtype=np.float64).copy()
        for ci, vi in candidate["remove"]:
            H_new[int(ci), int(vi)] = 0.0
        for ci, vi in candidate["add"]:
            H_new[int(ci), int(vi)] = 1.0
        return H_new

    def _commit_swap(
        self,
        H: np.ndarray,
        candidate: dict[str, Any],
        plan: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        for ci, vi in candidate["remove"]:
            H[int(ci), int(vi)] = 0.0
        for ci, vi in candidate["add"]:
            H[int(ci), int(vi)] = 1.0

        result = {
            "removed_edge": candidate["removed_edge"],
            "added_edge": candidate["added_edge"],
            "partner_removed": candidate["partner_removed"],
            "partner_added": candidate["partner_added"],
            "source_gradient": candidate["source_gradient"],
            "target_gradient": candidate["target_gradient"],
            "beam_search_used": plan is not None,
            "beam_width": self.beam_width if plan is not None else 0,
            "planned_sequence_score": (
                round(float(plan["planned_sequence_score"]), self.precision)
                if plan is not None
                else None
            ),
        }
        return result

    def _find_partner_check(
        self,
        ci: int,
        vi: int,
        vj: int,
        check_neighbors: dict[int, set[int]],
        var_neighbors: dict[int, set[int]],
    ) -> int | None:
        for cj in sorted(var_neighbors[vj]):
            if cj == ci:
                continue
            if vi in check_neighbors[cj]:
                continue
            if len(check_neighbors[cj]) <= 1 or len(var_neighbors[vi]) <= 1:
                continue
            if self.avoid_4cycles and self._creates_4cycle(
                ci, vj, cj, vi, var_neighbors,
            ):
                continue
            return cj
        return None

    @staticmethod
    def _build_neighbors(
        H: np.ndarray,
    ) -> tuple[dict[int, set[int]], dict[int, set[int]]]:
        m, n = H.shape
        check_neighbors: dict[int, set[int]] = {ci: set() for ci in range(m)}
        var_neighbors: dict[int, set[int]] = {vi: set() for vi in range(n)}

        coords = np.argwhere(H != 0)
        for ci, vi in coords:
            cii = int(ci)
            vii = int(vi)
            check_neighbors[cii].add(vii)
            var_neighbors[vii].add(cii)

        return check_neighbors, var_neighbors

    @staticmethod
    def _creates_4cycle(
        ci1: int,
        vi1: int,
        ci2: int,
        vi2: int,
        var_neighbors: dict[int, set[int]],
    ) -> bool:
        shared_checks = var_neighbors.get(vi1, set()).intersection(
            var_neighbors.get(vi2, set()),
        )
        shared_checks.discard(ci1)
        shared_checks.discard(ci2)
        return len(shared_checks) > 0
