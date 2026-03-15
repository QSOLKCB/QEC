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

from src.qec.analysis.basin_depth import BasinDepthConfig, compute_basin_depth
from src.qec.analysis.nb_instability_gradient import NBInstabilityGradientAnalyzer
from src.qec.analysis.nb_spectral_basin_steering import NBSpectralBasinSteering
from src.qec.analysis.nb_trapping_set_predictor import NBTrappingSetPredictor
from src.qec.analysis.nonbacktracking_flow import NonBacktrackingFlowAnalyzer
from src.qec.analysis.spectral_frustration import SpectralFrustrationAnalyzer
from src.qec.analysis.trap_memory import TrapMemory, TrapMemoryConfig
from src.qec.discovery.spectral_beam_search import adaptive_beam_width, plan_two_step_swap


_ROUND = 12
MAX_TRAP_MODES = 32


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
        adaptive_beam: bool = False,
        beam_min: int = 3,
        beam_max: int = 10,
        depth_scale: float = 3.0,
        beam_diversity: bool = False,
        nb_alignment_bias: bool = False,
        eta_nb: float = 0.2,
        frustration_guided: bool = False,
        eta_frustration: float = 0.25,
        frustration_eval_limit: int = 8,
        track_trap_modes: bool = False,
        trap_memory_config: TrapMemoryConfig | None = None,
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
        self.adaptive_beam = bool(adaptive_beam)
        self.beam_min = max(1, int(beam_min))
        self.beam_max = max(self.beam_min, int(beam_max))
        self.depth_scale = float(depth_scale)
        self.beam_diversity = bool(beam_diversity)
        self.nb_alignment_bias = bool(nb_alignment_bias)
        self.eta_nb = float(eta_nb)
        self.frustration_guided = bool(frustration_guided)
        self.eta_frustration = float(eta_frustration)
        self.frustration_eval_limit = max(1, int(frustration_eval_limit))
        self.track_trap_modes = bool(track_trap_modes)
        self.trap_memory_config = trap_memory_config or TrapMemoryConfig()
        self.precision = precision
        self._analyzer = NBInstabilityGradientAnalyzer()
        self._trapping_predictor = NBTrappingSetPredictor(precision=precision)
        self._basin_steering = NBSpectralBasinSteering(precision=precision)
        self._nb_flow = NonBacktrackingFlowAnalyzer()
        self._frustration = SpectralFrustrationAnalyzer(precision=precision)
        self._trap_modes: list[np.ndarray] = []
        self._trap_memory = TrapMemory(max_traps=self.trap_memory_config.max_traps)
        self._energy_deltas: list[float] = []
        self._basin_depth_config = BasinDepthConfig(precision=precision)

    def mutate(
        self,
        H: np.ndarray | scipy.sparse.spmatrix,
        steps: int = 5,
    ) -> tuple[np.ndarray, list[dict[str, Any]]]:
        """Apply up to ``steps`` deterministic gradient rewiring swaps."""
        H_new = self._to_dense_copy(H)
        if not self.enabled or steps <= 0:
            return H_new, []

        self._energy_deltas = []
        if self.trap_memory_config.enabled:
            self._trap_memory = TrapMemory(max_traps=self.trap_memory_config.max_traps)
        mutations: list[dict[str, Any]] = []
        for _ in range(steps):
            gradient = self._analyzer.compute_gradient(H_new)
            step = self._apply_single_gradient_step(H_new, gradient)
            if step is None:
                break
            step.setdefault("beam_search_used", False)
            step.setdefault("beam_width", 0)
            step.setdefault("planned_sequence_score", None)
            step.setdefault("planner_depth", 1)
            step.setdefault("basin_depth", 0.0)
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

        self._energy_deltas = []
        if self.trap_memory_config.enabled:
            self._trap_memory = TrapMemory(max_traps=self.trap_memory_config.max_traps)
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
            step.setdefault("planner_depth", 1)
            step.setdefault("basin_depth", 0.0)
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

        flow_for_bias = self._nb_flow.compute_flow(H) if self.nb_alignment_bias else None
        nb_alignment_map = self._compute_nb_alignment_map(
            H,
            flow_for_bias,
        ) if self.nb_alignment_bias else {}

        candidates = self._enumerate_swap_candidates(
            H,
            gradient,
            adjusted_edge_scores=adjusted_edge_scores,
            nb_alignment_map=nb_alignment_map,
        )
        if self.frustration_guided and candidates:
            self._apply_frustration_guidance(H, candidates)

        if allow_beam and self._is_beam_activation_state(H, prediction):
            basin_depth = self._compute_current_basin_depth(H, prediction, flow_for_bias)
            assert np.isfinite(basin_depth)
            effective_beam_width = self.beam_width
            if self.adaptive_beam:
                effective_beam_width = adaptive_beam_width(
                    basin_depth=float(basin_depth),
                    beam_min=self.beam_min,
                    beam_max=self.beam_max,
                    depth_scale=self.depth_scale,
                )
                assert effective_beam_width >= self.beam_min
                assert effective_beam_width <= self.beam_max
            plan = plan_two_step_swap(
                H,
                enumerate_candidates=lambda graph: self._enumerate_swap_candidates(
                    graph,
                    self._analyzer.compute_gradient(graph),
                    nb_alignment_map=(
                        self._compute_nb_alignment_map(graph, self._nb_flow.compute_flow(graph))
                        if self.nb_alignment_bias
                        else {}
                    ),
                ),
                apply_swap=self._apply_swap_to_copy,
                beam_width=effective_beam_width,
                beam_diversity=self.beam_diversity,
                second_step_limit=self.second_step_limit,
                beam_score_weight=self.beam_score_weight,
            )
            if plan is not None:
                return self._commit_swap(
                    H,
                    plan["first_swap"],
                    plan,
                    beam_width_used=effective_beam_width,
                    basin_depth=basin_depth,
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
        nb_alignment_map: dict[tuple[int, int], float] | None = None,
    ) -> list[dict[str, Any]]:
        m, n = H.shape
        edge_scores = gradient["edge_scores"]
        node_instability = gradient["node_instability"]
        gradient_direction = gradient["gradient_direction"]
        if adjusted_edge_scores is None:
            adjusted_edge_scores = edge_scores
        if nb_alignment_map is None:
            nb_alignment_map = {}
        check_neighbors, var_neighbors = self._build_neighbors(H)
        var_set = set(range(n))
        non_neighbors = {
            ci: sorted(var_set - check_neighbors[ci])
            for ci in range(m)
        }

        ranked_edges = sorted(adjusted_edge_scores, key=lambda e: (-adjusted_edge_scores[e], e[0], e[1]))

        all_candidates: list[dict[str, Any]] = []
        for ci, vi in ranked_edges:
            if H[ci, vi] == 0:
                continue

            base_grad = gradient_direction.get((ci, vi), 0.0)
            candidates: list[tuple[float, int, int]] = []
            for vj in non_neighbors[ci]:
                if vj == vi:
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
            alignment = (
                nb_alignment_map.get((ci, vi), 0.0)
                + nb_alignment_map.get((cj, vj), 0.0)
            )

            candidate = {
                "removed_edge": (ci, vi),
                "added_edge": (ci, vj),
                "partner_removed": (cj, vj),
                "partner_added": (cj, vi),
                "source_gradient": round(float(base_grad), self.precision),
                "target_gradient": round(float(grad_target), self.precision),
                "alignment": round(float(alignment), self.precision),
                "score": round(
                    float(grad_target - base_grad) - (float(self.eta_nb) * float(alignment)),
                    self.precision,
                ),
                "swap_index": len(all_candidates),
                "remove": ((ci, vi), (cj, vj)),
                "add": ((ci, vj), (cj, vi)),
            }
            all_candidates.append(candidate)

        return all_candidates

    def _apply_frustration_guidance(self, H: np.ndarray, candidates: list[dict[str, Any]]) -> None:
        base = self._frustration.compute_frustration(H)
        if self.track_trap_modes:
            self._trap_modes = [np.asarray(mode, dtype=np.float64).copy() for mode in base.trap_modes]
        if self.trap_memory_config.enabled and self._is_strong_trap(base.max_ipr, base.trap_modes):
            base_vec = self._select_trap_vector(base.trap_modes)
            if base_vec is not None:
                self._trap_memory.add(base_vec)

        ranked = sorted(candidates, key=lambda c: (float(c["score"]), int(c["swap_index"])))
        eval_top_k = min(len(ranked), int(self.frustration_eval_limit))
        evaluated: set[int] = set()
        for cand in ranked[:eval_top_k]:
            H_swap = self._apply_swap_to_copy(H, cand)
            nxt = self._frustration.compute_frustration(H_swap)
            delta_frustration = round(float(nxt.frustration_score - base.frustration_score), self.precision)
            cand["frustration_before"] = round(float(base.frustration_score), self.precision)
            cand["frustration_after"] = round(float(nxt.frustration_score), self.precision)
            cand["delta_frustration"] = delta_frustration
            cand["negative_modes"] = int(nxt.negative_modes)
            cand["max_ipr"] = round(float(nxt.max_ipr), self.precision)
            trap_similarity = 0.0
            trap_penalty = 0.0
            if self.trap_memory_config.enabled:
                trap_vec = self._select_trap_vector(nxt.trap_modes)
                if trap_vec is not None:
                    trap_similarity = self._trap_memory.similarity(trap_vec)
                    trap_penalty = float(self.trap_memory_config.eta_trap) * float(trap_similarity)
                    cand["_trap_vector"] = trap_vec
            cand["trap_similarity"] = round(float(trap_similarity), self.precision)
            cand["trap_penalty"] = round(float(trap_penalty), self.precision)
            cand["trap_memory_size"] = int(len(self._trap_memory.trap_vectors))
            cand["score"] = round(
                float(cand["score"])
                + float(self.eta_frustration) * delta_frustration
                - float(trap_penalty),
                self.precision,
            )
            evaluated.add(int(cand["swap_index"]))

        base_score = round(float(base.frustration_score), self.precision)
        for cand in candidates:
            if int(cand["swap_index"]) in evaluated:
                continue
            cand["frustration_before"] = base_score
            cand["frustration_after"] = base_score
            cand["delta_frustration"] = 0.0
            cand["negative_modes"] = int(base.negative_modes)
            cand["max_ipr"] = round(float(base.max_ipr), self.precision)
            cand["trap_similarity"] = 0.0
            cand["trap_penalty"] = 0.0
            cand["trap_memory_size"] = int(len(self._trap_memory.trap_vectors)) if self.trap_memory_config.enabled else 0

        candidates.sort(key=lambda c: (float(c["score"]), int(c["swap_index"])))

    def _compute_nb_alignment_map(
        self,
        H: np.ndarray,
        flow: dict[str, Any],
    ) -> dict[tuple[int, int], float]:
        m, n = H.shape
        directed_edges = flow.get("directed_edges", [])
        directed_flow = np.asarray(flow.get("directed_edge_flow", np.zeros(0, dtype=np.float64)), dtype=np.float64)
        if not directed_edges or directed_flow.size == 0:
            return {}

        edge_idx = {edge: i for i, edge in enumerate(directed_edges)}
        alignment: dict[tuple[int, int], float] = {}
        for ci, vi in sorted((int(a), int(b)) for a, b in np.argwhere(H != 0)):
            idx_fwd = edge_idx.get((vi, n + ci))
            idx_rev = edge_idx.get((n + ci, vi))
            p_fwd = float(directed_flow[idx_fwd]) if idx_fwd is not None else 0.0
            p_rev = float(directed_flow[idx_rev]) if idx_rev is not None else 0.0
            alignment[(ci, vi)] = round(abs(p_fwd - p_rev), self.precision)
        return alignment

    def _compute_current_basin_depth(
        self,
        H: np.ndarray,
        prediction: dict[str, Any] | None,
        flow_for_bias: dict[str, Any] | None,
    ) -> float:
        pred = prediction if prediction is not None else self._trapping_predictor.predict_trapping_regions(H)
        flow = flow_for_bias if flow_for_bias is not None else self._nb_flow.compute_flow(H)
        flow_ipr = float(pred.get("ipr", 0.0))

        edge_flow = np.asarray(flow.get("edge_flow", np.zeros(0, dtype=np.float64)), dtype=np.float64)
        edge_reuse_rate = 0.0
        if edge_flow.size > 0:
            abs_flow = np.abs(edge_flow)
            denom = float(abs_flow.sum())
            if denom > 0.0:
                edge_reuse_rate = float(abs_flow.max()) / denom

        unstable_mode_persistence = 1.0 if float(pred.get("risk_score", pred.get("trapping_risk", 0.0))) > 0.0 else 0.0
        variable_flow = np.asarray(list(flow.get("variable_flow", [])), dtype=np.float64)
        if variable_flow.size == 0:
            energy_like = 0.0
        else:
            energy_like = float(np.mean(np.abs(variable_flow)))
        if not np.isfinite(energy_like):
            energy_like = 0.0
        delta = round(float(energy_like), self.precision)
        if not np.isfinite(delta):
            delta = 0.0
        self._energy_deltas.append(delta)
        if len(self._energy_deltas) > 64:
            self._energy_deltas = self._energy_deltas[-64:]

        depth = compute_basin_depth(
            flow_ipr=flow_ipr,
            edge_reuse_rate=edge_reuse_rate,
            unstable_mode_persistence=unstable_mode_persistence,
            energy_deltas=self._energy_deltas,
            config=self._basin_depth_config,
        )
        return float(depth["basin_depth"])

    def _apply_swap_to_copy(self, H: np.ndarray, candidate: dict[str, Any]) -> np.ndarray:
        if isinstance(H, np.ndarray) and H.dtype == np.float64:
            H_new = H.copy()
        else:
            H_new = np.array(H, dtype=np.float64, copy=True)
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
        *,
        beam_width_used: int = 0,
        basin_depth: float = 0.0,
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
            "frustration_before": round(float(candidate.get("frustration_before", 0.0)), self.precision),
            "frustration_after": round(float(candidate.get("frustration_after", 0.0)), self.precision),
            "delta_frustration": round(float(candidate.get("delta_frustration", 0.0)), self.precision),
            "negative_modes": int(candidate.get("negative_modes", 0)),
            "max_ipr": round(float(candidate.get("max_ipr", 0.0)), self.precision),
            "trap_similarity": round(float(candidate.get("trap_similarity", 0.0)), self.precision),
            "trap_penalty": round(float(candidate.get("trap_penalty", 0.0)), self.precision),
            "trap_memory_size": int(candidate.get("trap_memory_size", 0)),
            "beam_search_used": plan is not None,
            "beam_width": int(beam_width_used) if plan is not None else 0,
            "planned_sequence_score": (
                round(float(plan["planned_sequence_score"]), self.precision)
                if plan is not None
                else None
            ),
            "planner_depth": 2 if plan is not None else 1,
            "basin_depth": round(float(basin_depth), self.precision) if plan is not None else 0.0,
        }
        if self.trap_memory_config.enabled and "_trap_vector" in candidate:
            self._trap_memory.add(np.asarray(candidate["_trap_vector"], dtype=np.float64))
        return result

    @staticmethod
    def _is_strong_trap(max_ipr: float, trap_modes: tuple[np.ndarray, ...]) -> bool:
        return bool(trap_modes) or float(max_ipr) >= 0.2

    @staticmethod
    def _select_trap_vector(trap_modes: tuple[np.ndarray, ...]) -> np.ndarray | None:
        if not trap_modes:
            return None
        vectors: list[np.ndarray] = []
        for mode in trap_modes:
            vec = TrapMemory.canonicalize(np.asarray(mode, dtype=np.float64))
            if vec.size > 0 and float(np.linalg.norm(vec)) > 0.0:
                vectors.append(vec)
        if not vectors:
            return None
        vectors.sort(key=lambda vec: tuple(float(x) for x in vec.tolist()))
        return vectors[0]

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
