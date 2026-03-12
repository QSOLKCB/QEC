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


_ROUND = 12


class NBGradientMutator:
    """Deterministic instability-gradient guided mutation operator."""

    def __init__(
        self,
        *,
        enabled: bool = False,
        avoid_4cycles: bool = True,
        flow_damping: bool = False,
        precision: int = _ROUND,
    ) -> None:
        self.enabled = enabled
        self.avoid_4cycles = avoid_4cycles
        self.flow_damping = flow_damping
        self.precision = precision
        self._analyzer = NBInstabilityGradientAnalyzer()

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
                    prev_val = prev_direction.get(edge, value)
                    damped_direction[edge] = round(
                        0.5 * float(value) + 0.5 * float(prev_val),
                        self.precision,
                    )
                gradient["gradient_direction"] = damped_direction

            step = self._apply_single_gradient_step(H_new, gradient)
            if step is None:
                break
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
    ) -> dict[str, Any] | None:
        m, n = H.shape
        edge_scores = gradient["edge_scores"]
        node_instability = gradient["node_instability"]
        gradient_direction = gradient["gradient_direction"]

        check_neighbors, var_neighbors = self._build_neighbors(H)

        ranked_edges = sorted(
            edge_scores,
            key=lambda e: (-edge_scores[e], e[0], e[1]),
        )

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

            H[ci, vi] = 0.0
            H[cj, vj] = 0.0
            H[ci, vj] = 1.0
            H[cj, vi] = 1.0

            return {
                "removed_edge": (ci, vi),
                "added_edge": (ci, vj),
                "partner_removed": (cj, vj),
                "partner_added": (cj, vi),
                "source_gradient": round(float(base_grad), self.precision),
                "target_gradient": round(float(grad_target), self.precision),
            }

        return None

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
