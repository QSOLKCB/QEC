"""v16.1.0 — Adaptive mutation controller for basin-aware discovery."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

import numpy as np
import scipy.sparse

from qec.analysis.basin_diagnostics import BasinDiagnostics, BasinDiagnosticsConfig
from qec.analysis.nonbacktracking_flow import NonBacktrackingFlowAnalyzer
from qec.analysis.nb_instability_gradient import NBInstabilityGradientAnalyzer
if TYPE_CHECKING:
    from qec.discovery.mutation_nb_gradient import NBGradientMutator
else:
    NBGradientMutator = Any


@dataclass(frozen=True)
class AdaptiveMutationConfig:
    """Configuration for deterministic adaptive mutation control."""

    enabled: bool = False

    use_nb_flow: bool = True
    use_beam_search: bool = True

    trap_beam_width: int = 8
    plateau_beam_width: int = 4

    flow_iterations: int = 1
    beam_iterations: int = 1

    precision: int = 12


class NonBacktrackingFlowMutator:
    """Deterministic NB-flow mutation wrapper."""

    def __init__(self, *, precision: int = 12) -> None:
        self.precision = int(precision)
        from qec.discovery.mutation_nb_gradient import NBGradientMutator as _NBGradientMutator

        self._mutator = _NBGradientMutator(enabled=True, precision=self.precision)

    def mutate(
        self,
        H: np.ndarray | scipy.sparse.spmatrix,
        *,
        iterations: int,
    ) -> tuple[np.ndarray, list[dict[str, Any]]]:
        steps = max(0, int(iterations))
        return self._mutator.mutate_flow(H, iterations=steps)


class AdaptiveMutationController:
    """Switch mutation strategy deterministically based on basin state."""

    def __init__(
        self,
        *,
        config: AdaptiveMutationConfig | None = None,
        basin_diagnostics: BasinDiagnostics | None = None,
        flow_mutator: Any | None = None,
        beam_mutator: Any | None = None,
    ) -> None:
        self.config = config or AdaptiveMutationConfig()
        self.basin_diagnostics = basin_diagnostics or BasinDiagnostics(
            config=BasinDiagnosticsConfig(precision=self.config.precision),
        )
        self.flow_mutator = flow_mutator or NonBacktrackingFlowMutator(
            precision=self.config.precision,
        )
        if beam_mutator is None:
            from qec.discovery.mutation_nb_gradient import NBGradientMutator as _NBGradientMutator

            self.beam_mutator = _NBGradientMutator(
                enabled=True,
                enable_spectral_beam_search=True,
                beam_width=max(1, int(self.config.trap_beam_width)),
                beam_activation_states=("free_descent", "metastable_plateau", "localized_trap"),
                precision=self.config.precision,
            )
        else:
            self.beam_mutator = beam_mutator
        self._flow_analyzer = NonBacktrackingFlowAnalyzer()
        self._gradient_analyzer = NBInstabilityGradientAnalyzer()

    def mutate(
        self,
        H: np.ndarray | scipy.sparse.spmatrix,
        *,
        max_iterations: int = 1,
    ) -> tuple[np.ndarray, list[dict[str, Any]]]:
        """Run basin-aware mutation loop and return trajectory metadata."""
        H_curr = np.asarray(scipy.sparse.csr_matrix(H, dtype=np.float64).toarray(), dtype=np.float64)
        if not self.config.enabled or max_iterations <= 0:
            return H_curr, []

        trajectory: list[dict[str, Any]] = []
        for iteration in range(int(max_iterations)):
            state, signal = self._classify_state(H_curr)
            if state == "converged":
                trajectory.append(
                    {
                        "iteration": int(iteration),
                        "mutation_strategy": "terminated",
                        "basin_state": "converged",
                        "beam_width": 0,
                        "flow_mode_index": 0,
                    },
                )
                break

            if state in {"free_descent", "metastable_plateau"} and self.config.use_nb_flow:
                H_next, step_log = self.flow_mutator.mutate(
                    H_curr,
                    iterations=self.config.flow_iterations,
                )
                strategy = "nb_flow"
                beam_width = 0
            elif state == "localized_trap" and self.config.use_beam_search:
                beam_width = self._beam_width_for_state(state)
                if hasattr(self.beam_mutator, "beam_width"):
                    self.beam_mutator.beam_width = int(beam_width)
                H_next, step_log = self.beam_mutator.mutate(
                    H_curr,
                    steps=self.config.beam_iterations,
                )
                strategy = "beam_search"
            else:
                H_next = H_curr.copy()
                step_log = []
                strategy = "noop"
                beam_width = 0

            flow_mode_index = self._extract_flow_mode_index(step_log)
            trajectory.append(
                {
                    "iteration": int(iteration),
                    "mutation_strategy": strategy,
                    "basin_state": state,
                    "beam_width": int(beam_width),
                    "flow_mode_index": int(flow_mode_index),
                    "signal": signal,
                },
            )

            H_next_arr = np.asarray(H_next, dtype=np.float64)
            if np.array_equal(H_next_arr, H_curr):
                break
            H_curr = H_next_arr

        return H_curr, trajectory

    def _classify_state(self, H: np.ndarray) -> tuple[str, dict[str, Any]]:
        if hasattr(self.basin_diagnostics, "classify_state"):
            result = self.basin_diagnostics.classify_state(H)
            if isinstance(result, dict):
                return str(result.get("basin_state", "free_descent")), dict(result)
            return str(result), {}

        metrics = self._diagnostic_signals(H)
        result = self.basin_diagnostics.update(**metrics)
        return str(result.get("basin_state", "free_descent")), dict(result)

    def _diagnostic_signals(self, H: np.ndarray) -> dict[str, Any]:
        energy = round(float(np.sum(H, dtype=np.float64)), self.config.precision)
        gradient = self._gradient_analyzer.compute_gradient(H)
        unstable_modes = int(sum(1 for v in gradient["edge_scores"].values() if float(v) > 0.0))

        flow_data = self._flow_analyzer.compute_flow(H)
        edge_flow = np.asarray(flow_data.get("edge_flow", np.zeros(0, dtype=np.float64)), dtype=np.float64)

        edges = [(int(ci), int(vi)) for ci, vi in np.argwhere(H != 0)]
        edges.sort()
        abs_flow = np.abs(edge_flow)
        ranked_indices = list(np.argsort(-abs_flow, kind="stable"))
        hot_edges = [edges[idx] for idx in ranked_indices if idx < len(edges)]

        return {
            "energy": energy,
            "unstable_modes": unstable_modes,
            "flow": edge_flow,
            "hot_edges": hot_edges,
        }

    def _beam_width_for_state(self, basin_state: str) -> int:
        if basin_state == "localized_trap":
            return max(1, int(self.config.trap_beam_width))
        return max(1, int(self.config.plateau_beam_width))

    @staticmethod
    def _extract_flow_mode_index(step_log: list[dict[str, Any]]) -> int:
        if not step_log:
            return 0
        last = step_log[-1]
        return int(last.get("flow_mode_index", 0))


__all__ = [
    "AdaptiveMutationConfig",
    "AdaptiveMutationController",
    "NBGradientMutator",
    "NonBacktrackingFlowMutator",
]
