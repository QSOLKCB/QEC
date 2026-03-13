"""v14.3.0 — Basin-aware deterministic controller for spectral graph descent."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import scipy.sparse as sp

from src.qec.analysis.basin_diagnostics import BasinDiagnostics, BasinDiagnosticsConfig
from src.qec.analysis.eigenmode_mutation import build_bethe_hessian, extract_unstable_modes
from src.qec.analysis.nonbacktracking_flow import NonBacktrackingFlowAnalyzer


@dataclass(frozen=True)
class BasinAwareFlowConfig:
    """Opt-in deterministic configuration for basin-aware control."""

    enabled: bool = False
    max_steps: int = 1000
    energy_window: int = 10
    slope_threshold: float = 1e-6
    ipr_threshold: float = 0.15
    trap_ipr_threshold: float = 0.3
    edge_reuse_threshold: float = 0.6
    escape_blacklist_size: int = 5
    escape_candidate_limit: int = 10
    eta_ipr: float = 0.1
    precision: int = 12


class BasinAwareSpectralFlow:
    """Wrap a local spectral descent engine with deterministic basin control."""

    def __init__(
        self,
        *,
        config: BasinAwareFlowConfig | None = None,
        descent_step: Callable[[np.ndarray], tuple[np.ndarray, dict[str, Any]]] | None = None,
        exploration_step: Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> None:
        self.config = config or BasinAwareFlowConfig()
        diag_cfg = BasinDiagnosticsConfig(
            energy_window=self.config.energy_window,
            slope_threshold=self.config.slope_threshold,
            ipr_threshold=self.config.ipr_threshold,
            trap_ipr_threshold=self.config.trap_ipr_threshold,
            edge_reuse_threshold=self.config.edge_reuse_threshold,
            reuse_window=self.config.energy_window,
            precision=self.config.precision,
        )
        self.diagnostics = BasinDiagnostics(config=diag_cfg)
        self._descent_step = descent_step
        self._exploration_step = exploration_step
        self._flow_analyzer = NonBacktrackingFlowAnalyzer()

    def run(self, H: np.ndarray | sp.spmatrix) -> dict[str, Any]:
        """Execute basin-aware spectral flow and return trajectory metadata."""
        H_curr = np.asarray(sp.csr_matrix(H, dtype=np.float64).toarray(), dtype=np.float64)
        history: list[dict[str, Any]] = []

        if not self.config.enabled:
            H_next, metadata = self._apply_descent(H_curr)
            metadata = dict(metadata)
            metadata.setdefault("basin_state", "free_descent")
            metadata.setdefault("trap_score", 0.0)
            metadata.setdefault("escape_triggered", False)
            history.append(metadata)
            return {"H": H_next, "trajectory": history}

        for _ in range(int(self.config.max_steps)):
            metrics = self._evaluate(H_curr)
            diag = self.diagnostics.update(
                energy=float(metrics["energy"]),
                unstable_modes=int(metrics["unstable_modes"]),
                flow=np.asarray(metrics["flow"], dtype=np.float64),
                hot_edges=list(metrics.get("hot_edges", [])),
            )
            basin_state = str(diag["basin_state"])
            escape_triggered = False

            if basin_state == "converged":
                history.append(
                    {
                        **metrics,
                        **diag,
                        "escape_triggered": False,
                    },
                )
                break

            if basin_state == "localized_trap":
                H_next, action_meta = self._escape_step(H_curr, metrics)
                escape_triggered = True
            elif basin_state == "metastable_plateau":
                H_next = self._apply_exploration(H_curr)
                action_meta = {"action": "exploration_swap"}
            else:
                H_next, action_meta = self._apply_descent(H_curr)

            history.append(
                {
                    **self._public_metrics(metrics),
                    **diag,
                    **action_meta,
                    "escape_triggered": escape_triggered,
                },
            )

            if np.array_equal(H_next, H_curr):
                break
            H_curr = H_next

        return {"H": H_curr, "trajectory": history}

    def _apply_descent(self, H: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        if self._descent_step is None:
            return H.copy(), {"action": "noop_descent"}
        H_next, metadata = self._descent_step(H.copy())
        return np.asarray(H_next, dtype=np.float64), dict(metadata)

    def _apply_exploration(self, H: np.ndarray) -> np.ndarray:
        if self._exploration_step is None:
            return H.copy()
        return np.asarray(self._exploration_step(H.copy()), dtype=np.float64)

    def _evaluate(self, H: np.ndarray) -> dict[str, Any]:
        energy = float(np.sum(H))
        B, _ = build_bethe_hessian(H)
        unstable_modes = len(extract_unstable_modes(B, num_modes=20))
        flow_result = self._flow_analyzer.compute_flow(H)
        flow = np.asarray(flow_result.get("edge_flow", np.zeros(0, dtype=np.float64)), dtype=np.float64)
        abs_flow = np.abs(flow)
        ranked_idx = list(np.argsort(-abs_flow, kind="stable"))
        matrix_edges = [(int(ci), int(vi)) for ci, vi in np.argwhere(H != 0)]
        matrix_edges.sort()
        hot_edges = [
            matrix_edges[idx]
            for idx in ranked_idx[: self.config.escape_blacklist_size]
            if idx < len(matrix_edges)
        ]
        return {
            "energy": round(energy, self.config.precision),
            "unstable_modes": int(unstable_modes),
            "flow": flow,
            "hot_edges": hot_edges,
            "candidates": self._enumerate_swap_candidates(H),
        }

    @staticmethod
    def _public_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
        out = {k: v for k, v in metrics.items() if k not in {"flow", "candidates"}}
        out["hot_edges"] = [tuple(edge) for edge in out.get("hot_edges", [])]
        return out

    def _escape_step(self, H: np.ndarray, metrics: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
        candidates = list(metrics.get("candidates", []))
        if not candidates:
            return H.copy(), {"action": "escape_noop"}

        blacklist = {
            edge
            for edge in sorted(metrics.get("hot_edges", []))[: self.config.escape_blacklist_size]
        }
        limited = self._rank_candidates(H, candidates, blacklist)[: self.config.escape_candidate_limit]
        if not limited:
            return H.copy(), {"action": "escape_noop"}

        best = limited[0]
        H_next = self._apply_swap(H, best["swap"])
        return H_next, {
            "action": "escape_swap",
            "escape_swap": best["swap"],
            "escape_score": best["score"],
        }

    def _rank_candidates(
        self,
        H: np.ndarray,
        candidates: list[tuple[int, int, int, int]],
        blacklist: set[tuple[int, int]],
    ) -> list[dict[str, Any]]:
        baseline = self._evaluate_ipr(H)
        ranked: list[dict[str, Any]] = []
        for ci, vi, cj, vj in sorted(candidates):
            if (ci, vi) in blacklist or (cj, vj) in blacklist:
                continue
            H_trial = self._apply_swap(H, (ci, vi, cj, vj))
            if not self._is_valid_swap(H, H_trial):
                continue
            delta_flow = float(np.sum(H) - np.sum(H_trial))
            ipr_after = self._evaluate_ipr(H_trial)
            score = round(
                float(delta_flow + self.config.eta_ipr * (ipr_after - baseline)),
                self.config.precision,
            )
            ranked.append(
                {
                    "swap": (ci, vi, cj, vj),
                    "score": score,
                    "ipr_after": ipr_after,
                },
            )

        ranked.sort(key=lambda item: (item["score"], item["swap"]))
        return ranked

    def _evaluate_ipr(self, H: np.ndarray) -> float:
        flow = self._flow_analyzer.compute_flow(H)
        edge_flow = np.asarray(flow.get("edge_flow", np.zeros(0, dtype=np.float64)), dtype=np.float64)
        return self.diagnostics.compute_flow_ipr(edge_flow)

    @staticmethod
    def _enumerate_swap_candidates(H: np.ndarray) -> list[tuple[int, int, int, int]]:
        m, n = H.shape
        edges = [(int(ci), int(vi)) for ci, vi in np.argwhere(H != 0)]
        edges.sort()
        out: list[tuple[int, int, int, int]] = []

        var_neighbors: dict[int, list[int]] = {vi: [] for vi in range(n)}
        check_neighbors: dict[int, list[int]] = {ci: [] for ci in range(m)}
        for ci, vi in edges:
            var_neighbors[vi].append(ci)
            check_neighbors[ci].append(vi)

        for vi in var_neighbors:
            var_neighbors[vi] = sorted(var_neighbors[vi])

        for ci, vi in edges:
            for vj in range(n):
                if vj == vi or H[ci, vj] != 0:
                    continue
                for cj in var_neighbors[vj]:
                    if cj == ci:
                        continue
                    if H[cj, vi] != 0:
                        continue
                    out.append((ci, vi, cj, vj))

        out.sort()
        return out

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
    def _is_valid_swap(H_before: np.ndarray, H_after: np.ndarray) -> bool:
        if H_before.shape != H_after.shape:
            return False
        if not np.all((H_after == 0.0) | (H_after == 1.0)):
            return False
        if not np.array_equal(np.sum(H_before, axis=0), np.sum(H_after, axis=0)):
            return False
        if not np.array_equal(np.sum(H_before, axis=1), np.sum(H_after, axis=1)):
            return False
        return True


__all__ = ["BasinAwareFlowConfig", "BasinAwareSpectralFlow"]
