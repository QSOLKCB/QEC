"""v14.3.0 — Deterministic basin diagnostics for spectral-flow trajectories."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


_BASIN_STATES = (
    "free_descent",
    "metastable_plateau",
    "localized_trap",
    "converged",
)


@dataclass(frozen=True)
class BasinDiagnosticsConfig:
    """Deterministic configuration for basin-state classification."""

    energy_window: int = 10
    slope_threshold: float = 1e-6
    ipr_threshold: float = 0.15
    trap_ipr_threshold: float = 0.3
    edge_reuse_threshold: float = 0.6
    reuse_window: int = 10
    precision: int = 12


@dataclass
class BasinDiagnostics:
    """Track rolling trajectory diagnostics and classify basin states."""

    config: BasinDiagnosticsConfig = field(default_factory=BasinDiagnosticsConfig)
    energy_trace: list[float] = field(default_factory=list)
    unstable_modes_trace: list[int] = field(default_factory=list)
    recent_hot_edges: list[tuple[int, int]] = field(default_factory=list)

    def update(
        self,
        *,
        energy: float,
        unstable_modes: int,
        flow: np.ndarray,
        hot_edges: list[tuple[int, int]] | None = None,
    ) -> dict[str, float | int | str]:
        """Update trajectory signals and return deterministic state metadata."""
        self.energy_trace.append(round(float(energy), self.config.precision))
        self.unstable_modes_trace.append(int(unstable_modes))

        if hot_edges:
            ordered_edges = sorted((int(ci), int(vi)) for ci, vi in hot_edges)
            self.recent_hot_edges.extend(ordered_edges)

        window = int(max(1, self.config.reuse_window))
        if len(self.recent_hot_edges) > window:
            self.recent_hot_edges = self.recent_hot_edges[-window:]

        slope = self.compute_energy_slope()
        flow_ipr = self.compute_flow_ipr(flow)
        edge_reuse_rate = self.compute_edge_reuse_rate()
        unstable_persistence = self.compute_unstable_mode_persistence()
        state = self.classify(
            unstable_modes=unstable_modes,
            slope=slope,
            flow_ipr=flow_ipr,
            edge_reuse_rate=edge_reuse_rate,
        )

        trap_score = round(
            float(max(0.0, flow_ipr - self.config.trap_ipr_threshold) + edge_reuse_rate + unstable_persistence),
            self.config.precision,
        )

        return {
            "basin_state": state,
            "energy_slope": slope,
            "flow_ipr": flow_ipr,
            "edge_reuse_rate": edge_reuse_rate,
            "unstable_mode_persistence": unstable_persistence,
            "trap_score": trap_score,
        }

    def compute_energy_slope(self) -> float:
        """Return rolling finite-difference slope over configured energy window."""
        if len(self.energy_trace) < 2:
            return 0.0
        k = min(int(self.config.energy_window), len(self.energy_trace) - 1)
        if k <= 0:
            return 0.0
        slope = (self.energy_trace[-1] - self.energy_trace[-1 - k]) / float(k)
        return round(float(slope), self.config.precision)

    def compute_flow_ipr(self, flow: np.ndarray) -> float:
        """Compute flow localization from absolute directed-edge flow values."""
        flow_arr = np.asarray(flow, dtype=np.float64)
        if flow_arr.size == 0:
            return 0.0
        abs_flow = np.abs(flow_arr)
        denom = float(abs_flow.sum())
        if denom <= 0.0:
            return 0.0
        ipr = float(np.sum(abs_flow * abs_flow)) / float(denom * denom)
        return round(ipr, self.config.precision)

    def compute_edge_reuse_rate(self) -> float:
        """Fraction of recent hot-edge touches that repeat prior edge choices."""
        if not self.recent_hot_edges:
            return 0.0
        total = float(len(self.recent_hot_edges))
        unique = float(len(set(self.recent_hot_edges)))
        rate = 1.0 - (unique / total)
        return round(rate, self.config.precision)

    def compute_unstable_mode_persistence(self) -> float:
        """Persistence ratio of unstable-mode presence over recent trajectory."""
        if not self.unstable_modes_trace:
            return 0.0
        window = min(len(self.unstable_modes_trace), int(max(1, self.config.energy_window)))
        recent = self.unstable_modes_trace[-window:]
        active = sum(1 for count in recent if int(count) > 0)
        return round(float(active) / float(window), self.config.precision)

    def classify(
        self,
        *,
        unstable_modes: int,
        slope: float,
        flow_ipr: float,
        edge_reuse_rate: float,
    ) -> str:
        """Classify basin regime using deterministic threshold logic."""
        if int(unstable_modes) == 0:
            return "converged"
        if abs(float(slope)) < float(self.config.slope_threshold) and flow_ipr < float(self.config.ipr_threshold):
            return "metastable_plateau"
        if (
            flow_ipr >= float(self.config.trap_ipr_threshold)
            and edge_reuse_rate > float(self.config.edge_reuse_threshold)
        ):
            return "localized_trap"
        return "free_descent"


__all__ = [
    "BasinDiagnostics",
    "BasinDiagnosticsConfig",
    "_BASIN_STATES",
]
