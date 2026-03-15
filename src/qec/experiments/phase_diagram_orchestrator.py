"""Deterministic phase-diagram orchestration for experiment grids."""

from __future__ import annotations

import json
from itertools import product
from pathlib import Path
from typing import Any, Callable

from .experiment_hash import ExperimentHash, ExperimentRunner


MetricExecutor = Callable[[dict[str, Any]], dict[str, Any]]


class PhaseDiagramOrchestrator:
    """Run deterministic 2-D sweeps and build phase-diagram artifacts."""

    METRIC_KEYS = (
        "bp_converged",
        "spectral_radius",
        "mean_llr",
        "decoder_success_rate",
    )

    def __init__(
        self,
        experiment_runner: ExperimentRunner | None = None,
        experiment_registry: dict[str, MetricExecutor] | None = None,
        artifacts_root: str | Path = "experiments",
    ) -> None:
        self.artifacts_root = Path(artifacts_root)
        self.runner = experiment_runner or ExperimentRunner(artifacts_root=self.artifacts_root)
        self.experiment_registry = dict(experiment_registry or {})

    def generate_grid(self, config: dict[str, Any]) -> list[tuple[Any, Any]]:
        """Generate deterministic (x, y) grid points with stable ordering."""
        x_values = list(config["x_axis"]["values"])
        y_values = list(config["y_axis"]["values"])
        return list(product(x_values, y_values))

    def run(self, config: dict[str, Any]) -> dict[str, Any]:
        """Execute the full phase-diagram sweep and write a JSON artifact."""
        experiment_name = str(config["experiment"])
        if experiment_name not in self.experiment_registry:
            known = ", ".join(sorted(self.experiment_registry)) or "<none>"
            raise ValueError(f"unknown experiment '{experiment_name}'. available: {known}")

        x_name = str(config["x_axis"]["name"])
        y_name = str(config["y_axis"]["name"])
        x_values = list(config["x_axis"]["values"])
        y_values = list(config["y_axis"]["values"])

        dataset_grid: list[list[dict[str, Any]]] = []
        statuses: list[str] = []

        for x_value in x_values:
            row: list[dict[str, Any]] = []
            for y_value in y_values:
                point_config = self._build_point_config(config, x_name, x_value, y_name, y_value)
                statuses.append(self._run_status(point_config))
                result = self.runner.run(point_config, self.experiment_registry[experiment_name])
                row.append(self._extract_metrics(result))
            dataset_grid.append(row)

        payload = {
            "experiment": experiment_name,
            "x_axis": x_name,
            "y_axis": y_name,
            "x_values": x_values,
            "y_values": y_values,
            "grid": dataset_grid,
            "heatmap": generate_stability_heatmap({"grid": dataset_grid}),
        }

        artifact_dir = self._artifact_dir(config)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = artifact_dir / "phase_diagram.json"
        artifact_path.write_text(f"{json.dumps(payload, sort_keys=True, indent=2)}\n", encoding="utf-8")

        return {
            "artifact": str(artifact_path),
            "grid_size": [len(x_values), len(y_values)],
            "statuses": statuses,
            "phase_diagram": payload,
        }

    def _build_point_config(
        self,
        config: dict[str, Any],
        x_name: str,
        x_value: Any,
        y_name: str,
        y_value: Any,
    ) -> dict[str, Any]:
        base_config = dict(config.get("experiment_config", {}))
        base_config.update({x_name: x_value, y_name: y_value})
        return {
            "experiment": config["experiment"],
            "parameters": base_config,
        }

    def _run_status(self, point_config: dict[str, Any]) -> str:
        exp_hash = ExperimentHash.compute(point_config)
        exp_dir = self.artifacts_root / exp_hash
        has_cache = all((exp_dir / name).exists() for name in ("metadata.json", "config.json", "results.json"))
        return "cache hit" if has_cache else "running experiment"

    def _artifact_dir(self, config: dict[str, Any]) -> Path:
        phase_hash = ExperimentHash.compute({"phase_diagram": config})
        return self.artifacts_root / phase_hash

    def _extract_metrics(self, result: dict[str, Any]) -> dict[str, Any]:
        return {key: result.get(key) for key in self.METRIC_KEYS}


def generate_stability_heatmap(data: dict[str, Any]) -> list[list[int]]:
    """Classify each cell as stable (+1), marginal (0), or unstable (-1)."""
    heatmap: list[list[int]] = []
    for row in data.get("grid", []):
        heatmap_row: list[int] = []
        for cell in row:
            spectral_radius = cell.get("spectral_radius")
            bp_converged = bool(cell.get("bp_converged"))

            if spectral_radius is None:
                heatmap_row.append(0)
            elif bp_converged and float(spectral_radius) < 1.0:
                heatmap_row.append(1)
            elif (not bp_converged) and float(spectral_radius) >= 1.0:
                heatmap_row.append(-1)
            else:
                heatmap_row.append(0)
        heatmap.append(heatmap_row)
    return heatmap


def render_ascii_heatmap(data: dict[str, Any], heatmap: list[list[int]]) -> str:
    """Render an ASCII heatmap using +, 0, - labels."""
    symbol = {1: "+", 0: "0", -1: "-"}
    y_values = [str(v) for v in data.get("y_values", [])]
    lines = ["Phase Diagram", "", "    " + " ".join(y_values)]

    for idx, x_value in enumerate(data.get("x_values", [])):
        row = heatmap[idx] if idx < len(heatmap) else []
        row_symbols = " ".join(symbol.get(int(v), "?") for v in row)
        lines.append(f"{x_value:<4} {row_symbols}")
    return "\n".join(lines)
