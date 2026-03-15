"""Deterministic BP threshold estimation from phase-diagram artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class BPThresholdEstimator:
    """Estimate decoder threshold from a phase-diagram grid."""

    def __init__(self, smooth: bool = False, smooth_window: int = 3) -> None:
        self.smooth = bool(smooth)
        self.smooth_window = int(smooth_window)

    def load_phase_diagram(self, path: str | Path) -> dict[str, Any]:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("phase_diagram.json must be a JSON object")
        if "grid" not in data:
            raise ValueError("phase_diagram.json missing required field: grid")
        return data

    def compute_success_rates(self, phase_diagram: dict[str, Any]) -> list[tuple[float, float]]:
        grid = phase_diagram.get("grid")
        if not isinstance(grid, list) or not grid:
            return []

        x_values = phase_diagram.get("x_values")
        if not isinstance(x_values, list) or len(x_values) != len(grid):
            x_values = [float(i) for i in range(len(grid))]

        rates: list[tuple[float, float]] = []
        for i, row in enumerate(grid):
            if not isinstance(row, list) or not row:
                success_probability = 0.0
            else:
                successes = sum(1 for value in row if int(value) == 1)
                success_probability = float(successes) / float(len(row))
            rates.append((float(x_values[i]), success_probability))

        rates.sort(key=lambda item: item[0])
        if self.smooth and self.smooth_window > 1:
            return self._moving_average(rates, window=self.smooth_window)
        return rates

    def estimate_threshold(self, success_rates: list[tuple[float, float]]) -> dict[str, Any]:
        if not success_rates:
            return {"threshold": None, "method": "50_percent_crossing"}

        crossing = None
        for i in range(len(success_rates) - 1):
            x0, p0 = success_rates[i]
            x1, p1 = success_rates[i + 1]
            if p0 >= 0.5 and p1 < 0.5:
                if p0 == p1:
                    crossing = x0
                else:
                    ratio = (0.5 - p0) / (p1 - p0)
                    crossing = x0 + ratio * (x1 - x0)
                break

        if crossing is None:
            eligible = [x for x, p in success_rates if p >= 0.5]
            crossing = max(eligible) if eligible else None

        return {"threshold": crossing, "method": "50_percent_crossing"}

    def build_ascii_report(
        self,
        success_rates: list[tuple[float, float]],
        threshold: dict[str, Any],
    ) -> str:
        lines = ["BP Threshold Report", "", "error_rate  success"]
        for error_rate, success in success_rates:
            lines.append(f"{error_rate:.6f}  {success:.6f}")
        lines.append("")
        value = threshold.get("threshold")
        if value is None:
            lines.append("Estimated threshold ≈ unavailable")
        else:
            lines.append(f"Estimated threshold ≈ {float(value):.6f}")
        return "\n".join(lines)

    def write_threshold_artifact(
        self,
        path: str | Path,
        experiment: str,
        success_rates: list[tuple[float, float]],
        threshold: dict[str, Any],
    ) -> dict[str, Any]:
        payload = {
            "data_points": [[float(x), float(y)] for x, y in success_rates],
            "estimated_threshold": threshold.get("threshold"),
            "experiment": experiment,
            "method": threshold.get("method", "50_percent_crossing"),
        }
        serialized = json.dumps(payload, sort_keys=True, indent=2)
        Path(path).write_text(f"{serialized}\n", encoding="utf-8")
        return payload

    @staticmethod
    def _moving_average(
        success_rates: list[tuple[float, float]],
        window: int,
    ) -> list[tuple[float, float]]:
        half_window = window // 2
        smoothed: list[tuple[float, float]] = []
        for i, (x, _) in enumerate(success_rates):
            start = max(0, i - half_window)
            end = min(len(success_rates), i + half_window + 1)
            values = [success_rates[j][1] for j in range(start, end)]
            smoothed.append((x, sum(values) / float(len(values))))
        return smoothed
