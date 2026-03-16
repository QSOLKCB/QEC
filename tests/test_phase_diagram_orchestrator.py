from __future__ import annotations

import json
from pathlib import Path

from src.qec.experiments.experiment_hash import ExperimentHash
from src.qec.experiments.phase_diagram_orchestrator import (
    PhaseDiagramOrchestrator,
    generate_stability_heatmap,
)


def _config() -> dict[str, object]:
    return {
        "experiment": "bp-threshold",
        "x_axis": {"name": "physical_error_rate", "values": [0.01, 0.02]},
        "y_axis": {"name": "decoder_iterations", "values": [10, 20]},
    }


def test_grid_generation_is_deterministic() -> None:
    orchestrator = PhaseDiagramOrchestrator(experiment_registry={"bp-threshold": lambda _: {}})
    points = orchestrator.generate_grid(_config())
    assert points == [(0.01, 10), (0.01, 20), (0.02, 10), (0.02, 20)]


def test_cache_reuse_for_phase_diagram_runs(tmp_path: Path) -> None:
    calls = {"count": 0}

    def execute(config: dict[str, object]) -> dict[str, object]:
        calls["count"] += 1
        params = config["parameters"]
        p = float(params["physical_error_rate"])
        i = float(params["decoder_iterations"])
        return {
            "bp_converged": p < 0.03,
            "spectral_radius": round(0.8 + p + i / 200.0, 12),
            "mean_llr": round(1.0 - p, 12),
            "decoder_success_rate": round(1.0 - p / 2.0, 12),
        }

    orchestrator = PhaseDiagramOrchestrator(
        artifacts_root=tmp_path,
        experiment_registry={"bp-threshold": execute},
    )

    run1 = orchestrator.run(_config())
    run2 = orchestrator.run(_config())

    assert run1["phase_diagram"] == run2["phase_diagram"]
    assert calls["count"] >= 4
    assert run1["statuses"].count("running experiment") >= 1
    assert len(run1["statuses"]) == len(run2["statuses"]) == 4


def test_phase_diagram_json_is_deterministic(tmp_path: Path) -> None:
    def execute(_: dict[str, object]) -> dict[str, object]:
        return {
            "bp_converged": True,
            "spectral_radius": 0.9,
            "mean_llr": 1.2,
            "decoder_success_rate": 0.95,
        }

    orchestrator = PhaseDiagramOrchestrator(
        artifacts_root=tmp_path,
        experiment_registry={"bp-threshold": execute},
    )

    result1 = orchestrator.run(_config())
    first = Path(result1["artifact"]).read_text(encoding="utf-8")

    result2 = orchestrator.run(_config())
    second = Path(result2["artifact"]).read_text(encoding="utf-8")

    assert first == second

    parsed = json.loads(first)
    artifact_dir = tmp_path / ExperimentHash.compute({"phase_diagram": _config()})
    assert artifact_dir / "phase_diagram.json" == Path(result1["artifact"])
    assert parsed["experiment"] == "bp-threshold"


def test_generate_stability_heatmap_classification() -> None:
    data = {
        "grid": [
            [
                {"bp_converged": True, "spectral_radius": 0.8},
                {"bp_converged": True, "spectral_radius": 1.1},
            ],
            [
                {"bp_converged": False, "spectral_radius": 1.2},
                {"bp_converged": False, "spectral_radius": 0.9},
            ],
        ]
    }
    assert generate_stability_heatmap(data) == [[1, 0], [-1, 0]]
