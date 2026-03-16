from __future__ import annotations

import json

import numpy as np

from src.qec.analysis.spectral_phase_space import project_2d, project_3d
from src.qec.analysis.spectral_trajectory import SpectralTrajectoryRecorder
from src.qec.analysis.trajectory_replay import replay_spectral_trajectory
from src.qec.discovery.discovery_engine import run_structure_discovery
from src.qec.experiments.experiment_hash import ExperimentRunner


def _default_spec() -> dict[str, int]:
    return {
        "num_variables": 6,
        "num_checks": 3,
        "variable_degree": 2,
        "check_degree": 4,
    }


def test_trajectory_recording() -> None:
    recorder = SpectralTrajectoryRecorder()

    recorder.record([1.0, 0.2, 0.1])
    recorder.record([1.1, 0.25, 0.15])

    assert recorder.length() == 2
    arr = recorder.as_array()
    assert arr.shape == (2, 3)
    assert arr.dtype == np.float64


def test_trajectory_json_export(tmp_path) -> None:
    recorder = SpectralTrajectoryRecorder()
    recorder.record([1.0, 0.2])
    recorder.record([1.1, 0.25])

    payload = recorder.to_json()
    assert "spectral_trajectory" in payload
    assert len(payload["spectral_trajectory"]) == 2

    path = tmp_path / "trajectory.json"
    recorder.save(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data == payload


def test_replay_spectral_trajectory() -> None:
    trajectory = [[1.0, 0.2, 0.1], [1.1, 0.25, 0.15]]
    replayed = list(replay_spectral_trajectory(trajectory))

    assert len(replayed) == 2
    assert replayed[0].dtype == np.float64
    np.testing.assert_allclose(replayed[1], np.asarray(trajectory[1], dtype=np.float64))


def test_projection_utilities() -> None:
    trajectory = np.asarray(
        [
            [1.0, 0.2, 0.1, 0.5],
            [1.1, 0.25, 0.15, 0.45],
            [1.2, 0.3, 0.2, 0.4],
        ],
        dtype=np.float64,
    )

    p2 = project_2d(trajectory, dims=(0, 2))
    p3 = project_3d(trajectory, dims=(0, 1, 3))

    assert p2.shape == (3, 2)
    assert p3.shape == (3, 3)
    np.testing.assert_allclose(p2[:, 0], trajectory[:, 0])


def test_discovery_trajectory_determinism_and_length() -> None:
    spec = _default_spec()
    r1 = run_structure_discovery(
        spec,
        num_generations=2,
        population_size=4,
        base_seed=42,
        enable_spectral_trajectory=True,
    )
    r2 = run_structure_discovery(
        spec,
        num_generations=2,
        population_size=4,
        base_seed=42,
        enable_spectral_trajectory=True,
    )

    assert "spectral_trajectory" in r1
    assert len(r1["spectral_trajectory"]) > 0
    assert r1["spectral_trajectory"] == r2["spectral_trajectory"]


def test_harness_wraps_trajectory_result(tmp_path) -> None:
    runner = ExperimentRunner(artifacts_root=tmp_path)

    def execute(_: dict[str, object]) -> dict[str, object]:
        return {"best": 1, "spectral_trajectory": [[1.0, 0.2, 0.1]]}

    wrapped = runner.run({"seed": 0, "experiment_name": "traj"}, execute)

    assert "result" in wrapped
    assert "metadata" in wrapped
    assert "experiment_hash" in wrapped
    assert wrapped["spectral_trajectory"] == [[1.0, 0.2, 0.1]]
