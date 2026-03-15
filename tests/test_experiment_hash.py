from __future__ import annotations

import json
from pathlib import Path

from src.qec.experiments.experiment_hash import (
    ExperimentEnvironmentMetadata,
    ExperimentHash,
    ExperimentRunner,
)


def test_deterministic_hash_identical_configs() -> None:
    metadata = ExperimentEnvironmentMetadata(
        git_commit="abc",
        repo_version="1.0.0",
        python_version="3.12.0",
        numpy_version="2.0.0",
        scipy_version="1.13.0",
    )
    config = {"alpha": 1, "beta": {"x": 2, "y": [1, 2, 3]}}

    h1 = ExperimentHash.compute(config, metadata=metadata)
    h2 = ExperimentHash.compute(config, metadata=metadata)

    assert h1 == h2
    assert len(h1) == 16


def test_hash_sensitivity_config_change() -> None:
    metadata = ExperimentEnvironmentMetadata(
        git_commit="abc",
        repo_version="1.0.0",
        python_version="3.12.0",
        numpy_version="2.0.0",
        scipy_version="1.13.0",
    )

    h1 = ExperimentHash.compute({"alpha": 1}, metadata=metadata)
    h2 = ExperimentHash.compute({"alpha": 2}, metadata=metadata)

    assert h1 != h2


def test_cache_hit_skips_execution(tmp_path: Path) -> None:
    runner = ExperimentRunner(artifacts_root=tmp_path)
    config = {"seed": 7, "p": 0.05}

    calls = {"count": 0}

    def execute(spec: dict[str, object]) -> dict[str, object]:
        calls["count"] += 1
        return {"status": "ran", "seed": spec["seed"]}

    first = runner.run(config, execute)
    second = runner.run(config, execute)

    assert first == second
    assert calls["count"] == 1


def test_cache_miss_runs_execution(tmp_path: Path) -> None:
    runner = ExperimentRunner(artifacts_root=tmp_path)

    calls = {"count": 0}

    def execute(spec: dict[str, object]) -> dict[str, object]:
        calls["count"] += 1
        return {"status": "ran", "seed": spec["seed"]}

    result_a = runner.run({"seed": 1}, execute)
    result_b = runner.run({"seed": 2}, execute)

    assert result_a != result_b
    assert calls["count"] == 2


def test_artifact_integrity_required_for_cache(tmp_path: Path) -> None:
    runner = ExperimentRunner(artifacts_root=tmp_path)
    config = {"seed": 11}

    call_counter = {"count": 0}

    def execute(spec: dict[str, object]) -> dict[str, object]:
        call_counter["count"] += 1
        return {"ok": True, "seed": spec["seed"]}

    first = runner.run(config, execute)
    assert first["ok"] is True
    assert call_counter["count"] == 1

    exp_hash = ExperimentHash.compute(config)
    exp_dir = tmp_path / exp_hash
    (exp_dir / "results.json").unlink()

    second = runner.run(config, execute)

    assert second["ok"] is True
    assert call_counter["count"] == 2
    assert (exp_dir / "metadata.json").exists()
    assert (exp_dir / "config.json").exists()
    assert (exp_dir / "results.json").exists()

    with (exp_dir / "metadata.json").open("r", encoding="utf-8") as f:
        metadata = json.load(f)
    assert metadata["experiment_hash"] == exp_hash
