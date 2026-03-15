from __future__ import annotations

from pathlib import Path

from src.qec.experiments.experiment_hash import ExperimentHash, ExperimentRunner
from src.qec.experiments.parameter_sweep import generate_parameter_grid


def test_generate_parameter_grid_is_deterministic() -> None:
    sweep = {
        "iterations": [10, 20],
        "seed": [1, 2],
        "code_length": [512, 1024],
    }

    expected = [
        {"code_length": 512, "iterations": 10, "seed": 1},
        {"code_length": 512, "iterations": 10, "seed": 2},
        {"code_length": 512, "iterations": 20, "seed": 1},
        {"code_length": 512, "iterations": 20, "seed": 2},
        {"code_length": 1024, "iterations": 10, "seed": 1},
        {"code_length": 1024, "iterations": 10, "seed": 2},
        {"code_length": 1024, "iterations": 20, "seed": 1},
        {"code_length": 1024, "iterations": 20, "seed": 2},
    ]

    first = generate_parameter_grid(sweep)
    second = generate_parameter_grid(sweep)

    assert first == expected
    assert second == expected


def test_cache_reuse_for_identical_configs(tmp_path: Path) -> None:
    runner = ExperimentRunner(artifacts_root=tmp_path)
    config = {"seed": 5, "iterations": 10}
    calls = {"count": 0}

    def execute(spec: dict[str, object]) -> dict[str, object]:
        calls["count"] += 1
        return {"seed": spec["seed"], "iterations": spec["iterations"]}

    first = runner.run(config, execute)
    second = runner.run(config, execute)

    assert first == second
    assert calls["count"] == 1


def test_hash_changes_for_callable_and_dirty_repo() -> None:
    config = {"seed": 1}
    metadata_clean = ExperimentHash.collect_environment_metadata()
    metadata_dirty = type(metadata_clean)(
        git_commit=metadata_clean.git_commit,
        repo_version=metadata_clean.repo_version,
        python_version=metadata_clean.python_version,
        numpy_version=metadata_clean.numpy_version,
        scipy_version=metadata_clean.scipy_version,
        dirty_repo=(not metadata_clean.dirty_repo) if isinstance(metadata_clean.dirty_repo, bool) else True,
    )

    clean_hash = ExperimentHash.compute(config, experiment_callable="module.one:run", metadata=metadata_clean)
    other_callable_hash = ExperimentHash.compute(
        config,
        experiment_callable="module.two:run",
        metadata=metadata_clean,
    )
    dirty_hash = ExperimentHash.compute(config, experiment_callable="module.one:run", metadata=metadata_dirty)

    assert clean_hash != other_callable_hash
    assert clean_hash != dirty_hash
