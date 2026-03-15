from __future__ import annotations

import json
from pathlib import Path

from src.qec.experiments.experiment_metadata import ExperimentMetadata, git_commit
from src.qec.experiments.experiment_runner import ExperimentRunner


_REQUIRED_FIELDS = {
    "repo_version",
    "git_commit",
    "timestamp_utc",
    "python",
    "numpy",
    "scipy",
    "experiment_hash",
    "experiment_callable",
    "dirty_repo",
    "seed",
    "hostname",
    "platform",
}


def _dummy_experiment(**_: object) -> dict[str, object]:
    return {"zeta": 1, "alpha": [3, 2, 1]}


def test_metadata_collection_has_required_fields() -> None:
    metadata = ExperimentMetadata(seed=42, timestamp="2026-03-14T10:22:33Z").collect()
    assert _REQUIRED_FIELDS.issubset(metadata.keys())
    assert metadata["seed"] == 42


def test_git_detection_falls_back_when_git_missing(monkeypatch) -> None:
    def _raise(*args, **kwargs):
        raise OSError("git missing")

    monkeypatch.setattr("src.qec.experiments.experiment_metadata.subprocess.run", _raise)
    assert git_commit() == "unknown"


def test_deterministic_json_key_ordering(tmp_path: Path) -> None:
    config = {
        "run_name": "ordering",
        "seed": 7,
        "output_root": str(tmp_path),
        "function_kwargs": {},
    }
    runner = ExperimentRunner(config)
    first = runner.run(_dummy_experiment)
    second = runner.run(_dummy_experiment)

    first_results = Path(first["artifact_dir"]) / "results.json"
    second_results = Path(second["artifact_dir"]) / "results.json"
    first_text = first_results.read_text(encoding="utf-8")
    second_text = second_results.read_text(encoding="utf-8")

    assert first_text == second_text
    assert first_text.index('"alpha"') < first_text.index('"zeta"')


def test_artifact_creation_writes_expected_files(tmp_path: Path) -> None:
    config = {
        "run_name": "smoke",
        "seed": 42,
        "output_root": str(tmp_path),
        "function_kwargs": {"x": 1},
    }
    runner = ExperimentRunner(config)
    payload = runner.run(_dummy_experiment)
    artifact_dir = Path(payload["artifact_dir"])

    assert artifact_dir.exists()
    assert (artifact_dir / "metadata.json").exists()
    assert (artifact_dir / "config.json").exists()
    assert (artifact_dir / "results.json").exists()

    metadata = json.loads((artifact_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["seed"] == 42
    assert metadata["runtime_parameters"]["run_name"] == "smoke"
