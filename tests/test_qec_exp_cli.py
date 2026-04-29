from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from qec.experiments.experiment_hash import ExperimentHash

_REPO_ROOT = Path(__file__).resolve().parents[1]


def test_estimate_threshold_cli_runs(tmp_path: Path) -> None:
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.qec.experiments.qec_exp_cli",
            "--artifacts-root",
            str(tmp_path),
            "estimate-threshold",
            "bp-threshold",
        ],
        cwd=str(_REPO_ROOT),
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0, result.stderr
    assert "Estimated threshold:" in result.stdout

    exp_hash = ExperimentHash.compute({"experiment": "bp-threshold"})
    artifact_path = tmp_path / exp_hash / "threshold_estimate.json"
    assert artifact_path.exists()

    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert payload["experiment"] == "bp-threshold"
    assert payload["method"] == "50_percent_crossing"


def test_pyproject_console_script_targets_packaged_module() -> None:
    import tomllib

    pyproject = tomllib.loads((Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(encoding="utf-8"))
    assert pyproject["project"]["scripts"]["qec-exp"] == "qec.experiments.qec_exp_cli:main"
