from __future__ import annotations

import subprocess
import sys


def test_nb_perturbation_experiment_runs() -> None:
    proc = subprocess.run(
        [sys.executable, "experiments/nb_perturbation_scoring_experiment.py"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "NB Perturbation Scoring" in proc.stdout
    assert "predicted_top3" in proc.stdout
    assert "exact_top3" in proc.stdout
