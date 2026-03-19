from __future__ import annotations

import json

import numpy as np

from qec.analysis.bp_diagnostics import BPDiagnostics, collect_bp_diagnostics
from qec.discovery.threshold_search import SpectralSearchConfig, run_spectral_threshold_search
from qec.generation.deterministic_construction import construct_deterministic_tanner_graph


def _small_graph() -> np.ndarray:
    spec = {
        "num_variables": 8,
        "num_checks": 4,
        "variable_degree": 2,
        "check_degree": 4,
    }
    return construct_deterministic_tanner_graph(spec)


def test_collect_bp_diagnostics_extracts_expected_fields() -> None:
    diag = collect_bp_diagnostics(
        {
            "residuals": [0.2, 1e-9],
            "syndrome_weights": [4, 0],
        },
    )

    assert isinstance(diag, BPDiagnostics)
    assert diag.iterations_to_converge == 2
    assert diag.converged is True
    assert diag.final_residual == 1e-09
    assert diag.residual_history == [0.2, 1e-09]
    assert diag.syndrome_weight_history == [4, 0]


def test_bp_diagnostics_artifacts_are_deterministic(tmp_path, monkeypatch) -> None:
    H0 = _small_graph()

    from qec.discovery import threshold_search as mod

    monkeypatch.setattr(
        mod.PhaseDiagramOrchestrator,
        "evaluate",
        lambda self, H, *, max_phase_diagram_size, seed: {
            "measured_boundary": {"mean_boundary_spectral_radius": 0.045},
            "grid_results": [
                {"mean_residual_norm": 1e-9},
                {"mean_residual_norm": 1e-10},
            ],
        },
    )

    cfg = SpectralSearchConfig(
        iterations=1,
        max_phase_diagram_size=1,
        output_dir=str(tmp_path),
        enable_bp_diagnostics=True,
    )
    run_spectral_threshold_search(H0, config=cfg)

    candidate_metrics_text = (tmp_path / "candidate_metrics.json").read_text(encoding="utf-8")
    summary_text = (tmp_path / "convergence_summary.json").read_text(encoding="utf-8")

    assert '"bp_iterations"' in candidate_metrics_text
    assert '"bp_final_residual"' in candidate_metrics_text
    assert '"bp_converged"' in candidate_metrics_text

    summary = json.loads(summary_text)
    assert summary == {
        "convergence_rate": 1.0,
        "max_bp_iterations": 2,
        "mean_bp_iterations": 2.0,
        "mean_final_residual": 1e-10,
    }

    run_spectral_threshold_search(H0, config=cfg)
    summary_text_second = (tmp_path / "convergence_summary.json").read_text(encoding="utf-8")
    assert summary_text == summary_text_second
