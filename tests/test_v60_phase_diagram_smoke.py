"""
Smoke test for v6.0 phase diagram with spectral stability diagnostics.

Validates:
  - Demo runs cleanly and produces valid output with spectral fields
  - ASCII heatmap is printed
  - JSON artifact is produced
  - Deterministic output
  - Spectral diagnostic fields are present in phase diagram cells
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any

import pytest

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from scripts.run_v59_phase_diagram_demo import run_phase_diagram_demo
from qec.diagnostics.phase_heatmap import print_phase_heatmap


class TestV60PhaseDiagramSmoke:
    def test_demo_runs_cleanly(self):
        """Demo produces valid result dict with spectral fields."""
        result = run_phase_diagram_demo()
        assert "phase_diagram" in result
        assert "boundary_analysis" in result

        pd = result["phase_diagram"]
        assert "grid_axes" in pd
        assert "cells" in pd
        assert len(pd["cells"]) > 0

    def test_spectral_fields_present(self):
        """Phase diagram cells contain v6.0 spectral diagnostic fields."""
        result = run_phase_diagram_demo()
        pd = result["phase_diagram"]

        for cell in pd["cells"]:
            assert "mean_spectral_radius" in cell
            assert "mean_bethe_min_eigenvalue" in cell
            assert "mean_bp_stability_score" in cell
            assert "mean_jacobian_spectral_radius_est" in cell

    def test_spectral_fields_populated(self):
        """Spectral fields have non-None values for all cells."""
        result = run_phase_diagram_demo()
        pd = result["phase_diagram"]

        for cell in pd["cells"]:
            assert cell["mean_spectral_radius"] is not None
            assert cell["mean_bethe_min_eigenvalue"] is not None
            assert cell["mean_bp_stability_score"] is not None
            assert cell["mean_jacobian_spectral_radius_est"] is not None

    def test_localization_fields_present(self):
        """Phase diagram cells contain v6.1 localization diagnostic fields."""
        result = run_phase_diagram_demo()
        pd = result["phase_diagram"]

        for cell in pd["cells"]:
            assert "mean_nb_max_ipr" in cell
            assert "mean_nb_num_localized_modes" in cell
            assert "mean_nb_top_localization_score" in cell

    def test_localization_fields_populated(self):
        """Localization fields have non-None values for all cells."""
        result = run_phase_diagram_demo()
        pd = result["phase_diagram"]

        for cell in pd["cells"]:
            assert cell["mean_nb_max_ipr"] is not None
            assert cell["mean_nb_num_localized_modes"] is not None
            assert cell["mean_nb_top_localization_score"] is not None

    def test_ascii_heatmap_output(self, capsys):
        """ASCII heatmap prints valid output."""
        result = run_phase_diagram_demo()
        pd = result["phase_diagram"]

        output = print_phase_heatmap(pd)

        assert "Phase diagram summary" in output
        assert "Legend:" in output
        assert "success basin" in output

    def test_json_artifact_produced(self):
        """Result is fully JSON-serializable."""
        result = run_phase_diagram_demo()
        serialized = json.dumps(result, indent=2)
        assert len(serialized) > 0
        deserialized = json.loads(serialized)
        assert "phase_diagram" in deserialized

    def test_determinism(self):
        """Repeated runs produce identical output."""
        r1 = run_phase_diagram_demo()
        r2 = run_phase_diagram_demo()
        j1 = json.dumps(r1, sort_keys=True)
        j2 = json.dumps(r2, sort_keys=True)
        assert j1 == j2

    def test_backward_compatible_fields(self):
        """v5.9 fields remain present and valid."""
        result = run_phase_diagram_demo()
        pd = result["phase_diagram"]

        for cell in pd["cells"]:
            assert "dominant_phase" in cell
            assert cell["dominant_phase"] in {-1, 0, 1}
            assert "success_fraction" in cell
            assert "boundary_fraction" in cell
            assert "failure_fraction" in cell
            assert "phase_entropy" in cell
            total = (cell["success_fraction"] +
                     cell["boundary_fraction"] +
                     cell["failure_fraction"])
            assert abs(total - 1.0) < 1e-10

    def test_trapping_candidate_fields_present(self):
        """Phase diagram cells contain v6.2 trapping-set candidate fields."""
        result = run_phase_diagram_demo()
        pd = result["phase_diagram"]

        for cell in pd["cells"]:
            assert "mean_nb_candidate_nodes" in cell
            assert "mean_nb_max_node_participation" in cell
            assert "mean_nb_candidate_clusters" in cell

    def test_trapping_candidate_fields_populated(self):
        """Trapping-set candidate fields have non-None values."""
        result = run_phase_diagram_demo()
        pd = result["phase_diagram"]

        for cell in pd["cells"]:
            assert cell["mean_nb_candidate_nodes"] is not None
            assert cell["mean_nb_max_node_participation"] is not None
            assert cell["mean_nb_candidate_clusters"] is not None

    def test_json_roundtrip(self):
        """Output survives JSON serialization roundtrip."""
        result = run_phase_diagram_demo()
        serialized = json.dumps(result)
        deserialized = json.loads(serialized)
        assert json.dumps(deserialized, sort_keys=True) == json.dumps(result, sort_keys=True)
