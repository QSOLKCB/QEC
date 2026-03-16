from __future__ import annotations

import json

import numpy as np

from src.qec.analysis.spectral_defect_atlas import SpectralDefectAtlas
from src.qec.discovery.nb_eigenvector_flow_mutation import NBEigenvectorFlowMutator
from src.qec.discovery.threshold_search import PhaseDiagramOrchestrator, SpectralSearchConfig, run_spectral_threshold_search
from src.qec.generation.deterministic_construction import construct_deterministic_tanner_graph


def _graph() -> np.ndarray:
    spec = {
        "num_variables": 8,
        "num_checks": 4,
        "variable_degree": 2,
        "check_degree": 4,
    }
    return construct_deterministic_tanner_graph(spec)


def test_flow_normalization() -> None:
    mut = NBEigenvectorFlowMutator()
    flow = mut.compute_flow(np.array([1.0, -2.0, 3.0], dtype=np.float64))
    assert flow.dtype == np.float64
    assert np.isclose(np.sum(flow), 1.0)
    assert np.all(flow >= 0.0)


def test_deterministic_edge_selection() -> None:
    mut = NBEigenvectorFlowMutator()
    flow = np.array([0.1, 0.6, 0.6, 0.2], dtype=np.float64)
    assert mut.select_edge(flow) == 1


def test_mutation_reproducibility() -> None:
    mut = NBEigenvectorFlowMutator()
    H = _graph()
    vec = np.array([0.1, -0.3, 0.8, 0.2, 0.05, 0.01], dtype=np.float64)

    first, meta_first = mut.mutate(H, vec)
    second, meta_second = mut.mutate(H, vec)

    assert np.array_equal(first, second)
    assert meta_first == meta_second


def test_flow_metrics_artifact_logging(tmp_path, monkeypatch) -> None:
    H0 = _graph()

    monkeypatch.setattr(
        PhaseDiagramOrchestrator,
        "evaluate",
        lambda self, H, *, max_phase_diagram_size, seed: {"measured_boundary": {"mean_boundary_spectral_radius": 0.1}},
    )

    cfg = SpectralSearchConfig(
        iterations=1,
        max_phase_diagram_size=1,
        output_dir=str(tmp_path),
        enable_nb_flow_mutation=True,
    )
    run_spectral_threshold_search(H0, config=cfg)

    payload = json.loads((tmp_path / "candidate_metrics.json").read_text(encoding="utf-8"))
    flow_entries = [m for m in payload["candidates"] if m.get("source") == "nb_flow"]
    assert len(flow_entries) == 1
    assert isinstance(flow_entries[0].get("flow_edge_index"), int)
    assert isinstance(flow_entries[0].get("flow_strength"), float)


def test_mutation_context_metadata_with_atlas() -> None:
    H = _graph()
    vec = np.array([0.4, -0.8, 0.2, 0.1, 0.05, 0.01], dtype=np.float64)
    atlas = SpectralDefectAtlas(max_patterns=5)
    sig = atlas.signature(vec)
    atlas.record(sig, "flow_edge_2", 0.01)

    mut = NBEigenvectorFlowMutator()
    _, meta = mut.mutate(
        H,
        vec,
        context={
            "spectral_defect_atlas": atlas,
            "enable_spectral_defect_atlas": True,
            "nb_spectral_radius": 1.23,
        },
    )

    assert meta["atlas_hit"] is True
    assert isinstance(meta["defect_signature"], str)
    assert isinstance(meta["atlas_pattern_index"], int)
    assert meta["repair_action"] == "flow_edge_2"
