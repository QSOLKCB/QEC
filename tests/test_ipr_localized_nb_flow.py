from __future__ import annotations

import json

import numpy as np

from qec.discovery.nb_eigenvector_flow_mutation import (
    NBEigenvectorFlowMutator,
    compute_ipr_localization,
    select_localized_edges,
)
from qec.discovery.threshold_search import (
    PhaseDiagramOrchestrator,
    SpectralSearchConfig,
    run_spectral_threshold_search,
)
from qec.generation.deterministic_construction import construct_deterministic_tanner_graph


ROUND = 12


def _graph() -> np.ndarray:
    spec = {
        "num_variables": 8,
        "num_checks": 4,
        "variable_degree": 2,
        "check_degree": 4,
    }
    return construct_deterministic_tanner_graph(spec)


def test_ipr_correctness_known_vectors() -> None:
    v_uniform = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float64)
    assert compute_ipr_localization(v_uniform) == 0.25

    v_localized = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    assert compute_ipr_localization(v_localized) == 1.0

    v_zero = np.zeros(4, dtype=np.float64)
    assert compute_ipr_localization(v_zero) == 0.0


def test_deterministic_localization_stable_ranking() -> None:
    flow = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float64)
    vec = np.array([0.8, -0.8, 0.2, -0.8, 0.1], dtype=np.float64)

    selected = select_localized_edges(flow, vec, top_fraction=0.6)
    # equal |v| values at indices 0,1,3 must be index-stable
    assert np.array_equal(selected, np.array([0, 1, 3], dtype=np.int64))


def test_ipr_localized_mutation_reproducibility() -> None:
    mut = NBEigenvectorFlowMutator()
    H = _graph()
    vec = np.array([0.1, -0.3, 0.8, 0.2, 0.05, 0.01], dtype=np.float64)

    first, meta_first = mut.mutate(
        H,
        vec,
        use_ipr_localization=True,
    )
    second, meta_second = mut.mutate(
        H,
        vec,
        use_ipr_localization=True,
    )

    assert np.array_equal(first, second)
    assert meta_first == meta_second
    assert meta_first["localization_edge_count"] >= 1
    assert isinstance(meta_first["ipr_localization_score"], float)


def test_artifact_metadata_contains_ipr_fields(tmp_path, monkeypatch) -> None:
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
        enable_ipr_localized_nb_flow=True,
        ipr_localization_fraction=0.5,
    )
    run_spectral_threshold_search(H0, config=cfg)

    payload = json.loads((tmp_path / "candidate_metrics.json").read_text(encoding="utf-8"))
    flow_entries = [m for m in payload["candidates"] if m.get("source") == "nb_flow"]
    assert len(flow_entries) == 1

    entry = flow_entries[0]
    assert "ipr_localization_score" in entry
    assert "localization_edge_count" in entry
    assert entry["localization_edge_count"] >= 1
    assert entry["ipr_localization_score"] == round(float(entry["ipr_localization_score"]), ROUND)
