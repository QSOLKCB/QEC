from __future__ import annotations

import numpy as np

from src.qec.analysis.spectral_trapping_sets import (
    detect_localization_cluster,
    extract_trapping_subgraph,
    repair_trapping_set,
)
from src.qec.discovery.nb_eigenvector_flow_mutation import NBEigenvectorFlowMutator
from src.qec.discovery.threshold_search import SpectralSearchConfig, run_spectral_threshold_search


def _matrix() -> np.ndarray:
    return np.array(
        [
            [1.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 1.0],
        ],
        dtype=np.float64,
    )


def test_detect_localization_cluster_selects_expected_nodes() -> None:
    v = np.array([0.01, -0.2, 0.9, -0.8, 0.15], dtype=np.float64)
    nodes = detect_localization_cluster(v, threshold_fraction=0.5)
    assert np.array_equal(nodes, np.array([2, 3], dtype=np.int64))


def test_extract_trapping_subgraph_has_deterministic_content() -> None:
    H = _matrix()
    sub = extract_trapping_subgraph(H, np.array([1], dtype=np.int64))
    assert np.array_equal(sub["check_nodes"], np.array([0, 1], dtype=np.int64))
    assert np.array_equal(sub["variable_nodes"], np.array([0, 1, 2], dtype=np.int64))
    assert np.array_equal(
        sub["edges"],
        np.array([[0, 0], [0, 1], [1, 1], [1, 2]], dtype=np.int64),
    )


def test_repair_trapping_set_is_deterministic_and_degree_preserving() -> None:
    H = _matrix()
    cluster = np.array([1, 2], dtype=np.int64)
    repaired1 = repair_trapping_set(H, cluster)
    repaired2 = repair_trapping_set(H, cluster)

    assert np.array_equal(repaired1, repaired2)
    assert np.array_equal(np.sum(H, axis=0), np.sum(repaired1, axis=0))
    assert np.array_equal(np.sum(H, axis=1), np.sum(repaired1, axis=1))


def test_threshold_search_spectral_trapping_repair_integration(tmp_path, monkeypatch) -> None:
    from src.qec.discovery import threshold_search as mod

    H0 = _matrix()

    monkeypatch.setattr(
        mod.PhaseDiagramOrchestrator,
        "evaluate",
        lambda self, H, *, max_phase_diagram_size, seed: {
            "measured_boundary": {"mean_boundary_spectral_radius": 0.25}
        },
    )
    monkeypatch.setattr(
        mod,
        "compute_nb_spectrum",
        lambda H: {
            "spectral_radius": 0.5,
            "eigenvector": np.array([0.0, 0.9, 0.1, 0.7], dtype=np.float64),
            "ipr": 0.0,
            "edge_energy": np.array([], dtype=np.float64),
            "eeec": 0.0,
            "sis": 0.0,
        },
    )

    cfg = SpectralSearchConfig(
        iterations=1,
        output_dir=str(tmp_path),
        enable_nb_flow_mutation=True,
        enable_spectral_trapping_repair=True,
        trapping_localization_fraction=0.5,
    )
    result = run_spectral_threshold_search(H0, config=cfg)

    metrics = result["history"][0]["spectral_metrics"]
    assert "spectral_cluster_size" in metrics
    assert "trapping_repair_applied" in metrics
    assert isinstance(metrics["spectral_cluster_size"], int)
    assert isinstance(metrics["trapping_repair_applied"], bool)


def test_repair_stage_runs_inside_flow_mutator(monkeypatch) -> None:
    mut = NBEigenvectorFlowMutator()
    H = _matrix()
    vec = np.array([0.2, 0.8, 0.1, 0.7], dtype=np.float64)

    calls = {"count": 0}
    from src.qec.discovery import nb_eigenvector_flow_mutation as mod

    def _spy_repair(graph, cluster_nodes):
        calls["count"] += 1
        return np.asarray(graph, dtype=np.float64).copy()

    monkeypatch.setattr(mod, "repair_trapping_set", _spy_repair)

    _, meta = mut.mutate(
        H,
        vec,
        enable_spectral_trapping_repair=True,
        trapping_localization_fraction=0.5,
    )

    assert calls["count"] == 1
    assert isinstance(meta["spectral_cluster_size"], int)
    assert isinstance(meta["trapping_repair_applied"], bool)
