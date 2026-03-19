from __future__ import annotations

import numpy as np

from qec.discovery.mutation_interface import NBEigenvectorFlowMutation
from qec.discovery.nb_eigenvector_flow_mutation import NBEigenvectorFlowMutator


def test_single_mode_path_remains_deterministic() -> None:
    operator = NBEigenvectorFlowMutation(NBEigenvectorFlowMutator())
    H = np.asarray(
        [
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    leading = np.asarray([0.9, 0.1, 0.0, 0.0], dtype=np.float64)

    out1 = operator.mutate(
        H,
        {"leading_vector": leading},
        {"enable_multi_mode_nb_mutation": False},
    )
    out2 = operator.mutate(
        H,
        {"leading_vector": leading},
        {"enable_multi_mode_nb_mutation": False},
    )

    assert out1[2] == "nb_flow"
    assert np.array_equal(out1[0], out2[0])
    assert out1[3] == out2[3]
import json

import numpy as np

from qec.discovery.nb_eigenvector_flow_mutation import (
    nb_flow_mutation,
    spectral_annealing_strength,
)
from qec.discovery.threshold_search import (
    PhaseDiagramOrchestrator,
    SpectralSearchConfig,
    run_spectral_threshold_search,
)
from qec.generation.deterministic_construction import construct_deterministic_tanner_graph
from qec.spectral.nb_spectrum import compute_nb_spectral_gap


def _graph() -> np.ndarray:
    return construct_deterministic_tanner_graph(
        {
            "num_variables": 8,
            "num_checks": 4,
            "variable_degree": 2,
            "check_degree": 4,
        }
    )


def test_compute_nb_spectral_gap_orders_by_magnitude():
    eigenvalues = np.array([1.0 + 0.0j, -4.0 + 0.0j, 3.0 + 0.0j], dtype=np.complex128)
    assert compute_nb_spectral_gap(eigenvalues) == 1.0


def test_spectral_annealing_strength_monotonic():
    small_gap = spectral_annealing_strength(0.1)
    large_gap = spectral_annealing_strength(1.0)
    assert small_gap > large_gap


def test_mutation_size_deterministic_for_same_input():
    H = _graph()
    eigenvector = np.linspace(1.0, 2.0, num=16, dtype=np.float64)
    eigenvalues = np.array([3.0 + 0.0j, 2.8 + 0.0j, 0.2 + 0.0j], dtype=np.complex128)

    _, meta_a = nb_flow_mutation(
        H,
        eigenvector,
        eigenvalues,
        annealing=True,
        base_mutation_size=4,
    )
    _, meta_b = nb_flow_mutation(
        H,
        eigenvector,
        eigenvalues,
        annealing=True,
        base_mutation_size=4,
    )

    assert meta_a["mutation_size"] == meta_b["mutation_size"]
    assert meta_a["mutation_size"] == 3


def test_threshold_search_records_annealing_metadata(tmp_path, monkeypatch):
    H0 = _graph()

    monkeypatch.setattr(
        PhaseDiagramOrchestrator,
        "evaluate",
        lambda self, H, *, max_phase_diagram_size, seed: {
            "measured_boundary": {"mean_boundary_spectral_radius": 0.05}
        },
    )

    cfg = SpectralSearchConfig(
        iterations=1,
        max_phase_diagram_size=1,
        output_dir=str(tmp_path),
        enable_nb_flow_mutation=True,
        enable_nb_spectral_annealing=True,
        annealing_base_mutation_size=4,
    )
    run_spectral_threshold_search(H0, config=cfg)

    payload = json.loads((tmp_path / "candidate_metrics.json").read_text(encoding="utf-8"))
    flow_metrics = next(m for m in payload["candidates"] if m["source"] == "nb_flow")

    assert "nb_spectral_gap" in flow_metrics
    assert "annealing_strength" in flow_metrics
    assert "mutation_size" in flow_metrics
    if flow_metrics["nb_spectral_gap"] is not None:
        assert round(float(flow_metrics["nb_spectral_gap"]), 12) == float(flow_metrics["nb_spectral_gap"])
    if flow_metrics["annealing_strength"] is not None:
        assert round(float(flow_metrics["annealing_strength"]), 12) == float(flow_metrics["annealing_strength"])
    if flow_metrics["mutation_size"] is not None:
        assert isinstance(flow_metrics["mutation_size"], int)
