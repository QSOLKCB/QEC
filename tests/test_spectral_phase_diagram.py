from __future__ import annotations

import json

import numpy as np

from qec.analysis.spectral_landscape_memory import SpectralLandscapeMemory
from qec.analysis.spectral_phase_boundaries import detect_phase_boundaries
from qec.analysis.spectral_phase_diagram import generate_spectral_phase_diagram


def _memory() -> SpectralLandscapeMemory:
    memory = SpectralLandscapeMemory(dim=4)
    memory.add([0.2, 0.1, 0.0, 0.0], threshold=0.01)
    memory.add([0.8, 0.2, 0.0, 0.0], threshold=0.01)
    memory.add([0.9, 0.4, 0.0, 0.0], threshold=0.01)
    return memory


def test_phase_diagram_region_order_is_deterministic() -> None:
    motif_clusters = [
        {"centroid": [1.0, 0.2, 0.1, 0.0], "motif_ids": [3, 7]},
        {"centroid": [0.5, 0.3, 0.2, 0.1], "motif_ids": [2]},
    ]
    tiers = [1, 2]
    diagram = generate_spectral_phase_diagram(motif_clusters, tiers, _memory())
    region_ids = [r["region_id"] for r in diagram["regions"]]
    assert region_ids == sorted(region_ids)


def test_boundary_detection_stability() -> None:
    p1 = detect_phase_boundaries(_memory())
    p2 = detect_phase_boundaries(_memory())
    assert json.dumps(p1, sort_keys=True) == json.dumps(p2, sort_keys=True)


def test_phase_diagram_reproducibility() -> None:
    motif_clusters = [
        {"centroid": np.asarray([0.1, 0.2, 0.3, 0.4], dtype=np.float64), "motif_ids": [0]},
        {"centroid": np.asarray([0.4, 0.3, 0.2, 0.1], dtype=np.float64), "motif_ids": [1]},
    ]
    tiers = [0, 1]
    d1 = generate_spectral_phase_diagram(motif_clusters, tiers, _memory())
    d2 = generate_spectral_phase_diagram(motif_clusters, tiers, _memory())
    assert json.dumps(d1, sort_keys=True) == json.dumps(d2, sort_keys=True)
