from __future__ import annotations

import numpy as np

from src.qec.analysis.spectral_motif_clustering import cluster_spectral_motifs
from src.qec.discovery.motif_library import SpectralMotifLibrary


def test_cluster_spectral_motifs_deterministic() -> None:
    motifs = [
        {"motif_id": 2, "spectral_signature": np.array([2.0, 0.0], dtype=np.float64), "frequency": 1},
        {"motif_id": 0, "spectral_signature": np.array([0.0, 0.0], dtype=np.float64), "frequency": 2},
        {"motif_id": 1, "spectral_signature": np.array([1.0, 0.0], dtype=np.float64), "frequency": 3},
    ]
    c1 = cluster_spectral_motifs(motifs, k=2)
    c2 = cluster_spectral_motifs(motifs, k=2)

    assert [c["cluster_id"] for c in c1] == [c["cluster_id"] for c in c2]
    assert [c["motif_ids"] for c in c1] == [c["motif_ids"] for c in c2]
    for c in c1:
        assert np.asarray(c["centroid"], dtype=np.float64).dtype == np.float64


def test_motif_library_cluster_query_tiebreak() -> None:
    lib = SpectralMotifLibrary()
    lib.add_motifs(
        [
            {"motif_id": 1, "spectral_signature": np.array([0.0, 0.0], dtype=np.float64), "frequency": 1},
            {"motif_id": 2, "spectral_signature": np.array([2.0, 0.0], dtype=np.float64), "frequency": 1},
        ]
    )
    clusters = lib.cluster_motifs(k=2)
    assert len(clusters) == 2

    found = lib.get_cluster(np.array([1.0, 0.0], dtype=np.float64))
    assert found is not None
    assert int(found["cluster_id"]) == 0


def test_cluster_centroid_frequency_weighting() -> None:
    motifs = [
        {"motif_id": 0, "spectral_signature": np.array([0.0, 0.0], dtype=np.float64), "frequency": 1},
        {"motif_id": 1, "spectral_signature": np.array([10.0, 0.0], dtype=np.float64), "frequency": 3},
    ]
    clusters = cluster_spectral_motifs(motifs, k=1)
    assert len(clusters) == 1
    centroid = np.asarray(clusters[0]["centroid"], dtype=np.float64)
    assert np.allclose(centroid, np.array([7.5, 0.0], dtype=np.float64))
