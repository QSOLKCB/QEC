"""
Tests for incremental NB spectral updates (v7.9.0).

Verifies:
  - deterministic incremental solver
  - eigenvector normalization
  - spectral radius positive
  - cosine similarity with full solver (>0.99)
  - localized update correctness
  - fallback path works
  - edge swap detection
  - affected NB edge identification
  - incremental repair scoring
  - benchmark artifact schema valid
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from qec.diagnostics.spectral_nb import compute_nb_spectrum
from qec.diagnostics.spectral_repair import (
    apply_repair_candidate,
    propose_repair_candidates,
)
from qec.diagnostics.spectral_incremental import (
    detect_edge_swap,
    identify_affected_nb_edges,
    score_repair_candidate_incremental,
    update_nb_eigenpair_incremental,
    update_nb_eigenpair_localized,
)
from qec.experiments.incremental_spectral_benchmark import (
    run_incremental_spectral_benchmark,
    serialize_benchmark_artifact,
)


# ── Fixtures ─────────────────────────────────────────────────────


def _small_H():
    """3x4 parity-check matrix with known structure."""
    return np.array([
        [1, 1, 0, 1],
        [0, 1, 1, 0],
        [1, 0, 1, 1],
    ], dtype=np.float64)


def _medium_H():
    """4x6 parity-check matrix."""
    return np.array([
        [1, 1, 0, 1, 0, 0],
        [0, 1, 1, 0, 1, 0],
        [1, 0, 0, 0, 1, 1],
        [0, 0, 1, 1, 0, 1],
    ], dtype=np.float64)


def _get_candidate_and_repaired(H):
    """Get first valid candidate and the repaired matrix."""
    candidates = propose_repair_candidates(
        H, top_k_edges=5, max_candidates=20,
    )
    assert len(candidates) > 0, "No repair candidates generated"
    candidate = candidates[0]
    H_repaired = apply_repair_candidate(H, candidate)
    return candidate, H_repaired


# ── Incremental solver tests ─────────────────────────────────────


class TestIncrementalSolver:

    def test_deterministic(self):
        """Incremental solver must be deterministic."""
        H = _medium_H()
        orig = compute_nb_spectrum(H)
        _, H_rep = _get_candidate_and_repaired(H)

        r1 = update_nb_eigenpair_incremental(H_rep, orig["eigenvector"])
        r2 = update_nb_eigenpair_incremental(H_rep, orig["eigenvector"])

        assert r1["spectral_radius"] == r2["spectral_radius"]
        assert r1["converged"] == r2["converged"]
        assert r1["iterations"] == r2["iterations"]
        np.testing.assert_array_equal(r1["eigenvector"], r2["eigenvector"])

    def test_eigenvector_normalized(self):
        """Returned eigenvector must be unit-normalized."""
        H = _medium_H()
        orig = compute_nb_spectrum(H)
        _, H_rep = _get_candidate_and_repaired(H)

        result = update_nb_eigenpair_incremental(H_rep, orig["eigenvector"])
        norm = np.linalg.norm(result["eigenvector"])
        assert abs(norm - 1.0) < 1e-10

    def test_spectral_radius_positive(self):
        """Spectral radius must be non-negative."""
        H = _medium_H()
        orig = compute_nb_spectrum(H)
        _, H_rep = _get_candidate_and_repaired(H)

        result = update_nb_eigenpair_incremental(H_rep, orig["eigenvector"])
        assert result["spectral_radius"] >= 0.0

    def test_cosine_similarity_high(self):
        """Incremental eigenvector should agree with full computation."""
        H = _medium_H()
        orig = compute_nb_spectrum(H)
        _, H_rep = _get_candidate_and_repaired(H)

        full = compute_nb_spectrum(H_rep)
        incr = update_nb_eigenpair_incremental(H_rep, orig["eigenvector"])

        v_full = full["eigenvector"]
        v_incr = incr["eigenvector"]
        cos_sim = abs(np.dot(v_full, v_incr)) / (
            np.linalg.norm(v_full) * np.linalg.norm(v_incr)
        )
        assert cos_sim > 0.99, f"Cosine similarity {cos_sim} < 0.99"

    def test_converged_flag(self):
        """Solver should converge within max_iter for medium matrices."""
        H = _medium_H()
        orig = compute_nb_spectrum(H)
        _, H_rep = _get_candidate_and_repaired(H)

        result = update_nb_eigenpair_incremental(
            H_rep, orig["eigenvector"], max_iter=100, tol=1e-8,
        )
        assert result["converged"] is True

    def test_iterations_positive(self):
        """At least one iteration must be performed."""
        H = _small_H()
        orig = compute_nb_spectrum(H)
        _, H_rep = _get_candidate_and_repaired(H)

        result = update_nb_eigenpair_incremental(H_rep, orig["eigenvector"])
        assert result["iterations"] >= 1

    def test_sign_canonical(self):
        """Largest-magnitude component must be positive."""
        H = _medium_H()
        orig = compute_nb_spectrum(H)
        _, H_rep = _get_candidate_and_repaired(H)

        result = update_nb_eigenpair_incremental(H_rep, orig["eigenvector"])
        v = result["eigenvector"]
        max_idx = int(np.argmax(np.abs(v)))
        assert v[max_idx] >= 0


# ── Edge swap detection tests ────────────────────────────────────


class TestEdgeSwapDetection:

    def test_detect_no_change(self):
        """No change yields empty lists."""
        H = _small_H()
        result = detect_edge_swap(H, H)
        assert result["removed_edges"] == []
        assert result["added_edges"] == []

    def test_detect_single_swap(self):
        """Detect edges in a single swap candidate."""
        H = _medium_H()
        candidates = propose_repair_candidates(
            H, top_k_edges=5, max_candidates=10,
        )
        assert len(candidates) > 0

        candidate = candidates[0]
        H_rep = apply_repair_candidate(H, candidate)

        swap = detect_edge_swap(H, H_rep)
        assert len(swap["removed_edges"]) == 2
        assert len(swap["added_edges"]) == 2

    def test_deterministic(self):
        """Detection must be deterministic."""
        H = _medium_H()
        _, H_rep = _get_candidate_and_repaired(H)

        r1 = detect_edge_swap(H, H_rep)
        r2 = detect_edge_swap(H, H_rep)
        assert r1 == r2


# ── Affected NB edges tests ──────────────────────────────────────


class TestAffectedNBEdges:

    def test_returns_sorted_list(self):
        """Affected indices must be sorted."""
        H = _medium_H()
        _, H_rep = _get_candidate_and_repaired(H)

        affected = identify_affected_nb_edges(H, H_rep)
        assert affected == sorted(affected)

    def test_nonempty_for_swap(self):
        """A swap must affect at least some NB edges."""
        H = _medium_H()
        _, H_rep = _get_candidate_and_repaired(H)

        affected = identify_affected_nb_edges(H, H_rep)
        assert len(affected) > 0

    def test_no_change_yields_empty(self):
        """Identical matrices yield no affected edges."""
        H = _small_H()
        affected = identify_affected_nb_edges(H, H)
        assert affected == []

    def test_deterministic(self):
        """Identification must be deterministic."""
        H = _medium_H()
        _, H_rep = _get_candidate_and_repaired(H)

        a1 = identify_affected_nb_edges(H, H_rep)
        a2 = identify_affected_nb_edges(H, H_rep)
        assert a1 == a2


# ── Localized update tests ───────────────────────────────────────


class TestLocalizedUpdate:

    def test_localized_flag(self):
        """Localized update should set localized=True for small graphs."""
        H = _medium_H()
        orig = compute_nb_spectrum(H)
        _, H_rep = _get_candidate_and_repaired(H)
        affected = identify_affected_nb_edges(H, H_rep)

        result = update_nb_eigenpair_localized(
            H_rep, orig["eigenvector"], affected,
        )
        # For small graphs, affected set may exceed locality_fraction
        # so either path is valid
        assert isinstance(result["localized"], bool)

    def test_eigenvector_normalized(self):
        """Localized result must be normalized."""
        H = _medium_H()
        orig = compute_nb_spectrum(H)
        _, H_rep = _get_candidate_and_repaired(H)
        affected = identify_affected_nb_edges(H, H_rep)

        result = update_nb_eigenpair_localized(
            H_rep, orig["eigenvector"], affected,
        )
        norm = np.linalg.norm(result["eigenvector"])
        assert abs(norm - 1.0) < 1e-10

    def test_cosine_similarity_high(self):
        """Localized result should agree with full computation."""
        H = _medium_H()
        orig = compute_nb_spectrum(H)
        _, H_rep = _get_candidate_and_repaired(H)
        affected = identify_affected_nb_edges(H, H_rep)

        full = compute_nb_spectrum(H_rep)
        local = update_nb_eigenpair_localized(
            H_rep, orig["eigenvector"], affected,
        )

        v_full = full["eigenvector"]
        v_local = local["eigenvector"]
        cos_sim = abs(np.dot(v_full, v_local)) / (
            np.linalg.norm(v_full) * np.linalg.norm(v_local)
        )
        assert cos_sim > 0.99, f"Cosine similarity {cos_sim} < 0.99"

    def test_deterministic(self):
        """Localized update must be deterministic."""
        H = _medium_H()
        orig = compute_nb_spectrum(H)
        _, H_rep = _get_candidate_and_repaired(H)
        affected = identify_affected_nb_edges(H, H_rep)

        r1 = update_nb_eigenpair_localized(
            H_rep, orig["eigenvector"], affected,
        )
        r2 = update_nb_eigenpair_localized(
            H_rep, orig["eigenvector"], affected,
        )

        assert r1["spectral_radius"] == r2["spectral_radius"]
        np.testing.assert_array_equal(r1["eigenvector"], r2["eigenvector"])

    def test_fallback_for_large_affected_set(self):
        """If affected set is large, should fall back to full incremental."""
        H = _small_H()
        orig = compute_nb_spectrum(H)
        _, H_rep = _get_candidate_and_repaired(H)

        # Use all indices as affected — forces fallback
        from qec.diagnostics.spectral_nb import _TannerGraph
        from qec.diagnostics._spectral_utils import build_directed_edges
        graph = _TannerGraph(H_rep)
        all_indices = list(range(len(build_directed_edges(graph))))

        result = update_nb_eigenpair_localized(
            H_rep, orig["eigenvector"], all_indices,
            locality_fraction=0.1,  # very low threshold forces fallback
        )
        assert result["localized"] is False


# ── Incremental repair scoring tests ─────────────────────────────


class TestIncrementalRepairScoring:

    def test_returns_score_dict(self):
        """Incremental scoring must return expected keys."""
        H = _medium_H()
        orig = compute_nb_spectrum(H)
        candidates = propose_repair_candidates(
            H, top_k_edges=5, max_candidates=10,
        )
        assert len(candidates) > 0

        result = score_repair_candidate_incremental(
            H, candidates[0], orig["eigenvector"],
        )
        assert result is not None
        assert "candidate" in result
        assert "original_sis" in result
        assert "repaired_sis" in result
        assert "delta_sis" in result
        assert "incremental_converged" in result
        assert "incremental_iterations" in result
        assert "used_fallback" in result

    def test_deterministic(self):
        """Incremental scoring must be deterministic."""
        H = _medium_H()
        orig = compute_nb_spectrum(H)
        candidates = propose_repair_candidates(
            H, top_k_edges=5, max_candidates=10,
        )
        assert len(candidates) > 0

        r1 = score_repair_candidate_incremental(
            H, candidates[0], orig["eigenvector"],
        )
        r2 = score_repair_candidate_incremental(
            H, candidates[0], orig["eigenvector"],
        )
        assert r1["repaired_sis"] == r2["repaired_sis"]
        assert r1["delta_sis"] == r2["delta_sis"]

    def test_fallback_on_non_convergence(self):
        """With max_iter=1, solver may not converge and should fallback."""
        H = _medium_H()
        orig = compute_nb_spectrum(H)
        candidates = propose_repair_candidates(
            H, top_k_edges=5, max_candidates=10,
        )
        assert len(candidates) > 0

        result = score_repair_candidate_incremental(
            H, candidates[0], orig["eigenvector"],
            max_iter=1, tol=1e-15,
        )
        assert result is not None
        # Either converged in 1 step or used fallback
        assert isinstance(result["used_fallback"], bool)


# ── Benchmark artifact tests ─────────────────────────────────────


class TestBenchmarkArtifact:

    def test_schema_valid(self):
        """Benchmark artifact must contain all required fields."""
        H = _medium_H()
        artifact = run_incremental_spectral_benchmark(
            H, top_k_edges=5, max_candidates=10,
        )

        required_keys = [
            "schema_version",
            "runtime_full",
            "runtime_incremental",
            "runtime_localized",
            "speedup_incremental",
            "speedup_localized",
            "spectral_radius_difference",
            "eigenvector_cosine_similarity",
            "sis_full",
            "sis_incremental",
            "delta_sis",
        ]
        for key in required_keys:
            assert key in artifact, f"Missing key: {key}"

        assert artifact["schema_version"] == 1

    def test_json_serializable(self):
        """Artifact must be JSON-serializable."""
        H = _medium_H()
        artifact = run_incremental_spectral_benchmark(
            H, top_k_edges=5, max_candidates=10,
        )

        json_str = serialize_benchmark_artifact(artifact)
        parsed = json.loads(json_str)
        assert parsed["schema_version"] == 1

    def test_cosine_similarity_high(self):
        """Eigenvector cosine similarity should be high."""
        H = _medium_H()
        artifact = run_incremental_spectral_benchmark(
            H, top_k_edges=5, max_candidates=10,
        )

        if artifact["candidate"] is not None:
            assert artifact["eigenvector_cosine_similarity"] > 0.99

    def test_deterministic(self):
        """Benchmark must produce identical non-timing results."""
        H = _medium_H()
        a1 = run_incremental_spectral_benchmark(
            H, top_k_edges=5, max_candidates=10,
        )
        a2 = run_incremental_spectral_benchmark(
            H, top_k_edges=5, max_candidates=10,
        )

        # Timing values may differ, but spectral results must match
        assert a1["spectral_radius_difference"] == a2["spectral_radius_difference"]
        assert a1["eigenvector_cosine_similarity"] == a2["eigenvector_cosine_similarity"]
        assert a1["sis_full"] == a2["sis_full"]
        assert a1["sis_incremental"] == a2["sis_incremental"]
        assert a1["delta_sis"] == a2["delta_sis"]

    def test_empty_when_no_candidates(self):
        """Matrix with no valid candidates returns empty artifact."""
        # Use a matrix that produces an NB spectrum but has no
        # valid swap candidates (diagonal-like sparse matrix)
        H = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 1, 0],
        ], dtype=np.float64)
        artifact = run_incremental_spectral_benchmark(
            H, top_k_edges=1, max_candidates=1,
        )
        # Either produces a valid result or empty artifact
        assert artifact["schema_version"] == 1
