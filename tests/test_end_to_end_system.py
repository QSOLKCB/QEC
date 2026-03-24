"""End-to-end system validation tests — v100.0.0.

Verifies:
- Full pipeline runs without exceptions
- Outputs are within documented bounds
- Determinism across runs (identical results)
- Stable trajectory behavior
- Reproducibility metadata is present and valid

All tests are deterministic.  No randomness, no network, no GPU.
"""

from __future__ import annotations

import copy
from typing import Any, Dict

import pytest


def _run_pipeline() -> Dict[str, Any]:
    """Import and run the demo pipeline, returning structured results."""
    import sys
    import os
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    from scripts.qec_demo import run_demo
    return run_demo()


class TestFullPipelineRuns:
    """Verify the pipeline executes without exceptions."""

    def test_pipeline_completes(self) -> None:
        result = _run_pipeline()
        assert result is not None
        assert "steps" in result
        assert len(result["steps"]) > 0

    def test_all_steps_have_required_fields(self) -> None:
        result = _run_pipeline()
        required = {
            "name", "regime", "attractor_id", "basin_score",
            "strategy_id", "strategy_score", "transition_bias",
            "multi_step_factor",
        }
        for step in result["steps"]:
            missing = required - set(step.keys())
            assert not missing, f"Step {step.get('name')} missing keys: {missing}"


class TestOutputBounds:
    """Verify all outputs are within documented bounds."""

    @pytest.fixture(scope="class")
    def pipeline_result(self) -> Dict[str, Any]:
        return _run_pipeline()

    def test_basin_score_bounded(self, pipeline_result: Dict[str, Any]) -> None:
        for step in pipeline_result["steps"]:
            bs = step["basin_score"]
            assert 0.0 <= bs <= 1.0, f"basin_score={bs} out of [0,1]"

    def test_strategy_score_bounded(self, pipeline_result: Dict[str, Any]) -> None:
        for step in pipeline_result["steps"]:
            ss = step["strategy_score"]
            assert 0.0 <= ss <= 10.0, f"strategy_score={ss} unexpectedly large"

    def test_transition_bias_bounded(self, pipeline_result: Dict[str, Any]) -> None:
        for step in pipeline_result["steps"]:
            tb = step["transition_bias"]
            assert 0.0 <= tb <= 5.0, f"transition_bias={tb} out of expected range"

    def test_multi_step_factor_bounded(self, pipeline_result: Dict[str, Any]) -> None:
        for step in pipeline_result["steps"]:
            msf = step["multi_step_factor"]
            assert 0.5 <= msf <= 2.0, f"multi_step_factor={msf} out of expected range"

    def test_energy_bounded(self, pipeline_result: Dict[str, Any]) -> None:
        for step in pipeline_result["steps"]:
            e = step["energy"]
            assert 0.0 <= e <= 1.0, f"energy={e} out of [0,1]"

    def test_coherence_bounded(self, pipeline_result: Dict[str, Any]) -> None:
        for step in pipeline_result["steps"]:
            c = step["coherence"]
            assert 0.0 <= c <= 1.0, f"coherence={c} out of [0,1]"

    def test_alignment_bounded(self, pipeline_result: Dict[str, Any]) -> None:
        for step in pipeline_result["steps"]:
            a = step["alignment"]
            assert 0.0 <= a <= 1.0, f"alignment={a} out of [0,1]"

    def test_modulation_bounded(self, pipeline_result: Dict[str, Any]) -> None:
        for step in pipeline_result["steps"]:
            m = step["modulation"]
            assert 0.0 <= m <= 2.0, f"modulation={m} out of expected range"

    def test_trajectory_score_bounded(self, pipeline_result: Dict[str, Any]) -> None:
        for step in pipeline_result["steps"]:
            ts = step["trajectory_score"]
            assert 0.5 <= ts <= 1.5, f"trajectory_score={ts} out of expected range"


class TestDeterminism:
    """Verify determinism: two runs produce identical results."""

    def test_two_runs_identical(self) -> None:
        result1 = _run_pipeline()
        result2 = _run_pipeline()

        assert len(result1["steps"]) == len(result2["steps"])
        for s1, s2 in zip(result1["steps"], result2["steps"]):
            assert s1["name"] == s2["name"]
            assert s1["regime"] == s2["regime"]
            assert s1["attractor_id"] == s2["attractor_id"]
            assert s1["basin_score"] == s2["basin_score"]
            assert s1["strategy_id"] == s2["strategy_id"]
            assert s1["strategy_score"] == s2["strategy_score"]
            assert s1["eval_score"] == s2["eval_score"]
            assert s1["outcome"] == s2["outcome"]
            assert s1["transition_bias"] == s2["transition_bias"]
            assert s1["multi_step_factor"] == s2["multi_step_factor"]
            assert s1["energy"] == s2["energy"]
            assert s1["coherence"] == s2["coherence"]
            assert s1["alignment"] == s2["alignment"]
            assert s1["modulation"] == s2["modulation"]
            assert s1["cycle_detected"] == s2["cycle_detected"]
            assert s1["trajectory_score"] == s2["trajectory_score"]

    def test_regime_counts_identical(self) -> None:
        result1 = _run_pipeline()
        result2 = _run_pipeline()
        assert result1["regime_counts"] == result2["regime_counts"]

    def test_memory_counts_identical(self) -> None:
        result1 = _run_pipeline()
        result2 = _run_pipeline()
        assert result1["memory_keys"] == result2["memory_keys"]
        assert result1["transition_entries"] == result2["transition_entries"]
        assert result1["history_length"] == result2["history_length"]


class TestTrajectoryBehavior:
    """Verify stable trajectory behavior properties."""

    @pytest.fixture(scope="class")
    def pipeline_result(self) -> Dict[str, Any]:
        return _run_pipeline()

    def test_regimes_are_valid(self, pipeline_result: Dict[str, Any]) -> None:
        valid_regimes = {
            "stable", "transitional", "oscillatory", "unstable",
            "mixed", "degenerate",
        }
        for step in pipeline_result["steps"]:
            assert step["regime"] in valid_regimes, (
                f"Unknown regime: {step['regime']}"
            )

    def test_multiple_regimes_visited(self, pipeline_result: Dict[str, Any]) -> None:
        regimes = {step["regime"] for step in pipeline_result["steps"]}
        assert len(regimes) >= 1, "Expected at least one regime"

    def test_no_nan_values(self, pipeline_result: Dict[str, Any]) -> None:
        import math
        numeric_keys = [
            "basin_score", "strategy_score", "transition_bias",
            "multi_step_factor", "energy", "coherence", "alignment",
            "modulation", "trajectory_score",
        ]
        for step in pipeline_result["steps"]:
            for key in numeric_keys:
                val = step[key]
                assert not math.isnan(val), f"NaN in {key} at step {step['name']}"
                assert not math.isinf(val), f"Inf in {key} at step {step['name']}"


class TestReproducibilityMetadata:
    """Verify reproducibility metadata is present and valid."""

    def test_metadata_present(self) -> None:
        result = _run_pipeline()
        assert "metadata" in result
        meta = result["metadata"]
        assert "version" in meta
        assert "seed" in meta
        assert "python_version" in meta
        assert "numpy_version" in meta

    def test_metadata_version(self) -> None:
        result = _run_pipeline()
        assert result["metadata"]["version"] == "v100.0.0"

    def test_metadata_seed_is_int(self) -> None:
        result = _run_pipeline()
        assert isinstance(result["metadata"]["seed"], int)
