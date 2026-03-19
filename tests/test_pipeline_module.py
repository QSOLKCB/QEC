"""Tests for the pipeline orchestration module (v71.0.2).

Verifies that build_full_pipeline produces identical output to
the inline pipeline it replaced, and enforces pipeline invariants.
"""

import copy
import json

import pytest

from qec.experiments.benchmark_stress import (
    _run_single_genome_suite,
    normalize_decoder_genome,
    run_benchmark_stress,
)
from qec.modules.pipeline.pipeline import build_full_pipeline


class TestPipelineEquivalence:
    """build_full_pipeline must produce identical output to run_benchmark_stress."""

    def test_single_mode_equivalence(self):
        """Single-genome pipeline output matches run_benchmark_stress."""
        genome = normalize_decoder_genome(None)
        suite = _run_single_genome_suite(10, 8, "benchmark_stress_v71.0.0", genome)

        # Build via pipeline
        pipeline_result = build_full_pipeline([suite], mode="single")

        # Build via run_benchmark_stress
        direct_result = run_benchmark_stress(n_vars=10, n_iters=8)

        # Strip timing (non-deterministic)
        for s in pipeline_result["scenarios"]:
            s.pop("timing", None)
        for s in direct_result["scenarios"]:
            s.pop("timing", None)

        j_pipeline = json.dumps(pipeline_result, sort_keys=True)
        j_direct = json.dumps(direct_result, sort_keys=True)
        assert j_pipeline == j_direct, "Pipeline output differs from direct output"

    def test_sweep_mode_equivalence(self):
        """Sweep pipeline output matches run_benchmark_stress."""
        genomes_raw = [
            {"clip_value": 1.0, "damping": 0.0},
            {"clip_value": 5.0, "damping": 0.3},
        ]
        genomes = [normalize_decoder_genome(g) for g in genomes_raw]

        suites = [
            _run_single_genome_suite(10, 8, "benchmark_stress_v71.0.0", g)
            for g in genomes
        ]
        pipeline_result = build_full_pipeline(suites, mode="sweep")

        direct_result = run_benchmark_stress(
            n_vars=10, n_iters=8, genomes=genomes_raw,
        )

        # Strip timing
        for suite in pipeline_result["results"]:
            for s in suite["scenarios"]:
                s.pop("timing", None)
        for suite in direct_result["results"]:
            for s in suite["scenarios"]:
                s.pop("timing", None)

        j_pipeline = json.dumps(pipeline_result, sort_keys=True)
        j_direct = json.dumps(direct_result, sort_keys=True)
        assert j_pipeline == j_direct, "Sweep pipeline output differs"


class TestPipelineDeterminism:
    """build_full_pipeline must be deterministic."""

    def test_determinism(self):
        """Two identical pipeline invocations produce identical output."""
        genome = normalize_decoder_genome(None)

        suite1 = _run_single_genome_suite(10, 8, "benchmark_stress_v71.0.0", genome)
        result1 = build_full_pipeline([suite1], mode="single")
        for s in result1["scenarios"]:
            s.pop("timing", None)

        suite2 = _run_single_genome_suite(10, 8, "benchmark_stress_v71.0.0", genome)
        result2 = build_full_pipeline([suite2], mode="single")
        for s in result2["scenarios"]:
            s.pop("timing", None)

        j1 = json.dumps(result1, sort_keys=True)
        j2 = json.dumps(result2, sort_keys=True)
        assert j1 == j2, "Pipeline is not deterministic"


class TestPipelineInvariants:
    """Pipeline invariant enforcement (v71.0.2)."""

    def test_invalid_mode_raises(self):
        """Invalid mode must raise ValueError."""
        genome = normalize_decoder_genome(None)
        suite = _run_single_genome_suite(10, 8, "benchmark_stress_v71.0.0", genome)
        with pytest.raises(ValueError, match="Invalid pipeline mode"):
            build_full_pipeline([suite], mode="bogus")

    def test_empty_suites_raises(self):
        """Empty suites list must raise ValueError."""
        with pytest.raises(ValueError, match="non-empty"):
            build_full_pipeline([], mode="single")

    def test_input_immutability(self):
        """Input suite dict must not be mutated by the pipeline."""
        genome = normalize_decoder_genome(None)
        suite = _run_single_genome_suite(10, 8, "benchmark_stress_v71.0.0", genome)
        snapshot = copy.deepcopy(suite)
        build_full_pipeline([suite], mode="single")
        assert suite == snapshot, "Pipeline mutated the input suite dict"
