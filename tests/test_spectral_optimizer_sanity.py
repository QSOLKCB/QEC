"""
Tests for v7.5.0 — Spectral Optimizer Sanity Experiment + Predictor Probe.

Verifies:
  - Determinism: repeated runs produce identical JSON output
  - Decoder isolation: module does not import src/qec/decoder/
  - Predictor integration: predicted_failure fields exist
  - JSON stability: serialization roundtrip succeeds
  - Output schema: all required keys present with correct types
"""

from __future__ import annotations

import ast
import json
import os
import sys

import numpy as np
import pytest

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from qec.experiments.spectral_optimizer_sanity_experiment import (
    run_spectral_optimizer_sanity_experiment,
    print_sanity_report,
    _predict_instability,
    _decode_trials,
)


# ── Toy inputs ────────────────────────────────────────────────────────


def _make_H() -> np.ndarray:
    """Simple 3x5 parity-check matrix."""
    return np.array([
        [1, 1, 0, 1, 0],
        [0, 1, 1, 0, 1],
        [1, 0, 1, 1, 0],
    ], dtype=np.float64)


def _toy_syndrome(H: np.ndarray, e: np.ndarray) -> np.ndarray:
    """Compute syndrome s = H @ e mod 2."""
    return ((H.astype(np.int32) @ np.asarray(e).astype(np.int32)) % 2).astype(
        np.uint8
    )


def _toy_decode(
    H: np.ndarray,
    llr: np.ndarray,
    *,
    max_iters: int = 50,
    mode: str = "min_sum",
    schedule: str = "flooding",
    syndrome_vec=None,
    energy_trace: bool = False,
    llr_history: int = 0,
) -> tuple:
    """Deterministic stub decoder: hard-decision on LLR sign."""
    n = H.shape[1]
    correction = np.zeros(n, dtype=np.uint8)
    for i in range(n):
        if llr[i] < 0:
            correction[i] = 1
    return (correction, 1)


def _make_instances(H: np.ndarray, n_trials: int = 5) -> list[dict]:
    """Generate deterministic toy instances."""
    rng = np.random.default_rng(42)
    n = H.shape[1]
    instances = []
    for _ in range(n_trials):
        e = (rng.random(n) < 0.1).astype(np.uint8)
        s = _toy_syndrome(H, e)
        base_llr = np.log(0.9 / 0.1)
        llr = base_llr * (1 - 2 * e.astype(np.float64))
        instances.append({"e": e, "s": s, "llr": llr})
    return instances


# ── Test: Determinism ─────────────────────────────────────────────────


class TestDeterminism:
    """Repeated runs produce identical JSON output."""

    def test_full_experiment_deterministic(self):
        H = _make_H()
        instances = _make_instances(H)
        r1 = run_spectral_optimizer_sanity_experiment(
            H, instances, _toy_decode, _toy_syndrome,
            max_iters=10, optimizer_max_iterations=3,
        )
        r2 = run_spectral_optimizer_sanity_experiment(
            H, instances, _toy_decode, _toy_syndrome,
            max_iters=10, optimizer_max_iterations=3,
        )
        j1 = json.dumps(r1, sort_keys=True)
        j2 = json.dumps(r2, sort_keys=True)
        assert j1 == j2

    def test_predict_instability_deterministic(self):
        H = _make_H()
        p1 = _predict_instability(H)
        p2 = _predict_instability(H)
        assert p1 == p2


# ── Test: Decoder Isolation ──────────────────────────────────────────


class TestDecoderIsolation:
    """Module does not import src/qec/decoder/."""

    def test_no_decoder_import(self):
        module_path = os.path.join(
            _repo_root,
            "src", "qec", "experiments",
            "spectral_optimizer_sanity_experiment.py",
        )
        with open(module_path, "r") as f:
            source = f.read()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                if isinstance(node, ast.ImportFrom) and node.module:
                    assert not node.module.startswith(
                        "src.qec.decoder"
                    ), f"Forbidden import: {node.module}"


# ── Test: Predictor Integration ──────────────────────────────────────


class TestPredictorIntegration:
    """Verify predicted_failure fields exist and are booleans."""

    def test_predictor_fields_present(self):
        H = _make_H()
        instances = _make_instances(H)
        report = run_spectral_optimizer_sanity_experiment(
            H, instances, _toy_decode, _toy_syndrome,
            max_iters=10, optimizer_max_iterations=3,
        )
        probe = report["predictor_probe"]
        assert "predicted_failure" in probe
        assert "actual_failure" in probe
        assert "predicted_failure_optimized" in probe
        assert "actual_failure_optimized" in probe
        assert isinstance(probe["predicted_failure"], bool)
        assert isinstance(probe["actual_failure"], bool)
        assert isinstance(probe["predicted_failure_optimized"], bool)
        assert isinstance(probe["actual_failure_optimized"], bool)


# ── Test: JSON Stability ─────────────────────────────────────────────


class TestJSONStability:
    """Serialization roundtrip must succeed."""

    def test_roundtrip(self):
        H = _make_H()
        instances = _make_instances(H)
        report = run_spectral_optimizer_sanity_experiment(
            H, instances, _toy_decode, _toy_syndrome,
            max_iters=10, optimizer_max_iterations=3,
        )
        serialized = json.dumps(report, sort_keys=True)
        deserialized = json.loads(serialized)
        re_serialized = json.dumps(deserialized, sort_keys=True)
        assert serialized == re_serialized


# ── Test: Output Schema ──────────────────────────────────────────────


class TestOutputSchema:
    """All required keys present with correct types."""

    def test_top_level_keys(self):
        H = _make_H()
        instances = _make_instances(H)
        report = run_spectral_optimizer_sanity_experiment(
            H, instances, _toy_decode, _toy_syndrome,
            max_iters=10, optimizer_max_iterations=3,
        )
        assert "baseline" in report
        assert "optimized" in report
        assert "optimizer" in report
        assert "structure" in report
        assert "predictor_probe" in report

    def test_baseline_keys(self):
        H = _make_H()
        instances = _make_instances(H)
        report = run_spectral_optimizer_sanity_experiment(
            H, instances, _toy_decode, _toy_syndrome,
            max_iters=10, optimizer_max_iterations=3,
        )
        b = report["baseline"]
        assert "fer" in b
        assert "wer" in b
        assert "avg_iterations" in b
        assert "actual_failure" in b
        assert isinstance(b["fer"], float)
        assert isinstance(b["wer"], float)
        assert isinstance(b["avg_iterations"], float)

    def test_structure_keys(self):
        H = _make_H()
        instances = _make_instances(H)
        report = run_spectral_optimizer_sanity_experiment(
            H, instances, _toy_decode, _toy_syndrome,
            max_iters=10, optimizer_max_iterations=3,
        )
        s = report["structure"]
        assert "initial_instability_score" in s
        assert "final_instability_score" in s
        assert "instability_delta" in s
        assert isinstance(s["initial_instability_score"], float)
        assert isinstance(s["instability_delta"], float)

    def test_optimizer_repair_count(self):
        H = _make_H()
        instances = _make_instances(H)
        report = run_spectral_optimizer_sanity_experiment(
            H, instances, _toy_decode, _toy_syndrome,
            max_iters=10, optimizer_max_iterations=3,
        )
        assert isinstance(report["optimizer"]["repair_count"], int)
        assert report["optimizer"]["repair_count"] >= 0


# ── Test: No Input Mutation ──────────────────────────────────────────


class TestNoInputMutation:
    """Experiment must not mutate input arrays."""

    def test_H_unchanged(self):
        H = _make_H()
        H_copy = H.copy()
        instances = _make_instances(H)
        run_spectral_optimizer_sanity_experiment(
            H, instances, _toy_decode, _toy_syndrome,
            max_iters=10, optimizer_max_iterations=3,
        )
        assert np.array_equal(H, H_copy)

    def test_instances_unchanged(self):
        H = _make_H()
        instances = _make_instances(H)
        originals = [
            {"e": inst["e"].copy(), "s": inst["s"].copy(), "llr": inst["llr"].copy()}
            for inst in instances
        ]
        run_spectral_optimizer_sanity_experiment(
            H, instances, _toy_decode, _toy_syndrome,
            max_iters=10, optimizer_max_iterations=3,
        )
        for orig, inst in zip(originals, instances):
            assert np.array_equal(orig["e"], inst["e"])
            assert np.array_equal(orig["s"], inst["s"])
            assert np.array_equal(orig["llr"], inst["llr"])


# ── Test: Console Report ─────────────────────────────────────────────


class TestConsoleReport:
    """Verify print_sanity_report runs without error."""

    def test_print_runs(self, capsys):
        H = _make_H()
        instances = _make_instances(H)
        report = run_spectral_optimizer_sanity_experiment(
            H, instances, _toy_decode, _toy_syndrome,
            max_iters=10, optimizer_max_iterations=3,
        )
        print_sanity_report(report)
        captured = capsys.readouterr()
        assert "Spectral Optimizer Sanity" in captured.out
        assert "Baseline" in captured.out
        assert "Optimized" in captured.out
        assert "Predictor" not in captured.out or "Predicted Failure" in captured.out
