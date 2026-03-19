"""
Determinism and golden vector tests for v3.1.2 interop layer.

Tests:
1. Canonical JSON produces byte-identical output for same input.
2. Artifact hash is stable across runs.
3. QEC-native runner produces identical records on double-run.
4. Golden vector: known-good output for a fixed config.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from bench.interop.serialize import canonical_json, artifact_hash, config_hash
from bench.interop.runners import run_qec_native
from bench.schema import validate_interop_record


# ── Golden vector ────────────────────────────────────────────────────
# This is the canonical JSON output for a specific config.
# If this changes, either the config changed or determinism is broken.

_GOLDEN_CONFIG = {
    "distances": [3],
    "p_values": [0.001],
    "trials": 50,
    "max_iters": 10,
    "seed": 42,
    "runtime_mode": "off",
    "deterministic_metadata": True,
}

# The expected artifact hash for the golden config.
# Updated whenever the golden config or deterministic behavior changes.
_GOLDEN_ARTIFACT_HASH = (
    "86babd2ec81daa165d3ce778b9eb71a3766667484e1c51a2000642ae08ec9569"
)


class TestCanonicalJson:
    """Test canonical JSON serialization."""

    def test_sorted_keys(self):
        obj = {"z": 1, "a": 2, "m": 3}
        result = canonical_json(obj)
        assert result == '{"a":2,"m":3,"z":1}'

    def test_compact_separators(self):
        obj = {"key": [1, 2, 3]}
        result = canonical_json(obj)
        assert " " not in result
        assert result == '{"key":[1,2,3]}'

    def test_numpy_conversion(self):
        obj = {"val": np.int64(42), "arr": np.array([1.0, 2.0])}
        result = canonical_json(obj)
        parsed = json.loads(result)
        assert parsed == {"arr": [1.0, 2.0], "val": 42}

    def test_nested_dict_sorted(self):
        obj = {"outer": {"z": 1, "a": 2}}
        result = canonical_json(obj)
        assert result == '{"outer":{"a":2,"z":1}}'

    def test_byte_identical_same_input(self):
        """Same input dict must produce identical bytes."""
        obj = {"x": 42, "y": [1, 2, 3], "z": {"a": 1}}
        r1 = canonical_json(obj)
        r2 = canonical_json(obj)
        assert r1 == r2
        assert r1.encode("utf-8") == r2.encode("utf-8")


class TestArtifactHash:
    """Test artifact hash stability."""

    def test_same_input_same_hash(self):
        obj = {"key": "value", "num": 42}
        h1 = artifact_hash(obj)
        h2 = artifact_hash(obj)
        assert h1 == h2

    def test_different_input_different_hash(self):
        obj1 = {"key": "value1"}
        obj2 = {"key": "value2"}
        assert artifact_hash(obj1) != artifact_hash(obj2)

    def test_hash_is_sha256_hex(self):
        h = artifact_hash({"test": True})
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)


class TestConfigHash:
    """Test config hash stability."""

    def test_same_config_same_hash(self):
        cfg = {"seed": 42, "trials": 100}
        h1 = config_hash(cfg)
        h2 = config_hash(cfg)
        assert h1 == h2

    def test_key_order_independent(self):
        """Config hash must be order-independent (sorted keys)."""
        cfg1 = {"b": 2, "a": 1}
        cfg2 = {"a": 1, "b": 2}
        assert config_hash(cfg1) == config_hash(cfg2)


class TestQecNativeDeterminism:
    """Test that QEC-native runner is deterministic."""

    def test_double_run_identical(self):
        """Running the same config twice must produce identical records."""
        kwargs = dict(
            distances=[3],
            p_values=[0.001],
            trials=50,
            max_iters=10,
            seed=42,
            runtime_mode="off",
            deterministic_metadata=True,
        )

        records1 = run_qec_native(**kwargs)
        records2 = run_qec_native(**kwargs)

        assert len(records1) == len(records2)
        for r1, r2 in zip(records1, records2):
            j1 = canonical_json(r1)
            j2 = canonical_json(r2)
            assert j1 == j2, "QEC-native runner is not deterministic!"

    def test_artifact_hash_stable(self):
        """Artifact hash must be identical across runs."""
        kwargs = dict(
            distances=[3],
            p_values=[0.001],
            trials=50,
            max_iters=10,
            seed=42,
            runtime_mode="off",
            deterministic_metadata=True,
        )

        records = run_qec_native(**kwargs)
        h1 = records[0]["determinism"]["artifact_hash"]

        records2 = run_qec_native(**kwargs)
        h2 = records2[0]["determinism"]["artifact_hash"]

        assert h1 == h2


class TestGoldenVector:
    """Test golden reference vector stability."""

    def test_golden_artifact_hash(self):
        """The golden config must produce the known artifact hash."""
        records = run_qec_native(**_GOLDEN_CONFIG)
        assert len(records) == 1

        rec = records[0]
        validate_interop_record(rec)

        actual_hash = rec["determinism"]["artifact_hash"]
        assert actual_hash == _GOLDEN_ARTIFACT_HASH, (
            f"Golden vector hash mismatch!\n"
            f"  Expected: {_GOLDEN_ARTIFACT_HASH}\n"
            f"  Got:      {actual_hash}\n"
            "This means deterministic output has changed."
        )

    def test_golden_record_structure(self):
        """Golden record must have expected structure."""
        records = run_qec_native(**_GOLDEN_CONFIG)
        rec = records[0]

        assert rec["schema_version"] == "3.1.2"
        assert rec["benchmark_kind"] == "direct_comparison"
        assert rec["code_family"] == "qldpc_css"
        assert rec["representation"] == "pcm"
        assert rec["noise_model"] == "bsc_bitflip"
        assert rec["channel_model"] == "oracle"
        assert rec["tool"]["name"] == "qec_bp"
        assert rec["tool"]["category"] == "native"
        assert rec["seed"] == 42
        assert rec["trials"] == 50
        assert isinstance(rec["results"]["logical_error_rate"], float)
        assert isinstance(rec["determinism"]["stable_sweep_hash"], str)

    def test_golden_logical_error_rate(self):
        """Golden record must produce known logical error rate."""
        records = run_qec_native(**_GOLDEN_CONFIG)
        rec = records[0]
        # At d=3, p=0.001, trials=50: expect low error rate
        assert rec["results"]["logical_error_rate"] == 0.0


class TestReportDeterminism:
    """Test that report output is deterministic."""

    def test_report_double_run_identical(self):
        """Same suite result must produce identical report string."""
        from bench.interop.baselines import run_baseline_suite
        from bench.interop.report import generate_report

        kwargs = dict(
            native_distances=[3],
            native_p_values=[0.001],
            native_trials=50,
            native_max_iters=10,
            seed=42,
            runtime_mode="off",
            deterministic_metadata=True,
        )

        suite1 = run_baseline_suite(**kwargs)
        report1 = generate_report(suite1)

        suite2 = run_baseline_suite(**kwargs)
        report2 = generate_report(suite2)

        assert report1 == report2, "Report output is not deterministic!"
