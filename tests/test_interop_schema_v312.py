"""
Schema v3.1.2 validation tests.

Tests that validate_interop_record enforces all required fields,
enum constraints, and sub-dict structure.
"""

from __future__ import annotations

import copy

import pytest

from bench.schema import (
    validate_interop_record,
    SCHEMA_VERSION,
    INTEROP_SCHEMA_VERSION,
    _SUPPORTED_SCHEMA_VERSIONS,
    _SUPPORTED_INTEROP_VERSIONS,
)


def _make_valid_interop_record() -> dict:
    """Create a minimal valid v3.1.2 interop record."""
    return {
        "schema_version": "3.1.2",
        "tool": {
            "name": "qec_bp",
            "version": "native",
            "category": "native",
        },
        "benchmark_kind": "direct_comparison",
        "code_family": "qldpc_css",
        "representation": "pcm",
        "seed": 12345,
        "noise_model": "bsc_bitflip",
        "trials": 200,
        "results": {
            "logical_error_rate": 0.05,
            "mean_iters": 3.5,
        },
        "determinism": {
            "canonical_json": {
                "sort_keys": True,
                "separators": [",", ":"],
            },
            "stable_sweep_hash": "a" * 64,
            "artifact_hash": "b" * 64,
        },
    }


class TestSchemaVersionConstants:
    """Verify schema version constants are correct."""

    def test_core_schema_version_unchanged(self):
        """Core SCHEMA_VERSION must remain 3.0.1."""
        assert SCHEMA_VERSION == "3.0.1"

    def test_interop_schema_version(self):
        """INTEROP_SCHEMA_VERSION must be 3.1.2."""
        assert INTEROP_SCHEMA_VERSION == "3.1.2"

    def test_core_supported_versions(self):
        """Core validator still supports 3.0.0 and 3.0.1."""
        assert _SUPPORTED_SCHEMA_VERSIONS == {"3.0.0", "3.0.1"}

    def test_interop_supported_versions(self):
        """Interop validator supports 3.1.2."""
        assert _SUPPORTED_INTEROP_VERSIONS == {"3.1.2"}


class TestValidInteropRecord:
    """Test that valid records pass validation."""

    def test_valid_direct_comparison(self):
        rec = _make_valid_interop_record()
        validate_interop_record(rec)  # Must not raise

    def test_valid_reference_baseline(self):
        rec = _make_valid_interop_record()
        rec["benchmark_kind"] = "reference_baseline"
        rec["code_family"] = "surface_code"
        rec["representation"] = "stim_circuit"
        rec["tool"] = {
            "name": "stim_pymatching",
            "version": "stim=1.12",
            "category": "third_party",
        }
        validate_interop_record(rec)

    def test_skipped_record_accepted(self):
        rec = {
            "status": "skipped",
            "reason": "stim not installed",
            "tool": {"name": "stim", "version": None, "category": "third_party"},
            "benchmark_kind": "reference_baseline",
            "code_family": "repetition",
        }
        validate_interop_record(rec)  # Must not raise


class TestSkippedRecordValidation:
    """Test that skipped records require lightweight structure."""

    def test_bare_skipped_record_fails(self):
        """Skipped record with only status is rejected."""
        rec = {"status": "skipped"}
        with pytest.raises(ValueError, match="reason"):
            validate_interop_record(rec)

    def test_skipped_missing_reason_fails(self):
        """Skipped record without reason is rejected."""
        rec = {
            "status": "skipped",
            "tool": {"name": "stim", "version": None, "category": "third_party"},
            "benchmark_kind": "reference_baseline",
            "code_family": "repetition",
        }
        with pytest.raises(ValueError, match="reason"):
            validate_interop_record(rec)

    def test_skipped_missing_tool_name_fails(self):
        """Skipped record with tool missing name is rejected."""
        rec = {
            "status": "skipped",
            "reason": "stim not installed",
            "tool": {"version": None, "category": "third_party"},
            "benchmark_kind": "reference_baseline",
            "code_family": "repetition",
        }
        with pytest.raises(ValueError, match="tool.*name"):
            validate_interop_record(rec)

    def test_skipped_missing_tool_entirely_fails(self):
        """Skipped record without tool dict is rejected."""
        rec = {
            "status": "skipped",
            "reason": "stim not installed",
            "benchmark_kind": "reference_baseline",
            "code_family": "repetition",
        }
        with pytest.raises(ValueError, match="tool"):
            validate_interop_record(rec)

    def test_skipped_missing_benchmark_kind_fails(self):
        """Skipped record without benchmark_kind is rejected."""
        rec = {
            "status": "skipped",
            "reason": "stim not installed",
            "tool": {"name": "stim", "version": None, "category": "third_party"},
            "code_family": "repetition",
        }
        with pytest.raises(ValueError, match="benchmark_kind"):
            validate_interop_record(rec)

    def test_skipped_missing_code_family_fails(self):
        """Skipped record without code_family is rejected."""
        rec = {
            "status": "skipped",
            "reason": "stim not installed",
            "tool": {"name": "stim", "version": None, "category": "third_party"},
            "benchmark_kind": "reference_baseline",
        }
        with pytest.raises(ValueError, match="code_family"):
            validate_interop_record(rec)

    def test_proper_skipped_record_passes(self):
        """Well-formed skipped record passes validation."""
        rec = {
            "status": "skipped",
            "reason": "pymatching not installed",
            "tool": {"name": "pymatching", "version": None, "category": "third_party"},
            "benchmark_kind": "reference_baseline",
            "code_family": "surface_code",
        }
        validate_interop_record(rec)  # Must not raise


class TestMissingRequiredKeys:
    """Test that missing required keys are caught."""

    @pytest.mark.parametrize("key", [
        "schema_version", "tool", "benchmark_kind", "code_family",
        "representation", "seed", "noise_model", "trials", "results",
        "determinism",
    ])
    def test_missing_top_level_key(self, key):
        rec = _make_valid_interop_record()
        del rec[key]
        with pytest.raises(ValueError, match=f"Missing required key.*{key}"):
            validate_interop_record(rec)


class TestInvalidEnumValues:
    """Test that invalid enum values are caught."""

    def test_invalid_benchmark_kind(self):
        rec = _make_valid_interop_record()
        rec["benchmark_kind"] = "invalid_kind"
        with pytest.raises(ValueError, match="Invalid benchmark_kind"):
            validate_interop_record(rec)

    def test_invalid_code_family(self):
        rec = _make_valid_interop_record()
        rec["code_family"] = "invalid_family"
        with pytest.raises(ValueError, match="Invalid code_family"):
            validate_interop_record(rec)

    def test_invalid_representation(self):
        rec = _make_valid_interop_record()
        rec["representation"] = "invalid_repr"
        with pytest.raises(ValueError, match="Invalid representation"):
            validate_interop_record(rec)

    def test_invalid_tool_category(self):
        rec = _make_valid_interop_record()
        rec["tool"]["category"] = "invalid_cat"
        with pytest.raises(ValueError, match="Invalid tool category"):
            validate_interop_record(rec)

    def test_invalid_schema_version(self):
        rec = _make_valid_interop_record()
        rec["schema_version"] = "99.99.99"
        with pytest.raises(ValueError, match="Unsupported interop schema_version"):
            validate_interop_record(rec)


class TestSubDictValidation:
    """Test that sub-dict structure is enforced."""

    def test_missing_tool_name(self):
        rec = _make_valid_interop_record()
        del rec["tool"]["name"]
        with pytest.raises(ValueError, match="tool missing required keys"):
            validate_interop_record(rec)

    def test_missing_determinism_artifact_hash(self):
        rec = _make_valid_interop_record()
        del rec["determinism"]["artifact_hash"]
        with pytest.raises(ValueError, match="determinism missing required keys"):
            validate_interop_record(rec)

    def test_missing_results_logical_error_rate(self):
        rec = _make_valid_interop_record()
        del rec["results"]["logical_error_rate"]
        with pytest.raises(ValueError, match="logical_error_rate"):
            validate_interop_record(rec)


class TestOptionalChannelModel:
    """Test the optional channel_model field."""

    def test_record_without_channel_model_passes(self):
        """Interop record without channel_model is valid."""
        rec = _make_valid_interop_record()
        assert "channel_model" not in rec
        validate_interop_record(rec)  # Must not raise

    def test_record_with_channel_model_oracle_passes(self):
        """Interop record with channel_model='oracle' is valid."""
        rec = _make_valid_interop_record()
        rec["channel_model"] = "oracle"
        validate_interop_record(rec)  # Must not raise

    def test_record_with_channel_model_string_passes(self):
        """Any string channel_model is accepted."""
        rec = _make_valid_interop_record()
        rec["channel_model"] = "estimated"
        validate_interop_record(rec)  # Must not raise

    def test_record_with_channel_model_non_string_fails(self):
        """Non-string channel_model must be rejected."""
        rec = _make_valid_interop_record()
        rec["channel_model"] = 42
        with pytest.raises(ValueError, match="channel_model must be str"):
            validate_interop_record(rec)


class TestAllValidCodeFamilies:
    """Test that all valid code families are accepted."""

    @pytest.mark.parametrize("cf", [
        "qldpc_css", "surface_code", "repetition", "other",
    ])
    def test_valid_code_family(self, cf):
        rec = _make_valid_interop_record()
        rec["code_family"] = cf
        validate_interop_record(rec)

    @pytest.mark.parametrize("rep", [
        "pcm", "stim_circuit", "stim_dem", "other",
    ])
    def test_valid_representation(self, rep):
        rec = _make_valid_interop_record()
        rec["representation"] = rep
        validate_interop_record(rec)


class TestDeterminismFieldFormats:
    """Test that determinism field format validation is enforced."""

    def test_invalid_stable_sweep_hash_length(self):
        """stable_sweep_hash must be exactly 64 hex chars."""
        rec = _make_valid_interop_record()
        rec["determinism"]["stable_sweep_hash"] = "abc123"
        with pytest.raises(ValueError, match="stable_sweep_hash"):
            validate_interop_record(rec)

    def test_invalid_artifact_hash_length(self):
        """artifact_hash must be exactly 64 hex chars."""
        rec = _make_valid_interop_record()
        rec["determinism"]["artifact_hash"] = "short"
        with pytest.raises(ValueError, match="artifact_hash"):
            validate_interop_record(rec)

    def test_invalid_stable_sweep_hash_not_hex(self):
        """stable_sweep_hash must be hex characters only."""
        rec = _make_valid_interop_record()
        rec["determinism"]["stable_sweep_hash"] = "g" * 64
        with pytest.raises(ValueError, match="stable_sweep_hash"):
            validate_interop_record(rec)

    def test_invalid_artifact_hash_not_string(self):
        """artifact_hash must be a string, not None or int."""
        rec = _make_valid_interop_record()
        rec["determinism"]["artifact_hash"] = None
        with pytest.raises(ValueError, match="artifact_hash"):
            validate_interop_record(rec)

    def test_canonical_json_sort_keys_false_fails(self):
        """canonical_json.sort_keys must be true."""
        rec = _make_valid_interop_record()
        rec["determinism"]["canonical_json"]["sort_keys"] = False
        with pytest.raises(ValueError, match="sort_keys must be true"):
            validate_interop_record(rec)

    def test_canonical_json_wrong_separators_fails(self):
        """canonical_json.separators must be [',', ':']."""
        rec = _make_valid_interop_record()
        rec["determinism"]["canonical_json"]["separators"] = [", ", ": "]
        with pytest.raises(ValueError, match="separators"):
            validate_interop_record(rec)


class TestMeanItersRequirement:
    """Test that mean_iters is required for direct_comparison only."""

    def test_direct_comparison_without_mean_iters_fails(self):
        """direct_comparison records must have mean_iters."""
        rec = _make_valid_interop_record()
        assert rec["benchmark_kind"] == "direct_comparison"
        del rec["results"]["mean_iters"]
        with pytest.raises(ValueError, match="mean_iters"):
            validate_interop_record(rec)

    def test_reference_baseline_without_mean_iters_passes(self):
        """reference_baseline records do not require mean_iters."""
        rec = _make_valid_interop_record()
        rec["benchmark_kind"] = "reference_baseline"
        del rec["results"]["mean_iters"]
        validate_interop_record(rec)  # Must not raise

    def test_direct_comparison_with_mean_iters_passes(self):
        """direct_comparison with mean_iters passes."""
        rec = _make_valid_interop_record()
        assert "mean_iters" in rec["results"]
        validate_interop_record(rec)  # Must not raise
