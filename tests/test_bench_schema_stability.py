"""
Tests for benchmark schema stability.

Verifies that:
- dumps_result() produces stable, deterministic formatting.
- validate_result() catches missing/invalid fields.
- Key ordering is stable across calls.
"""

import json

import pytest

from bench.schema import (
    SCHEMA_VERSION,
    canonicalize,
    dumps_result,
    validate_result,
)


def _make_valid_result(**overrides):
    """Build a minimal valid result object."""
    obj = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": "2026-01-01T00:00:00+00:00",
        "environment": {"python_version": "3.11.0", "platform": "Linux"},
        "config": {"seed": 42},
        "results": [
            {
                "decoder": "bp_test",
                "distance": 3,
                "p": 0.01,
                "fer": 0.5,
                "wer": 0.5,
                "mean_iters": 10.0,
            }
        ],
        "summaries": {"thresholds": [], "runtime_scaling": []},
    }
    obj.update(overrides)
    return obj


class TestDumpsStability:
    """dumps_result must produce identical bytes for the same input."""

    def test_same_input_same_output(self):
        obj = _make_valid_result()
        a = dumps_result(obj)
        b = dumps_result(obj)
        assert a == b

    def test_sort_keys(self):
        obj = _make_valid_result()
        text = dumps_result(obj)
        parsed = json.loads(text)
        assert list(parsed.keys()) == sorted(parsed.keys())

    def test_compact_separators(self):
        obj = _make_valid_result()
        text = dumps_result(obj)
        # Compact separators: no space after "," or ":"
        assert ", " not in text or text.count(", ") == 0
        # Actually check by re-serializing with compact separators
        expected = json.dumps(json.loads(text), sort_keys=True, separators=(",", ":"))
        assert text == expected

    def test_deterministic_across_key_insertion_order(self):
        """Key ordering must be stable regardless of dict insertion order."""
        obj1 = {"b": 2, "a": 1, "c": 3}
        obj2 = {"a": 1, "c": 3, "b": 2}
        assert dumps_result(obj1) == dumps_result(obj2)


class TestValidateResult:
    """validate_result must reject malformed objects."""

    def test_valid_passes(self):
        obj = _make_valid_result()
        validate_result(obj)  # Should not raise.

    def test_missing_top_key(self):
        obj = _make_valid_result()
        del obj["schema_version"]
        with pytest.raises(ValueError, match="Missing required top-level key"):
            validate_result(obj)

    def test_wrong_schema_version(self):
        obj = _make_valid_result(schema_version="1.0.0")
        with pytest.raises(ValueError, match="Unsupported schema_version"):
            validate_result(obj)

    def test_wrong_type(self):
        obj = _make_valid_result(results="not a list")
        with pytest.raises(ValueError, match="must be"):
            validate_result(obj)

    def test_missing_record_key(self):
        obj = _make_valid_result()
        del obj["results"][0]["fer"]
        with pytest.raises(ValueError, match="missing required keys"):
            validate_result(obj)

    def test_not_a_dict(self):
        with pytest.raises(ValueError, match="must be a dict"):
            validate_result("string")

    def test_results_entry_not_dict(self):
        obj = _make_valid_result(results=["not a dict"])
        with pytest.raises(ValueError, match="must be a dict"):
            validate_result(obj)
