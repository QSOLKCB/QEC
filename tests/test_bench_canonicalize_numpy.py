"""
Tests for canonicalize() — numpy type conversion and key ordering.
"""

import json

import numpy as np
import pytest

from bench.schema import canonicalize


class TestCanonicalizeNumpy:
    """canonicalize must convert all numpy types to plain Python."""

    def test_ndarray_to_list(self):
        result = canonicalize(np.array([1, 2, 3]))
        assert result == [1, 2, 3]
        assert isinstance(result, list)

    def test_nested_ndarray(self):
        result = canonicalize({"data": np.array([1.0, 2.0])})
        assert result == {"data": [1.0, 2.0]}
        assert isinstance(result["data"], list)

    def test_numpy_int(self):
        result = canonicalize(np.int64(42))
        assert result == 42
        assert isinstance(result, int)

    def test_numpy_float(self):
        result = canonicalize(np.float64(3.14))
        assert result == 3.14
        assert isinstance(result, float)

    def test_numpy_bool(self):
        result = canonicalize(np.bool_(True))
        assert result is True
        assert isinstance(result, bool)

    def test_tuple_to_list(self):
        result = canonicalize((1, 2, 3))
        assert result == [1, 2, 3]
        assert isinstance(result, list)

    def test_dict_keys_sorted(self):
        result = canonicalize({"c": 3, "a": 1, "b": 2})
        assert list(result.keys()) == ["a", "b", "c"]

    def test_nested_dict_keys_sorted(self):
        result = canonicalize({"z": {"b": 2, "a": 1}})
        assert list(result["z"].keys()) == ["a", "b"]

    def test_deep_copy(self):
        """Canonicalize must not mutate the original object."""
        original = {"key": np.array([1, 2, 3])}
        _ = canonicalize(original)
        assert isinstance(original["key"], np.ndarray)

    def test_json_serializable(self):
        """Canonicalized output must be JSON-serializable."""
        obj = {
            "arr": np.array([1.0, 2.0]),
            "int_val": np.int32(5),
            "float_val": np.float64(0.5),
            "bool_val": np.bool_(False),
            "nested": {"data": np.array([10, 20])},
        }
        result = canonicalize(obj)
        # Must not raise.
        text = json.dumps(result)
        assert isinstance(text, str)

    def test_plain_python_passthrough(self):
        """Plain Python types should pass through unchanged."""
        obj = {"a": 1, "b": "hello", "c": [1.0, 2.0], "d": True}
        result = canonicalize(obj)
        assert result == {"a": 1, "b": "hello", "c": [1.0, 2.0], "d": True}
