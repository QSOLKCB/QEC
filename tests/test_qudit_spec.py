"""
Tests for QuditSpec — dimension specification layer (v3.0.1).
"""

import json

import numpy as np
import pytest

from qudit.spec import QuditSpec


class TestQuditSpecDefaults:
    """Default spec is qubit (dimension=2)."""

    def test_default_dimension(self):
        spec = QuditSpec()
        assert spec.dimension == 2

    def test_default_encoding(self):
        spec = QuditSpec()
        assert spec.encoding == "qubit"

    def test_default_metadata(self):
        spec = QuditSpec()
        assert spec.metadata == {}

    def test_is_qubit(self):
        assert QuditSpec().is_qubit is True
        assert QuditSpec(dimension=3).is_qubit is False

    def test_qubit_factory(self):
        spec = QuditSpec.qubit()
        assert spec.dimension == 2
        assert spec.encoding == "qubit"


class TestQuditSpecValidation:
    """Validation rules for dimension, encoding, metadata."""

    def test_dimension_must_be_int_ge_2(self):
        QuditSpec(dimension=2)   # ok
        QuditSpec(dimension=7)   # ok
        with pytest.raises(ValueError, match="dimension"):
            QuditSpec(dimension=1)
        with pytest.raises(ValueError, match="dimension"):
            QuditSpec(dimension=0)
        with pytest.raises(ValueError, match="dimension"):
            QuditSpec(dimension=-3)

    def test_dimension_rejects_float(self):
        with pytest.raises(ValueError, match="dimension"):
            QuditSpec(dimension=2.5)

    def test_encoding_must_be_nonempty_str(self):
        QuditSpec(encoding="ququart")  # ok
        with pytest.raises(ValueError, match="encoding"):
            QuditSpec(encoding="")
        with pytest.raises(ValueError, match="encoding"):
            QuditSpec(encoding=42)

    def test_metadata_non_serializable_raises(self):
        with pytest.raises(ValueError, match="metadata"):
            QuditSpec(metadata={"fn": lambda x: x})

    def test_frozen(self):
        spec = QuditSpec()
        with pytest.raises(AttributeError):
            spec.dimension = 3


class TestQuditSpecSerialization:
    """JSON-safe, deterministic serialization."""

    def test_to_dict_sorted_keys(self):
        d = QuditSpec(dimension=3, encoding="qutrit").to_dict()
        assert list(d.keys()) == ["dimension", "encoding", "metadata"]

    def test_to_json_deterministic(self):
        s1 = QuditSpec(dimension=5, encoding="ququint", metadata={"a": 1}).to_json()
        s2 = QuditSpec(dimension=5, encoding="ququint", metadata={"a": 1}).to_json()
        assert s1 == s2

    def test_to_json_compact(self):
        s = QuditSpec().to_json()
        # Compact separators: no spaces after : or ,
        assert ": " not in s
        assert ", " not in s

    def test_round_trip(self):
        original = QuditSpec(dimension=4, encoding="ququart", metadata={"k": "v"})
        d = original.to_dict()
        restored = QuditSpec.from_dict(d)
        assert restored == original

    def test_from_dict_defaults(self):
        spec = QuditSpec.from_dict({})
        assert spec == QuditSpec()

    def test_json_round_trip(self):
        original = QuditSpec(dimension=3, encoding="qutrit")
        text = original.to_json()
        parsed = json.loads(text)
        restored = QuditSpec.from_dict(parsed)
        assert restored == original


class TestQuditSpecNumpySafety:
    """Numpy types in metadata are canonicalized."""

    def test_numpy_int_in_metadata(self):
        spec = QuditSpec(metadata={"n": np.int64(42)})
        assert isinstance(spec.metadata["n"], int)
        assert spec.metadata["n"] == 42

    def test_numpy_float_in_metadata(self):
        spec = QuditSpec(metadata={"x": np.float64(3.14)})
        assert isinstance(spec.metadata["x"], float)

    def test_numpy_array_in_metadata(self):
        spec = QuditSpec(metadata={"arr": np.array([1, 2, 3])})
        assert spec.metadata["arr"] == [1, 2, 3]
        assert isinstance(spec.metadata["arr"], list)

    def test_metadata_keys_sorted(self):
        spec = QuditSpec(metadata={"z": 1, "a": 2, "m": 3})
        assert list(spec.metadata.keys()) == ["a", "m", "z"]
