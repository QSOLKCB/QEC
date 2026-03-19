"""
Fuzz-style determinism tests for canonicalize().

Validates that the shared canonicalize() utility is:
  1. Idempotent.
  2. Deterministic for nested randomized structures.
  3. JSON-stable.
  4. Non-mutating on input.
  5. Stable across repeated runs.

Uses a seeded numpy RNG — no unseeded randomness is used anywhere.
"""

import copy
import json
import string

import numpy as np

from utils.canonicalize import canonicalize

# ---------------------------------------------------------------------------
# Fixed seed — all randomness derives from this single seeded generator.
# ---------------------------------------------------------------------------
_SEED = 123456789
_NUM_FUZZ_CASES = 50
_MAX_DEPTH = 3


def _random_nested_object(rng, depth=0):
    """Generate a random nested structure containing mixed Python/numpy types.

    Parameters
    ----------
    rng : numpy.random.Generator
        Seeded random generator.
    depth : int
        Current recursion depth (max ``_MAX_DEPTH``).

    Returns
    -------
    object
        A random nested structure.
    """
    if depth >= _MAX_DEPTH:
        # At max depth, return a random primitive.
        return _random_primitive(rng)

    # Choose a type category at random.
    kind = rng.integers(0, 6)  # 0-5

    if kind == 0:
        # dict with 1–4 string keys
        size = int(rng.integers(1, 5))
        return {
            _random_string(rng): _random_nested_object(rng, depth + 1)
            for _ in range(size)
        }
    elif kind == 1:
        # list with 1–4 elements
        size = int(rng.integers(1, 5))
        return [_random_nested_object(rng, depth + 1) for _ in range(size)]
    elif kind == 2:
        # tuple with 1–4 elements
        size = int(rng.integers(1, 5))
        return tuple(_random_nested_object(rng, depth + 1) for _ in range(size))
    elif kind == 3:
        # numpy scalar
        scalar_kind = rng.integers(0, 2)
        if scalar_kind == 0:
            return np.int64(int(rng.integers(-100, 101)))
        else:
            return np.float64(rng.standard_normal())
    elif kind == 4:
        # numpy array (1–5 elements)
        size = int(rng.integers(1, 6))
        arr_kind = rng.integers(0, 2)
        if arr_kind == 0:
            return rng.integers(-100, 101, size=size).astype(np.int64)
        else:
            return rng.standard_normal(size).astype(np.float64)
    else:
        # primitive
        return _random_primitive(rng)


def _random_primitive(rng):
    """Return a random JSON-compatible primitive value."""
    kind = rng.integers(0, 5)
    if kind == 0:
        return int(rng.integers(-1000, 1001))
    elif kind == 1:
        return float(rng.standard_normal())
    elif kind == 2:
        return _random_string(rng)
    elif kind == 3:
        return bool(rng.integers(0, 2))
    else:
        return None


def _random_string(rng, max_len=8):
    """Return a random lowercase ASCII string of length 1–max_len."""
    length = int(rng.integers(1, max_len + 1))
    chars = list(string.ascii_lowercase)
    indices = rng.integers(0, len(chars), size=length)
    return "".join(chars[i] for i in indices)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _generate_objects():
    """Generate a fixed list of ``_NUM_FUZZ_CASES`` random nested objects."""
    rng = np.random.default_rng(_SEED)
    return [_random_nested_object(rng) for _ in range(_NUM_FUZZ_CASES)]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCanonicalizeIdempotence:
    """canonicalize(canonicalize(x)) must equal canonicalize(x)."""

    def test_idempotence(self):
        objects = _generate_objects()
        for obj in objects:
            canon1 = canonicalize(obj)
            canon2 = canonicalize(canon1)
            assert canon1 == canon2


class TestCanonicalizeJsonRoundtripStability:
    """JSON serialization of canonicalized output must be stable."""

    def test_json_roundtrip_stability(self):
        objects = _generate_objects()
        for obj in objects:
            canon = canonicalize(obj)
            j1 = json.dumps(canon, sort_keys=True, separators=(",", ":"))
            j2 = json.dumps(
                canonicalize(obj), sort_keys=True, separators=(",", ":")
            )
            assert j1 == j2


class TestCanonicalizeNoInputMutation:
    """canonicalize() must never mutate the input object."""

    def test_no_input_mutation(self):
        objects = _generate_objects()
        for obj in objects:
            original = copy.deepcopy(obj)
            _ = canonicalize(obj)
            # For numpy arrays, use array_equal; for others, use ==.
            _assert_deep_equal(obj, original)


class TestCanonicalizeRepeatability:
    """Repeated canonicalization of the same input must produce identical output."""

    def test_repeatability_across_runs(self):
        objects = _generate_objects()
        first_pass = [canonicalize(obj) for obj in objects]

        # Second full pass.
        second_pass = [canonicalize(obj) for obj in objects]

        for a, b in zip(first_pass, second_pass):
            assert a == b


# ---------------------------------------------------------------------------
# Deep-equality helper (handles numpy arrays inside nested structures)
# ---------------------------------------------------------------------------

def _assert_deep_equal(a, b):
    """Assert structural equality, handling numpy arrays correctly."""
    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        np.testing.assert_array_equal(a, b)
    elif isinstance(a, dict) and isinstance(b, dict):
        assert set(a.keys()) == set(b.keys())
        for k in a:
            _assert_deep_equal(a[k], b[k])
    elif isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            _assert_deep_equal(x, y)
    else:
        assert a == b
