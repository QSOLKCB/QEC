"""Tests for ternary utility helpers and early-exit convergence."""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from tests.testing_utils import (
    _cache,
    to_ternary,
    assert_no_ternary_errors,
    assert_strict_ternary_success,
    deterministic_array_cache,
    minimal_parity_matrix_3x5,
    minimal_parity_matrix_4x6,
)
from qec.decoder.ternary.ternary_coevolution import (
    early_exit_convergence,
    detect_state_cycle,
    should_terminate,
)


@pytest.fixture(autouse=True)
def clear_cache():
    _cache.clear()
    yield
    _cache.clear()


# --- Ternary mapping tests ---

def test_ternary_mapping():
    arr = np.array([1.0, 0.0, -1.0])
    tern = to_ternary(arr)
    assert np.array_equal(tern, np.array([1, 0, -1], dtype=np.int8))


def test_ternary_dtype():
    arr = np.array([2.5, -0.1, 0.0])
    tern = to_ternary(arr)
    assert tern.dtype == np.int8
    assert np.array_equal(tern, np.array([1, -1, 0], dtype=np.int8))


# --- Ternary assertion tests ---

def test_no_ternary_errors_pass():
    arr = np.array([1.0, 0.0])
    assert_no_ternary_errors(arr)


def test_no_ternary_errors_fail():
    arr = np.array([-1.0])
    try:
        assert_no_ternary_errors(arr)
    except AssertionError:
        return
    assert False, "Expected failure"


def test_strict_ternary_success_pass():
    arr = np.array([1.0, 2.0, 0.5])
    assert_strict_ternary_success(arr)


def test_strict_ternary_success_fail():
    arr = np.array([1.0, 0.0])
    try:
        assert_strict_ternary_success(arr)
    except AssertionError:
        return
    assert False, "Expected failure"


# --- Early exit convergence tests ---

def test_early_exit_converged():
    x = np.ones(5)
    history = [x, x.copy(), x.copy(), x.copy()]
    assert early_exit_convergence(history)


def test_early_exit_not_converged():
    history = [
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
        np.array([1.0, 1.0, 1.0]),
    ]
    assert not early_exit_convergence(history)


def test_early_exit_insufficient_history():
    x = np.ones(3)
    assert not early_exit_convergence([x, x.copy()])


# --- Deterministic cache tests ---

def test_deterministic_cache_returns_same_object():
    a = deterministic_array_cache("test_key_1", lambda: np.array([1.0, 2.0]))
    b = deterministic_array_cache("test_key_1", lambda: np.array([9.0, 9.0]))
    assert a is b
    assert np.array_equal(a, np.array([1.0, 2.0]))


def test_deterministic_cache_readonly():
    arr = deterministic_array_cache("test_key_2", lambda: np.array([3.0]))
    assert not arr.flags.writeable


# --- Minimal matrix tests ---

def test_minimal_3x5_shape():
    H = minimal_parity_matrix_3x5()
    assert H.shape == (3, 5)
    assert H.dtype == np.float64


def test_minimal_4x6_shape():
    H = minimal_parity_matrix_4x6()
    assert H.shape == (4, 6)
    assert H.dtype == np.float64


# --- Bounded cache tests ---

def test_cache_bound():
    for i in range(40):
        deterministic_array_cache(f"bound_test_{i}", minimal_parity_matrix_3x5)
    assert len(_cache) <= 32


# --- Markovian cycle detection tests ---

def test_markov_cycle_detection():
    x = np.ones(5)
    y = np.zeros(5)

    hashes = []
    for state in [x, y, x, y, x]:
        h = hashlib.sha256(state.tobytes()).hexdigest()
        hashes.append(h)

    assert detect_state_cycle(hashes, hashes[-1])


def test_markov_no_cycle():
    states = [np.array([float(i)]) for i in range(10)]
    hashes = [hashlib.sha256(s.tobytes()).hexdigest() for s in states]
    current = hashlib.sha256(np.array([99.0]).tobytes()).hexdigest()
    assert not detect_state_cycle(hashes, current)


def test_markov_short_history_detects():
    """With relaxed window, short histories can still detect cycles."""
    h = hashlib.sha256(np.ones(3).tobytes()).hexdigest()
    assert detect_state_cycle([h, h], h)


def test_markov_empty_history():
    h = hashlib.sha256(np.ones(3).tobytes()).hexdigest()
    assert not detect_state_cycle([], h)


# --- Unified termination controller tests ---

def test_should_terminate_convergence():
    x = np.ones(5)
    history = [x, x.copy(), x.copy(), x.copy()]
    hashes = ["a", "b", "c", "d"]
    assert should_terminate(history, hashes, enable_markov=False)


def test_should_terminate_markov():
    x = np.ones(5)
    history = [x, x]
    hashes = ["a", "b", "a"]
    assert should_terminate(history, hashes, enable_convergence=False)


def test_should_terminate_curvature():
    x = np.ones(5)
    history = [x, x.copy(), x.copy()]
    hashes = ["a", "b", "c"]
    assert should_terminate(
        history,
        hashes,
        enable_convergence=False,
        enable_markov=False,
        enable_curvature=True,
    )


def test_curvature_guard_prevents_early_exit():
    """Curvature must not trigger on histories shorter than 3."""
    history = [np.ones(3), np.ones(3)]
    hashes = ["a", "b"]
    assert not should_terminate(
        history,
        hashes,
        enable_convergence=False,
        enable_markov=False,
        enable_curvature=True,
    )


def test_markov_no_slice_equivalence():
    """Zero-copy end_index produces the same result as old list slicing."""
    hashes = ["a", "b", "c", "b"]
    # old behavior: check last hash in hashes[-3:-1]
    expected = "b" in hashes[-3:-1]
    # new behavior via end_index
    actual = detect_state_cycle(
        hashes,
        "b",
        window=3,
        end_index=len(hashes) - 1,
    )
    assert expected == actual


def test_should_not_terminate():
    history = [
        np.array([1.0, 0.0]),
        np.array([0.5, 0.5]),
        np.array([0.2, 0.8]),
    ]
    hashes = ["a", "b", "c"]
    assert not should_terminate(history, hashes)
