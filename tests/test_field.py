# tests/test_field.py
import numpy as np
import pytest
from qldpc.field import GF2e
# ---------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------
def exhaustive_elements(gf):
    return list(range(gf.q))
def random_elements(gf, n=20, seed=123):
    rng = np.random.default_rng(seed)
    return list(rng.integers(0, gf.q, size=n))
# ---------------------------------------------------------
# Basic structural tests
# ---------------------------------------------------------
def test_inverse_correct_small_fields():
    """
    Exhaustively verify multiplicative inverses for small fields.
    """
    for e in [2, 3]:
        gf = GF2e(e)
        for a in range(1, gf.q):
            inv = gf.inv(a)
            assert gf.mul(a, inv) == 1
            assert gf.mul(inv, a) == 1
def test_zero_and_one_properties():
    gf = GF2e(3)
    for a in exhaustive_elements(gf):
        assert gf.add(a, 0) == a
        assert gf.mul(a, 0) == 0
        assert gf.mul(a, 1) == a
# ---------------------------------------------------------
# Field axioms
# ---------------------------------------------------------
def test_add_commutativity():
    gf = GF2e(3)
    for a in exhaustive_elements(gf):
        for b in exhaustive_elements(gf):
            assert gf.add(a, b) == gf.add(b, a)
def test_mul_commutativity():
    gf = GF2e(3)
    for a in exhaustive_elements(gf):
        for b in exhaustive_elements(gf):
            assert gf.mul(a, b) == gf.mul(b, a)
def test_distributivity():
    gf = GF2e(3)
    for a in exhaustive_elements(gf):
        for b in exhaustive_elements(gf):
            for c in exhaustive_elements(gf):
                left = gf.mul(a, gf.add(b, c))
                right = gf.add(gf.mul(a, b), gf.mul(a, c))
                assert left == right
def test_associativity_multiplication():
    gf = GF2e(3)
    for a in exhaustive_elements(gf):
        for b in exhaustive_elements(gf):
            for c in exhaustive_elements(gf):
                left = gf.mul(gf.mul(a, b), c)
                right = gf.mul(a, gf.mul(b, c))
                assert left == right
# ---------------------------------------------------------
# Companion matrix correctness
# ---------------------------------------------------------
def test_companion_matrix_identity_and_zero():
    gf = GF2e(3)
    C0 = gf.companion_matrix(0)
    C1 = gf.companion_matrix(1)
    assert np.all(C0 == 0)
    assert np.all(C1 == np.eye(gf.e, dtype=np.uint8))
def test_companion_matrix_multiplicativity_small():
    """
    Exhaustively verify:
        C(a*b) == C(a) @ C(b) mod 2
    for small fields.
    """
    for e in [2, 3]:
        gf = GF2e(e)
        for a in exhaustive_elements(gf):
            for b in exhaustive_elements(gf):
                C_ab = gf.companion_matrix(gf.mul(a, b))
                C_a = gf.companion_matrix(a)
                C_b = gf.companion_matrix(b)
                product = (C_a @ C_b) % 2
                assert np.array_equal(C_ab, product)
def test_companion_matrix_random_large():
    """
    Random sampling for e=4 to avoid heavy exhaustive tests.
    """
    gf = GF2e(4)
    elems = random_elements(gf, n=30)
    for a in elems:
        for b in elems:
            C_ab = gf.companion_matrix(gf.mul(a, b))
            C_a = gf.companion_matrix(a)
            C_b = gf.companion_matrix(b)
            product = (C_a @ C_b) % 2
            assert np.array_equal(C_ab, product)
# ---------------------------------------------------------
# Determinism test
# ---------------------------------------------------------
def test_deterministic_tables():
    gf1 = GF2e(3)
    gf2 = GF2e(3)
    assert np.array_equal(gf1.mul_table, gf2.mul_table)
    assert np.array_equal(gf1.inv_table, gf2.inv_table)
