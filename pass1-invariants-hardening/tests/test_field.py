"""Tests for GF(2^e) arithmetic."""

import pytest

from qldpc.field import GF2e


# ---------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------
@pytest.fixture(params=[1, 2, 3, 4, 5])
def field(request):
    """Parametrised fixture returning GF(2^e) for several e."""
    return GF2e(request.param)


# ---------------------------------------------------------------
# Inverse correctness: a * a^{-1} == 1 for all non-zero a
# ---------------------------------------------------------------
class TestInverse:
    def test_inverse_correctness(self, field: GF2e):
        for a in field.nonzero_elements():
            assert field.mul(a, field.inv(a)) == 1, (
                f"GF(2^{field.e}): {a} * inv({a}) != 1"
            )

    def test_zero_inverse_raises(self, field: GF2e):
        with pytest.raises(ValueError, match="Zero"):
            field.inv(0)


# ---------------------------------------------------------------
# Distributivity: a * (b + c) == a*b + a*c
# ---------------------------------------------------------------
class TestDistributivity:
    def test_distributivity(self, field: GF2e):
        elems = list(field.elements())
        for a in elems:
            for b in elems:
                for c in elems:
                    lhs = field.mul(a, field.add(b, c))
                    rhs = field.add(field.mul(a, b), field.mul(a, c))
                    assert lhs == rhs, (
                        f"GF(2^{field.e}): distributivity failed for "
                        f"a={a}, b={b}, c={c}"
                    )


# ---------------------------------------------------------------
# Associativity of multiplication: (a*b)*c == a*(b*c)
# ---------------------------------------------------------------
class TestAssociativity:
    def test_associativity_mul(self, field: GF2e):
        elems = list(field.elements())
        for a in elems:
            for b in elems:
                for c in elems:
                    lhs = field.mul(field.mul(a, b), c)
                    rhs = field.mul(a, field.mul(b, c))
                    assert lhs == rhs, (
                        f"GF(2^{field.e}): associativity failed for "
                        f"a={a}, b={b}, c={c}"
                    )

    def test_associativity_add(self, field: GF2e):
        elems = list(field.elements())
        for a in elems:
            for b in elems:
                for c in elems:
                    lhs = field.add(field.add(a, b), c)
                    rhs = field.add(a, field.add(b, c))
                    assert lhs == rhs


# ---------------------------------------------------------------
# Commutativity
# ---------------------------------------------------------------
class TestCommutativity:
    def test_commutative_mul(self, field: GF2e):
        elems = list(field.elements())
        for a in elems:
            for b in elems:
                assert field.mul(a, b) == field.mul(b, a)

    def test_commutative_add(self, field: GF2e):
        elems = list(field.elements())
        for a in elems:
            for b in elems:
                assert field.add(a, b) == field.add(b, a)


# ---------------------------------------------------------------
# Identity elements
# ---------------------------------------------------------------
class TestIdentity:
    def test_additive_identity(self, field: GF2e):
        for a in field.elements():
            assert field.add(a, 0) == a

    def test_multiplicative_identity(self, field: GF2e):
        for a in field.elements():
            assert field.mul(a, 1) == a
