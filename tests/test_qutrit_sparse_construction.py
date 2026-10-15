"""Constructor conformance against the independent full-coordinate scalar path."""
from itertools import product
from types import MappingProxyType
import random

import numpy as np
import pytest

from qec.decoder.qutrit import (
    ExactDecoder, Pauli, QutritStabilizerCode, cyclic_five_qutrit_code,
    shor_nine_qutrit_code, ternary_golay_qutrit_code, paulis_of_weight,
)
from qec.decoder.qutrit.exact import _one_site_syndromes, _sparse_syndrome


class ScalarDecoder(ExactDecoder):
    def __post_init__(self):
        if self.max_weight < 0:
            raise ValueError("max_weight must be non-negative")
        identity = Pauli((0,) * self.code.n, (0,) * self.code.n)
        table = {self.code.syndrome(identity): identity}
        for weight in range(1, self.max_weight + 1):
            for candidate in paulis_of_weight(self.code.n, weight):
                syndrome = self.code.syndrome(candidate)
                previous = table.get(syndrome)
                if previous is None:
                    table[syndrome] = candidate
                elif not self.code.is_stabilizer(candidate.compose(previous.inverse())):
                    raise ValueError("bounded error set violates exact correction conditions")
        object.__setattr__(self, "_table", MappingProxyType(table))


@pytest.mark.parametrize("factory,weight,count", [
    (cyclic_five_qutrit_code, 1, 41), (shor_nine_qutrit_code, 1, 73),
    (ternary_golay_qutrit_code, 2, 3609),
])
def test_exhaustive_ordered_tables_and_public_corrections(factory, weight, count):
    code = factory()
    expected, actual = ScalarDecoder(code, weight), ExactDecoder(code, weight)
    assert list(actual._table.items()) == list(expected._table.items())
    errors = [Pauli((0,) * code.n, (0,) * code.n)]
    errors += [e for w in range(1, weight + 1) for e in paulis_of_weight(code.n, w)]
    assert len(errors) == count
    for error in errors:
        assert actual.correct(error) == expected.correct(error)
        assert actual.correct(error).success
    with pytest.raises(TypeError):
        actual._table[(0,) * len(code.stabilizers)] = errors[0]


@pytest.mark.parametrize("weight", [0, 1, 2, False, True, np.int64(1)])
def test_degenerate_zero_syndrome_and_weight_boundaries(weight):
    code = QutritStabilizerCode(((1, 0),))
    actual, expected = ExactDecoder(code, weight), ScalarDecoder(code, weight)
    assert list(actual._table.items()) == list(expected._table.items())
    assert actual._table[(0,)] == Pauli((0,), (0,))


@pytest.mark.parametrize("weight", [-1, 1.0, 2])
def test_constructor_failures_match_scalar(weight):
    code = cyclic_five_qutrit_code()
    errors = []
    for decoder in (ScalarDecoder, ExactDecoder):
        with pytest.raises((ValueError, TypeError)) as caught:
            decoder(code, weight)
        errors.append((type(caught.value), str(caught.value)))
    assert errors[0] == errors[1]


def test_signed_scaled_permuted_generators_with_same_name():
    original = shor_nine_qutrit_code()
    checks = tuple(tuple(np.int64(-2 * x + 6) for x in row)
                   for row in reversed(original.stabilizers))
    changed = QutritStabilizerCode(checks, name=original.name)
    for code in (original, changed):
        assert list(ExactDecoder(code)._table.items()) == list(ScalarDecoder(code)._table.items())


def test_held_out_random_commuting_check_shapes_and_dense_errors():
    rng = random.Random(20261015)
    for n in range(1, 8):
        # X-only triangular independent generators commute, with varied rank.
        for rank in range(1, n + 1):
            checks = tuple(tuple((int(i == j) if j <= i else rng.randrange(3))
                                 for j in range(n)) + (0,) * n for i in range(rank))
            code = QutritStabilizerCode(checks)
            plan = _one_site_syndromes(code)
            for _ in range(30):
                error = Pauli(tuple(rng.randrange(3) for _ in range(n)),
                              tuple(rng.randrange(3) for _ in range(n)))
                assert _sparse_syndrome(error, plan) == code.syndrome(error)


def test_all_small_paulis_match_scalar_sign_and_coordinate_order():
    code = QutritStabilizerCode(((1, 2, 0, 0), (0, 0, 1, 1)))
    plan = _one_site_syndromes(code)
    for coordinates in product(range(3), repeat=4):
        error = Pauli(coordinates[:2], coordinates[2:])
        assert _sparse_syndrome(error, plan) == code.syndrome(error)


def test_subclass_syndrome_override_retains_scalar_authority():
    class ReversedCode(QutritStabilizerCode):
        def syndrome(self, error):
            return tuple(reversed(super().syndrome(error)))
    code = ReversedCode(cyclic_five_qutrit_code().stabilizers)
    assert list(ExactDecoder(code)._table.items()) == list(ScalarDecoder(code)._table.items())


@pytest.mark.parametrize("scalar", [True, 0.5])
def test_invalid_gf3_scalars_still_rejected(scalar):
    with pytest.raises((TypeError, ValueError)):
        QutritStabilizerCode(((scalar, 0),))


def test_held_out_dense_graph_state_tables_and_logical_collisions():
    rng = random.Random(854390)
    cases = failures = 0
    for n in range(1, 5):
        for _ in range(8):
            adjacency = [[0] * n for _ in range(n)]
            for i in range(n):
                for j in range(i, n):
                    adjacency[i][j] = adjacency[j][i] = rng.randrange(3)
            for rank in range(1, n + 1):
                checks = tuple(tuple(int(i == j) for j in range(n)) + tuple(adjacency[i])
                               for i in range(rank))
                code = QutritStabilizerCode(checks, name="same")
                for weight in range(3):
                    outcomes = []
                    for decoder in (ScalarDecoder, ExactDecoder):
                        try:
                            outcomes.append(("ok", list(decoder(code, weight)._table.items())))
                        except ValueError as error:
                            outcomes.append((type(error).__name__, str(error)))
                    assert outcomes[0] == outcomes[1]
                    cases += 1
                    failures += outcomes[0][0] != "ok"
    assert cases == 240
    assert failures == 96
