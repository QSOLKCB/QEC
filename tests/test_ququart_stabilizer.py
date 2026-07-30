from itertools import product

import pytest

from qec.decoder.ququart import (
    ExactDecoder,
    UncorrectableSyndrome,
    exact_distance,
    pack_binary_syndrome,
    packed_five_ququart_code,
    paulis_of_weight,
    unpack_ququart_syndrome,
)


def test_code_parameters_and_distance():
    code = packed_five_ququart_code()
    assert (code.n, code.k, code.distance_hint) == (5, 1, 3)
    assert code.syndrome_width == 8
    assert exact_distance(code, max_weight=3) == 3


def test_every_single_ququart_pauli_is_corrected():
    code = packed_five_ququart_code()
    decoder = ExactDecoder(code)
    errors = tuple(paulis_of_weight(code.n, 1))
    assert len(errors) == 5 * 15
    assert decoder.table_size == 76
    assert len({code.syndrome(error) for error in errors}) == 75
    assert all(decoder.correct(error).success for error in errors)


def test_unknown_syndrome_fails_closed():
    code = packed_five_ququart_code()
    decoder = ExactDecoder(code)
    known = {code.syndrome(error) for error in paulis_of_weight(code.n, 1)}
    unknown = next(
        syndrome
        for syndrome in product((0, 1), repeat=code.syndrome_width)
        if syndrome not in known and any(syndrome)
    )
    with pytest.raises(UncorrectableSyndrome):
        decoder.decode(unknown)


def test_syndrome_pack_round_trip():
    binary = (0, 1, 1, 0, 1, 0, 1, 0)
    packed = pack_binary_syndrome(binary)
    assert packed == (1, 2, 3, 0)
    assert unpack_ququart_syndrome(packed) == binary
