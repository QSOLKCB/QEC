from qec.decoder.ququart import ExactDecoder, packed_five_ququart_code, paulis_of_weight
from qec.decoder.ququart.cycle import run_harmonic_cycle
from qec.sonify.ququart_harmonics import encode_harmonics, read_harmonics


def test_exact_default_receiver_is_trusted():
    syndrome = (0, 1, 2, 3)
    result = read_harmonics(encode_harmonics(syndrome))
    assert result.syndrome == syndrome
    assert result.receiver_complete
    assert result.cross_harmonic_agreement
    assert result.parity_agreement
    assert result.trusted


def test_missing_receiver_role_fails_closed():
    syndrome = (0, 1, 2, 3)
    samples = encode_harmonics(syndrome)
    samples.pop(2)
    assert not read_harmonics(samples).trusted


def test_parity_corruption_fails_closed():
    syndrome = (0, 1, 2, 3)
    samples = encode_harmonics(syndrome)
    samples[2] = tuple(-value for value in samples[2])
    result = read_harmonics(samples)
    assert not result.parity_agreement
    assert not result.trusted


def test_cross_harmonic_disagreement_fails_closed():
    syndrome = (0, 1, 2, 3)
    samples = encode_harmonics(syndrome)
    samples[3] = encode_harmonics((1, 1, 2, 3), (3,))[3]
    result = read_harmonics(samples)
    assert not result.cross_harmonic_agreement
    assert not result.trusted


def test_every_single_error_survives_harmonic_cycle():
    code = packed_five_ququart_code()
    decoder = ExactDecoder(code)
    assert all(
        run_harmonic_cycle(decoder, error).success
        for error in paulis_of_weight(code.n, 1)
    )
