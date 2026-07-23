"""Tests for the qutrit harmonic observation boundary."""

from __future__ import annotations

import math

import pytest

from qec.sonify.etq303 import STATE_COUNT, etq_address, etq_index
from qec.sonify.qutrit_harmonics import (
    collective_modes,
    encode_harmonics,
    phasor,
    read_harmonics,
)


def test_informative_harmonics_round_trip():
    syndrome = (0, 1, 2, 2, 1, 0)
    readout = read_harmonics(encode_harmonics(syndrome))
    assert readout.syndrome == syndrome
    assert readout.informative_harmonics == (1, 2)
    assert readout.dark_harmonics == (3,)
    assert readout.cross_harmonic_agreement
    assert readout.receiver_complete
    assert readout.trusted


def test_third_harmonic_is_state_dark():
    assert phasor(0, 3) == pytest.approx(phasor(1, 3))
    assert phasor(1, 3) == pytest.approx(phasor(2, 3))
    with pytest.raises(ValueError, match="cannot identify"):
        read_harmonics(encode_harmonics((0, 1, 2), (3,)))


def test_cross_harmonic_disagreement_is_rejected():
    samples = encode_harmonics((1, 2))
    samples[2] = (phasor(2, 2), samples[2][1])
    readout = read_harmonics(samples, tolerance=2.0)
    assert not readout.cross_harmonic_agreement
    assert readout.ambiguous
    assert not readout.trusted


@pytest.mark.parametrize("orders", [(1, 3), (1, 2)])
def test_incomplete_receiver_never_becomes_trusted(orders):
    readout = read_harmonics(encode_harmonics((1, 2), orders))
    assert not readout.receiver_complete
    assert not readout.trusted


def test_exactly_ambiguous_sample_is_returned_as_untrusted():
    samples = encode_harmonics((0,))
    samples[1] = ((phasor(0, 1) + phasor(1, 1)) / 2,)
    readout = read_harmonics(samples)
    assert readout.ambiguous
    assert not readout.trusted


def test_dark_harmonic_detects_distortion_without_guessing_state():
    samples = encode_harmonics((2,))
    samples[3] = (0j,)
    readout = read_harmonics(samples, tolerance=0.35)
    assert readout.syndrome == (2,)
    assert readout.distortion == pytest.approx(1.0)
    assert not readout.trusted


def test_collective_mode_parseval_invariant():
    syndrome = (0, 1, 2, 1, 0, 2)
    modes = collective_modes(syndrome)
    assert sum(abs(mode) ** 2 for mode in modes) == pytest.approx(
        len(syndrome),
        abs=1e-12,
    )
    assert all(math.isfinite(abs(mode)) for mode in modes)


def test_etq303_address_is_a_bijection():
    addresses = {
        etq_index(*etq_address(index))
        for index in range(STATE_COUNT)
    }
    assert addresses == set(range(303))
