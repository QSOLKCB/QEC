"""One fail-closed packed-ququart QEC cycle through a harmonic channel."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from qec.sonify.ququart_harmonics import (
    DEFAULT_HARMONICS,
    HarmonicReadout,
    encode_harmonics,
    read_harmonics,
)

from .exact import ExactDecoder, UncorrectableSyndrome
from .packed import BinarySyndrome, PackedPauli
from .syndrome import QuquartSyndrome, pack_binary_syndrome, unpack_ququart_syndrome


@dataclass(frozen=True)
class HarmonicCycleResult:
    exact_binary_syndrome: BinarySyndrome
    exact_ququart_syndrome: QuquartSyndrome
    observation: HarmonicReadout
    accepted: bool
    correction: PackedPauli | None
    residual: PackedPauli
    success: bool


def run_harmonic_cycle(
    decoder: ExactDecoder,
    error: PackedPauli,
    *,
    samples: Mapping[int, Sequence[complex]] | None = None,
    harmonics: Sequence[int] = DEFAULT_HARMONICS,
    tolerance: float = 0.35,
) -> HarmonicCycleResult:
    """Observe, validate, unpack, and correct; never act on untrusted readout."""
    binary_syndrome = decoder.code.syndrome(error)
    ququart_syndrome = pack_binary_syndrome(binary_syndrome)
    observed_samples = (
        encode_harmonics(ququart_syndrome, harmonics)
        if samples is None
        else samples
    )
    observation = read_harmonics(observed_samples, tolerance=tolerance)
    if not observation.trusted:
        return HarmonicCycleResult(
            exact_binary_syndrome=binary_syndrome,
            exact_ququart_syndrome=ququart_syndrome,
            observation=observation,
            accepted=False,
            correction=None,
            residual=error,
            success=False,
        )

    observed_binary = unpack_ququart_syndrome(observation.syndrome)
    try:
        correction = decoder.decode(observed_binary)
    except UncorrectableSyndrome:
        return HarmonicCycleResult(
            exact_binary_syndrome=binary_syndrome,
            exact_ququart_syndrome=ququart_syndrome,
            observation=observation,
            accepted=False,
            correction=None,
            residual=error,
            success=False,
        )

    residual = error.compose(correction)
    return HarmonicCycleResult(
        exact_binary_syndrome=binary_syndrome,
        exact_ququart_syndrome=ququart_syndrome,
        observation=observation,
        accepted=True,
        correction=correction,
        residual=residual,
        success=decoder.code.is_stabilizer(residual),
    )
