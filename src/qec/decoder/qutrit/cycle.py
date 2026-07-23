"""One fail-closed qutrit QEC cycle through a harmonic syndrome channel."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from qec.sonify.qutrit_harmonics import (
    DEFAULT_HARMONICS,
    HarmonicReadout,
    encode_harmonics,
    read_harmonics,
)

from .exact import ExactDecoder, UncorrectableSyndrome
from .stabilizer import (
    Pauli,
    Syndrome,
)


@dataclass(frozen=True)
class HarmonicCycleResult:
    exact_syndrome: Syndrome
    observation: HarmonicReadout
    accepted: bool
    correction: Pauli | None
    residual: Pauli
    success: bool


def run_harmonic_cycle(
    decoder: ExactDecoder,
    error: Pauli,
    *,
    samples: Mapping[int, Sequence[complex]] | None = None,
    harmonics: Sequence[int] = DEFAULT_HARMONICS,
    tolerance: float = 0.35,
) -> HarmonicCycleResult:
    """Observe, validate, and correct; never act on an untrusted readout."""
    syndrome = decoder.code.syndrome(error)
    observed_samples = (
        encode_harmonics(syndrome, harmonics)
        if samples is None
        else samples
    )
    observation = read_harmonics(observed_samples, tolerance=tolerance)
    identity = Pauli((0,) * decoder.code.n, (0,) * decoder.code.n)
    if not observation.trusted:
        return HarmonicCycleResult(
            exact_syndrome=syndrome,
            observation=observation,
            accepted=False,
            correction=None,
            residual=error,
            success=False,
        )

    try:
        correction = decoder.decode(observation.syndrome)
    except UncorrectableSyndrome:
        return HarmonicCycleResult(
            exact_syndrome=syndrome,
            observation=observation,
            accepted=False,
            correction=None,
            residual=error,
            success=False,
        )

    residual = error.compose(correction)
    return HarmonicCycleResult(
        exact_syndrome=syndrome,
        observation=observation,
        accepted=True,
        correction=correction,
        residual=residual,
        success=decoder.code.is_stabilizer(residual),
    )
