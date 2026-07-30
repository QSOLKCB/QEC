"""Fail-closed harmonic observation of four-state syndrome symbols."""

from __future__ import annotations

import cmath
import math
from dataclasses import dataclass
from typing import Mapping, Sequence

from qec.decoder.ququart.syndrome import QuquartSyndrome, symbol, symbols

DEFAULT_HARMONICS = (1, 3, 2, 4)
I = 1j


def phasor(value: int, harmonic: int) -> complex:
    """Encode one four-state symbol at a positive harmonic order."""
    state = symbol(value)
    if isinstance(harmonic, bool) or not isinstance(harmonic, int):
        raise TypeError("harmonic order must be an integer")
    if harmonic <= 0:
        raise ValueError("harmonic order must be positive")
    return I ** ((harmonic * state) % 4)


def encode_harmonics(
    syndrome: Sequence[int],
    harmonics: Sequence[int] = DEFAULT_HARMONICS,
) -> dict[int, tuple[complex, ...]]:
    """Encode a four-state syndrome into deterministic spectral samples."""
    states = symbols(syndrome)
    orders = tuple(harmonics)
    if not states:
        raise ValueError("ququart syndrome must be non-empty")
    if len(set(orders)) != len(orders):
        raise ValueError("harmonic orders must be unique")
    return {
        order: tuple(phasor(state, order) for state in states)
        for order in orders
    }


@dataclass(frozen=True)
class HarmonicReadout:
    syndrome: QuquartSyndrome
    identifying_harmonics: tuple[int, ...]
    parity_harmonics: tuple[int, ...]
    dark_harmonics: tuple[int, ...]
    per_harmonic: tuple[tuple[int, QuquartSyndrome], ...]
    cross_harmonic_agreement: bool
    parity_agreement: bool
    receiver_complete: bool
    ambiguous: bool
    residual: float
    distortion: float
    trusted: bool


def _nearest_symbol(sample: complex, harmonic: int) -> tuple[int, float, bool]:
    scores = tuple(abs(sample - phasor(state, harmonic)) for state in range(4))
    best = min(range(4), key=scores.__getitem__)
    ordered = sorted(scores)
    ambiguous = math.isclose(ordered[0], ordered[1], abs_tol=1e-12)
    return best, scores[best], ambiguous


def read_harmonics(
    samples: Mapping[int, Sequence[complex]],
    *,
    tolerance: float = 0.35,
) -> HarmonicReadout:
    """Recover four-state symbols and reject inconsistent observations.

    Orders congruent to 1 and 3 modulo 4 identify all four states. Order 2
    observes only parity. Multiples of 4 are state-dark distortion references.
    The default receiver requires all three roles.
    """
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")
    if not samples:
        raise ValueError("at least one harmonic is required")

    normalized = {
        order: tuple(complex(sample) for sample in values)
        for order, values in samples.items()
    }
    if any(
        isinstance(order, bool) or not isinstance(order, int) or order <= 0
        for order in normalized
    ):
        raise ValueError("harmonic orders must be positive integers")
    widths = {len(values) for values in normalized.values()}
    if len(widths) != 1 or not next(iter(widths)):
        raise ValueError("harmonic sample vectors need one shared nonzero width")

    identifying = tuple(sorted(order for order in normalized if order % 4 in (1, 3)))
    parity = tuple(sorted(order for order in normalized if order % 4 == 2))
    dark = tuple(sorted(order for order in normalized if order % 4 == 0))
    if not identifying:
        raise ValueError("at least one odd harmonic is required to identify four states")

    decoded: list[tuple[int, QuquartSyndrome]] = []
    ambiguous = False
    for order in identifying:
        nearest = tuple(_nearest_symbol(sample, order) for sample in normalized[order])
        decisions = tuple(result[0] for result in nearest)
        ambiguous = ambiguous or any(result[2] for result in nearest)
        decoded.append((order, decisions))

    width = next(iter(widths))
    combined: list[int] = []
    residuals: list[float] = []
    for index in range(width):
        scores = tuple(
            sum(
                abs(normalized[order][index] - phasor(state, order)) ** 2
                for order in identifying
            )
            for state in range(4)
        )
        best = min(range(4), key=scores.__getitem__)
        ordered = sorted(scores)
        ambiguous = ambiguous or math.isclose(ordered[0], ordered[1], abs_tol=1e-12)
        combined.append(best)
        residuals.append(math.sqrt(scores[best] / len(identifying)))

    syndrome = tuple(combined)
    agreement = all(
        decisions == decoded[0][1]
        for _, decisions in decoded[1:]
    )
    parity_residual = max(
        (
            abs(sample - phasor(syndrome[index], order))
            for order in parity
            for index, sample in enumerate(normalized[order])
        ),
        default=0.0,
    )
    parity_agreement = parity_residual <= tolerance
    dark_residual = max(
        (
            abs(sample - 1)
            for order in dark
            for sample in normalized[order]
        ),
        default=0.0,
    )
    residual = max(residuals, default=0.0)
    distortion = max(residual, parity_residual, dark_residual)
    residues = {order % 4 for order in identifying}
    receiver_complete = residues == {1, 3} and bool(parity) and bool(dark)
    trusted = (
        receiver_complete
        and agreement
        and parity_agreement
        and not ambiguous
        and distortion <= tolerance
    )
    return HarmonicReadout(
        syndrome=syndrome,
        identifying_harmonics=identifying,
        parity_harmonics=parity,
        dark_harmonics=dark,
        per_harmonic=tuple(decoded),
        cross_harmonic_agreement=agreement,
        parity_agreement=parity_agreement,
        receiver_complete=receiver_complete,
        ambiguous=ambiguous,
        residual=residual,
        distortion=distortion,
        trusted=trusted,
    )


def collective_modes(syndrome: Sequence[int]) -> tuple[complex, ...]:
    """Unitary DFT of the H1 ququart phasor field."""
    states = symbols(syndrome)
    if not states:
        return ()
    size = len(states)
    field = tuple(phasor(state, 1) for state in states)
    scale = math.sqrt(size)
    return tuple(
        sum(
            value * cmath.exp(-2j * math.pi * mode * site / size)
            for site, value in enumerate(field)
        )
        / scale
        for mode in range(size)
    )
