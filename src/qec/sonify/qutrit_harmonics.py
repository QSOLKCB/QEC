"""Harmonic observation of GF(3) syndromes, separate from correction."""

from __future__ import annotations

import cmath
import math
from dataclasses import dataclass
from typing import Mapping, Sequence

from qec.decoder.qutrit.gf3 import Vector, vector

DEFAULT_HARMONICS = (1, 2, 3)
OMEGA = cmath.exp(2j * math.pi / 3)


def phasor(symbol: int, harmonic: int) -> complex:
    """Encode one qutrit symbol at a positive harmonic order."""
    value = vector((symbol,), width=1)[0]
    if isinstance(harmonic, bool) or not isinstance(harmonic, int):
        raise TypeError("harmonic order must be an integer")
    if harmonic <= 0:
        raise ValueError("harmonic order must be positive")
    return OMEGA ** ((harmonic * value) % 3)


def encode_harmonics(
    syndrome: Sequence[int],
    harmonics: Sequence[int] = DEFAULT_HARMONICS,
) -> dict[int, tuple[complex, ...]]:
    """Encode a syndrome into deterministic complex spectral samples."""
    symbols = vector(syndrome)
    orders = tuple(harmonics)
    if len(set(orders)) != len(orders):
        raise ValueError("harmonic orders must be unique")
    return {
        order: tuple(phasor(symbol, order) for symbol in symbols)
        for order in orders
    }


@dataclass(frozen=True)
class HarmonicReadout:
    syndrome: Vector
    informative_harmonics: tuple[int, ...]
    dark_harmonics: tuple[int, ...]
    per_harmonic: tuple[tuple[int, Vector], ...]
    cross_harmonic_agreement: bool
    receiver_complete: bool
    ambiguous: bool
    residual: float
    distortion: float
    trusted: bool


def _nearest_symbol(sample: complex, harmonic: int) -> tuple[int, float, bool]:
    scores = tuple(abs(sample - phasor(symbol, harmonic)) for symbol in range(3))
    best = min(range(3), key=scores.__getitem__)
    ordered = sorted(scores)
    ambiguous = math.isclose(ordered[0], ordered[1], abs_tol=1e-12)
    return best, scores[best], ambiguous


def read_harmonics(
    samples: Mapping[int, Sequence[complex]],
    *,
    tolerance: float = 0.35,
) -> HarmonicReadout:
    """Recover GF(3) symbols and reject inconsistent/distorted observations."""
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")
    if not samples:
        raise ValueError("at least one harmonic is required")

    normalized = {
        order: tuple(complex(sample) for sample in values)
        for order, values in samples.items()
    }
    widths = {len(values) for values in normalized.values()}
    if len(widths) != 1 or not next(iter(widths)):
        raise ValueError("harmonic sample vectors need one shared nonzero width")

    informative = tuple(sorted(order for order in normalized if order % 3))
    dark = tuple(sorted(order for order in normalized if order % 3 == 0))
    if not informative:
        raise ValueError("harmonics divisible by 3 cannot identify GF(3) state")

    decoded: list[tuple[int, Vector]] = []
    residuals: list[float] = []
    ambiguous = False
    for order in informative:
        nearest = tuple(
            _nearest_symbol(sample, order)
            for sample in normalized[order]
        )
        decisions = tuple(result[0] for result in nearest)
        ambiguous = ambiguous or any(result[2] for result in nearest)
        decoded.append((order, decisions))

    width = next(iter(widths))
    combined: list[int] = []
    for index in range(width):
        scores = tuple(
            sum(
                abs(normalized[order][index] - phasor(symbol, order)) ** 2
                for order in informative
            )
            for symbol in range(3)
        )
        best = min(range(3), key=scores.__getitem__)
        ordered = sorted(scores)
        if math.isclose(ordered[0], ordered[1], abs_tol=1e-12):
            ambiguous = True
        combined.append(best)
        residuals.append(math.sqrt(scores[best] / len(informative)))

    agreement = all(
        decisions == decoded[0][1]
        for _, decisions in decoded[1:]
    )
    dark_residual = max(
        (
            abs(sample - 1)
            for order in dark
            for sample in normalized[order]
        ),
        default=0.0,
    )
    residual = max(residuals, default=0.0)
    distortion = max(residual, dark_residual)
    residues = {order % 3 for order in informative}
    receiver_complete = residues == {1, 2} and bool(dark)
    return HarmonicReadout(
        syndrome=tuple(combined),
        informative_harmonics=informative,
        dark_harmonics=dark,
        per_harmonic=tuple(decoded),
        cross_harmonic_agreement=agreement,
        receiver_complete=receiver_complete,
        ambiguous=ambiguous,
        residual=residual,
        distortion=distortion,
        trusted=(
            receiver_complete
            and agreement
            and not ambiguous
            and distortion <= tolerance
        ),
    )


def collective_modes(syndrome: Sequence[int]) -> tuple[complex, ...]:
    """Unitary DFT of the qutrit phasor field; total power is invariant."""
    symbols = vector(syndrome)
    if not symbols:
        return ()
    size = len(symbols)
    field = tuple(phasor(symbol, 1) for symbol in symbols)
    scale = math.sqrt(size)
    return tuple(
        sum(
            value * cmath.exp(-2j * math.pi * mode * site / size)
            for site, value in enumerate(field)
        )
        / scale
        for mode in range(size)
    )
