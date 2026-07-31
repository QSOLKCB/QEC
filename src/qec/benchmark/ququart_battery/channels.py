"""Deterministic iid physical-noise batteries for the packed ququart decoder."""

from __future__ import annotations

import hashlib
import math
import random
from decimal import Decimal
from typing import Iterable, Sequence

from qec.decoder.ququart.codes import packed_five_ququart_code
from qec.decoder.ququart.exact import ExactDecoder, UncorrectableSyndrome
from qec.decoder.ququart.packed import PackedPauli, identity, local_pauli

from .oracle import DEFAULT_ERROR_RATES

FULL_PACKED_DEPOLARIZING = tuple(
    (lane0, lane1)
    for lane0 in ("I", "X", "Y", "Z")
    for lane1 in ("I", "X", "Y", "Z")
    if (lane0, lane1) != ("I", "I")
)
CHANNELS: dict[str, tuple[tuple[str, str], ...]] = {
    "full_packed_depolarizing": FULL_PACKED_DEPOLARIZING,
    "lane0_only": (("X", "I"), ("Y", "I"), ("Z", "I")),
    "lane1_only": (("I", "X"), ("I", "Y"), ("I", "Z")),
    "same_pauli_correlated": (("X", "X"), ("Y", "Y"), ("Z", "Z")),
}


def deterministic_rng(seed: int, *parts: object) -> random.Random:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    label = "|".join((str(seed), *(str(part) for part in parts)))
    digest = hashlib.sha256(label.encode("utf-8")).digest()
    return random.Random(int.from_bytes(digest[:16], "big"))


def sample_error(
    rng: random.Random,
    *,
    width: int,
    error_rate: float,
    channel: str = "full_packed_depolarizing",
) -> PackedPauli:
    """Sample independent per-site faults from one declared local channel."""

    if width <= 0:
        raise ValueError("width must be positive")
    if error_rate < 0.0 or error_rate > 1.0:
        raise ValueError("error_rate must be in [0, 1]")
    try:
        local_errors = CHANNELS[channel]
    except KeyError as error:
        raise ValueError(f"unknown ququart channel: {channel}") from error

    result = identity(width)
    for site in range(width):
        if rng.random() >= error_rate:
            continue
        lane0, lane1 = local_errors[rng.randrange(len(local_errors))]
        result = result.compose(local_pauli(width, site, lane0, lane1))
    return result


def classify_error(decoder: ExactDecoder, error: PackedPauli) -> str:
    try:
        result = decoder.correct(error)
    except UncorrectableSyndrome:
        return "detected_uncorrectable"
    return "corrected" if result.success else "logical_failure"


def _ratio(numerator: int, denominator: int) -> str:
    if denominator <= 0:
        raise ValueError("denominator must be positive")
    return format(Decimal(numerator) / Decimal(denominator), ".12E")


def wilson_interval(
    failures: int,
    trials: int,
    *,
    z: float = 1.959963984540054,
) -> tuple[str, str]:
    """Return a deterministic 95% Wilson interval as decimal strings."""

    if trials <= 0:
        raise ValueError("trials must be positive")
    if failures < 0 or failures > trials:
        raise ValueError("failures must be between zero and trials")
    p = failures / trials
    denominator = 1.0 + (z * z) / trials
    centre = (p + (z * z) / (2.0 * trials)) / denominator
    radius = (
        z
        * math.sqrt(
            (p * (1.0 - p) / trials)
            + (z * z) / (4.0 * trials * trials)
        )
        / denominator
    )
    return (
        format(max(0.0, centre - radius), ".12E"),
        format(min(1.0, centre + radius), ".12E"),
    )


def monte_carlo_rows(
    *,
    error_rates: Iterable[str] = DEFAULT_ERROR_RATES,
    channels: Sequence[str] = tuple(CHANNELS),
    trials: int = 5000,
    seed: int = 1701001,
) -> tuple[dict[str, str | int], ...]:
    """Run deterministic Monte Carlo cells with stable per-cell seeds."""

    if isinstance(trials, bool) or not isinstance(trials, int):
        raise TypeError("trials must be an integer")
    if trials <= 0:
        raise ValueError("trials must be positive")

    code = packed_five_ququart_code()
    decoder = ExactDecoder(code, max_weight=1)
    rows: list[dict[str, str | int]] = []
    for channel in channels:
        if channel not in CHANNELS:
            raise ValueError(f"unknown ququart channel: {channel}")
        for rate_text in error_rates:
            p = float(rate_text)
            rng = deterministic_rng(seed, "monte-carlo", channel, rate_text)
            counts = {
                "corrected": 0,
                "detected_uncorrectable": 0,
                "logical_failure": 0,
            }
            for _ in range(trials):
                error = sample_error(
                    rng,
                    width=code.n,
                    error_rate=p,
                    channel=channel,
                )
                counts[classify_error(decoder, error)] += 1
            frame_errors = (
                counts["detected_uncorrectable"]
                + counts["logical_failure"]
            )
            lower, upper = wilson_interval(frame_errors, trials)
            rows.append({
                "channel": channel,
                "physical_error_rate": rate_text,
                "trials": trials,
                "seed": seed,
                "corrected": counts["corrected"],
                "detected_uncorrectable": counts["detected_uncorrectable"],
                "logical_failure": counts["logical_failure"],
                "frame_errors": frame_errors,
                "frame_error_rate": _ratio(frame_errors, trials),
                "wilson95_low": lower,
                "wilson95_high": upper,
            })
    return tuple(rows)
