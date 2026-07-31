"""Deterministic harmonic receiver fault and end-to-end batteries."""

from __future__ import annotations

import random
from decimal import Decimal
from typing import Callable, Iterable, Mapping, Sequence

from qec.decoder.ququart.codes import packed_five_ququart_code
from qec.decoder.ququart.cycle import run_harmonic_cycle
from qec.decoder.ququart.exact import ExactDecoder
from qec.decoder.ququart.packed import paulis_of_weight
from qec.decoder.ququart.syndrome import pack_binary_syndrome
from qec.sonify.ququart_harmonics import encode_harmonics, phasor

from .channels import deterministic_rng, sample_error, wilson_interval

DEFAULT_NOISE_SIGMAS = ("0", "0.02", "0.05", "0.10", "0.20", "0.35", "0.50")

Samples = dict[int, tuple[complex, ...]]
Fault = Callable[[Samples, tuple[int, ...]], Samples]


def _copy_samples(samples: Mapping[int, Sequence[complex]]) -> Samples:
    return {
        order: tuple(complex(value) for value in values)
        for order, values in samples.items()
    }


def _clean(samples: Samples, syndrome: tuple[int, ...]) -> Samples:
    return _copy_samples(samples)


def _bounded_noise(samples: Samples, syndrome: tuple[int, ...]) -> Samples:
    offsets = {
        1: complex(0.025, -0.015),
        2: complex(-0.020, 0.010),
        3: complex(0.015, 0.020),
        4: complex(-0.010, -0.015),
    }
    return {
        order: tuple(value + offsets.get(order, 0j) for value in values)
        for order, values in samples.items()
    }


def _missing_h3(samples: Samples, syndrome: tuple[int, ...]) -> Samples:
    result = _copy_samples(samples)
    result.pop(3, None)
    return result


def _h1_h3_disagreement(samples: Samples, syndrome: tuple[int, ...]) -> Samples:
    result = _copy_samples(samples)
    values = list(result[3])
    values[0] = phasor((syndrome[0] + 1) % 4, 3)
    result[3] = tuple(values)
    return result


def _h2_parity_flip(samples: Samples, syndrome: tuple[int, ...]) -> Samples:
    result = _copy_samples(samples)
    values = list(result[2])
    values[0] = phasor((syndrome[0] + 1) % 4, 2)
    result[2] = tuple(values)
    return result


def _h4_dark_distortion(samples: Samples, syndrome: tuple[int, ...]) -> Samples:
    result = _copy_samples(samples)
    values = list(result[4])
    values[0] = 0j
    result[4] = tuple(values)
    return result


def _ambiguous_h1(samples: Samples, syndrome: tuple[int, ...]) -> Samples:
    result = _copy_samples(samples)
    values = list(result[1])
    left = phasor(syndrome[0], 1)
    right = phasor((syndrome[0] + 1) % 4, 1)
    values[0] = (left + right) / 2
    result[1] = tuple(values)
    return result


FAULTS: tuple[tuple[str, Fault, str], ...] = (
    ("clean", _clean, "accept_and_correct_all"),
    ("bounded_complex_noise", _bounded_noise, "accept_and_correct_all"),
    ("missing_h3", _missing_h3, "reject_all"),
    ("h1_h3_disagreement", _h1_h3_disagreement, "reject_all"),
    ("h2_parity_flip", _h2_parity_flip, "reject_all"),
    ("h4_dark_distortion", _h4_dark_distortion, "reject_all"),
    ("ambiguous_h1", _ambiguous_h1, "reject_all"),
)


def harmonic_fault_rows() -> tuple[dict[str, str | int], ...]:
    """Exhaustively inject deterministic receiver faults over all 75 errors."""

    code = packed_five_ququart_code()
    decoder = ExactDecoder(code, max_weight=1)
    errors = tuple(paulis_of_weight(code.n, 1))
    rows: list[dict[str, str | int]] = []

    for name, fault, expected in FAULTS:
        accepted = 0
        successful = 0
        false_accepts = 0
        receiver_rejections = 0
        decoder_rejections = 0
        receiver_false_trust = 0
        accepted_logical_residual = 0
        for error in errors:
            binary = code.syndrome(error)
            syndrome = pack_binary_syndrome(binary)
            samples = encode_harmonics(syndrome)
            result = run_harmonic_cycle(
                decoder,
                error,
                samples=fault(samples, syndrome),
            )
            trusted = result.observation.trusted
            syndrome_correct = trusted and result.observation.syndrome == syndrome
            accepted += int(result.accepted)
            successful += int(result.success)
            false_accepts += int(result.accepted and not result.success)
            receiver_rejections += int(not trusted)
            decoder_rejections += int(trusted and not result.accepted)
            receiver_false_trust += int(trusted and not syndrome_correct)
            accepted_logical_residual += int(
                result.accepted and syndrome_correct and not result.success
            )
        rows.append({
            "fault_case": name,
            "errors_tested": len(errors),
            "accepted": accepted,
            "rejected": len(errors) - accepted,
            "receiver_rejections": receiver_rejections,
            "decoder_rejections": decoder_rejections,
            "successful": successful,
            "false_accepts": false_accepts,
            "receiver_false_trust": receiver_false_trust,
            "incorrect_trusted_syndrome": receiver_false_trust,
            "accepted_logical_residual": accepted_logical_residual,
            "expected": expected,
            "claim_scope": "classical_harmonic_observation_fault_injection",
        })
    return tuple(rows)


def _noisy_samples(
    rng: random.Random,
    syndrome: tuple[int, ...],
    sigma: float,
) -> Samples:
    clean = encode_harmonics(syndrome)
    if sigma == 0.0:
        return clean
    return {
        order: tuple(
            value + complex(rng.gauss(0.0, sigma), rng.gauss(0.0, sigma))
            for value in values
        )
        for order, values in clean.items()
    }


def _ratio(numerator: int, denominator: int) -> str:
    if denominator <= 0:
        raise ValueError("denominator must be positive")
    return format(Decimal(numerator) / Decimal(denominator), ".12E")


def harmonic_end_to_end_rows(
    *,
    physical_error_rates: Iterable[str] = ("0.001", "0.003", "0.01", "0.03"),
    noise_sigmas: Iterable[str] = DEFAULT_NOISE_SIGMAS,
    trials: int = 2000,
    seed: int = 1701001,
    tolerance: float = 0.35,
) -> tuple[dict[str, str | int], ...]:
    """Combine iid packed faults with noisy H1/H3/H2/H4 observation.

    Receiver trust, decoder boundedness, and residual correctness are counted as
    separate layers so an accepted logical residual is not mislabeled as a
    harmonic false-trust event.
    """

    if isinstance(trials, bool) or not isinstance(trials, int):
        raise TypeError("trials must be an integer")
    if trials <= 0:
        raise ValueError("trials must be positive")

    code = packed_five_ququart_code()
    decoder = ExactDecoder(code, max_weight=1)
    rows: list[dict[str, str | int]] = []
    for p_text in physical_error_rates:
        p = float(p_text)
        for sigma_text in noise_sigmas:
            sigma = float(sigma_text)
            rng = deterministic_rng(
                seed,
                "harmonic-end-to-end",
                p_text,
                sigma_text,
            )
            accepted = 0
            receiver_rejections = 0
            decoder_rejections = 0
            successful = 0
            false_accepts = 0
            trusted_correct = 0
            receiver_false_trust = 0
            accepted_incorrect_syndrome = 0
            accepted_logical_residual = 0
            for _ in range(trials):
                error = sample_error(
                    rng,
                    width=code.n,
                    error_rate=p,
                    channel="full_packed_depolarizing",
                )
                exact_syndrome = pack_binary_syndrome(code.syndrome(error))
                result = run_harmonic_cycle(
                    decoder,
                    error,
                    samples=_noisy_samples(rng, exact_syndrome, sigma),
                    tolerance=tolerance,
                )
                trusted = result.observation.trusted
                syndrome_correct = (
                    trusted and result.observation.syndrome == exact_syndrome
                )
                accepted += int(result.accepted)
                successful += int(result.success)
                receiver_rejections += int(not trusted)
                decoder_rejections += int(trusted and not result.accepted)
                trusted_correct += int(syndrome_correct)
                receiver_false_trust += int(trusted and not syndrome_correct)
                accepted_incorrect_syndrome += int(
                    result.accepted and trusted and not syndrome_correct
                )
                accepted_logical_residual += int(
                    result.accepted and syndrome_correct and not result.success
                )
                false_accepts += int(result.accepted and not result.success)

            frame_errors = trials - successful
            lower, upper = wilson_interval(frame_errors, trials)
            rows.append({
                "physical_error_rate": p_text,
                "harmonic_noise_sigma": sigma_text,
                "tolerance": format(tolerance, ".6f"),
                "trials": trials,
                "seed": seed,
                "accepted": accepted,
                "rejected": trials - accepted,
                "receiver_rejections": receiver_rejections,
                "decoder_rejections": decoder_rejections,
                "trusted_correct_syndrome": trusted_correct,
                "receiver_false_trust": receiver_false_trust,
                "incorrect_trusted_syndrome": receiver_false_trust,
                "accepted_incorrect_syndrome": accepted_incorrect_syndrome,
                "accepted_logical_residual": accepted_logical_residual,
                "successful": successful,
                "false_accepts": false_accepts,
                "frame_errors": frame_errors,
                "frame_error_rate": _ratio(frame_errors, trials),
                "wilson95_low": lower,
                "wilson95_high": upper,
            })
    return tuple(rows)


def receiver_operating_rows(
    harmonic_rows: Iterable[Mapping[str, str | int]],
) -> tuple[dict[str, str | int], ...]:
    """Project end-to-end rows into disjoint receiver/decoder operating rates."""

    rows: list[dict[str, str | int]] = []
    for row in harmonic_rows:
        trials = int(row["trials"])
        rows.append({
            "physical_error_rate": str(row["physical_error_rate"]),
            "harmonic_noise_sigma": str(row["harmonic_noise_sigma"]),
            "trials": trials,
            "receiver_rejection_rate": _ratio(
                int(row["receiver_rejections"]), trials
            ),
            "decoder_rejection_rate": _ratio(
                int(row["decoder_rejections"]), trials
            ),
            "trusted_correct_syndrome_rate": _ratio(
                int(row["trusted_correct_syndrome"]), trials
            ),
            "receiver_false_trust_rate": _ratio(
                int(row["receiver_false_trust"]), trials
            ),
            "accepted_logical_residual_rate": _ratio(
                int(row["accepted_logical_residual"]), trials
            ),
            "success_rate": _ratio(int(row["successful"]), trials),
            "frame_error_rate": str(row["frame_error_rate"]),
        })
    return tuple(rows)
