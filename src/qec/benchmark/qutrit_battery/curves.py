"""Exact code-capacity curves and guaranteed-radius bounds."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, localcontext
from math import comb
from typing import Iterable

from .prime import PrimeStabilizerModel

DEFAULT_ERROR_RATES = (
    "1e-6",
    "3e-6",
    "1e-5",
    "3e-5",
    "1e-4",
    "3e-4",
    "1e-3",
    "3e-3",
    "1e-2",
    "3e-2",
    "5e-2",
    "1e-1",
    "2e-1",
)
EXACT_ENUMERATION_LIMIT = 1_000_000


@dataclass(frozen=True)
class BoundCode:
    code_id: str
    label: str
    family: str
    modulus: int
    n: int
    k: int
    distance: int
    source_url: str

    @property
    def radius(self) -> int:
        return (self.distance - 1) // 2


def decimal_text(value: Decimal) -> str:
    """Return a stable 13-significant-digit scientific value."""
    if not value:
        return "0"
    return format(value, ".12E").replace("E+", "e").replace("E-", "e-")


def _validate_probability(text: str) -> Decimal:
    value = Decimal(text)
    if value < 0 or value > 1:
        raise ValueError("error rates must be in [0,1]")
    return value


def radius_tail_probability(n: int, radius: int, error_rate: Decimal) -> Decimal:
    """P(W > radius) for iid per-site error probability."""
    with localcontext() as context:
        context.prec = 80
        guaranteed = sum(
            Decimal(comb(n, weight))
            * error_rate**weight
            * (Decimal(1) - error_rate) ** (n - weight)
            for weight in range(radius + 1)
        )
        return max(Decimal(0), Decimal(1) - guaranteed)


def decoded_failure_probability(
    model: PrimeStabilizerModel,
    success_counts: tuple[int, ...],
    error_rate: Decimal,
) -> Decimal:
    """Exact bounded-decoder block failure under iid depolarizing noise."""
    with localcontext() as context:
        context.prec = 80
        local_error = error_rate / Decimal(model.modulus * model.modulus - 1)
        success = sum(
            Decimal(count)
            * local_error**weight
            * (Decimal(1) - error_rate) ** (model.n - weight)
            for weight, count in enumerate(success_counts)
        )
        return max(Decimal(0), Decimal(1) - success)


def decoded_curve_rows(
    models: Iterable[PrimeStabilizerModel],
    error_rates: Iterable[str] = DEFAULT_ERROR_RATES,
) -> list[dict[str, str | int]]:
    rows: list[dict[str, str | int]] = []
    for model in models:
        counts = model.success_weight_counts(
            operation_limit=EXACT_ENUMERATION_LIMIT,
        )
        if counts is None:
            continue
        for error_rate_text in error_rates:
            error_rate = _validate_probability(error_rate_text)
            failure = decoded_failure_probability(model, counts, error_rate)
            rows.append({
                "error_rate": error_rate_text,
                "code_id": model.code_id,
                "label": model.label,
                "origin": model.origin,
                "alphabet_dimension": model.modulus,
                "n": model.n,
                "k": model.k,
                "distance": model.distance,
                "radius": model.radius,
                "logical_failure_rate": decimal_text(failure),
                "physical_to_logical_suppression": (
                    decimal_text(error_rate / failure) if failure else "inf"
                ),
                "estimator": "exact_success_coset_weight_enumerator",
                "noise_model": "iid_depolarizing_code_capacity_per_site",
                "syndrome_model": "perfect",
                "decoder": "exact_bounded_coset_leader",
                "enumerated_success_errors": sum(counts),
                "claim_scope": "simulation_only_not_hardware_performance",
            })
    return rows


def bound_codes(models: Iterable[PrimeStabilizerModel]) -> tuple[BoundCode, ...]:
    declared = [
        BoundCode(
            model.code_id,
            model.label,
            model.family,
            model.modulus,
            model.n,
            model.k,
            model.distance,
            model.source_url,
        )
        for model in models
    ]
    declared.extend((
        BoundCode(
            "surface_rotated_d5",
            "Rotated surface [[25,1,5]]",
            "surface",
            2,
            25,
            1,
            5,
            "https://www.nature.com/articles/s41586-024-08449-y",
        ),
        BoundCode(
            "surface_rotated_d7",
            "Rotated surface [[49,1,7]]",
            "surface",
            2,
            49,
            1,
            7,
            "https://www.nature.com/articles/s41586-024-08449-y",
        ),
        BoundCode(
            "quantinuum_12_2_4",
            "C4/C6-derived [[12,2,4]]",
            "concatenated_error_detecting",
            2,
            12,
            2,
            4,
            "https://arxiv.org/abs/2404.02280",
        ),
        BoundCode(
            "tesseract_16_4_4",
            "Tesseract subsystem [[16,4,4]]",
            "subsystem_color",
            2,
            16,
            4,
            4,
            "https://arxiv.org/abs/2409.04628",
        ),
        BoundCode(
            "bivariate_bicycle_144_12_12",
            "Bivariate bicycle [[144,12,12]]",
            "qldpc",
            2,
            144,
            12,
            12,
            "https://arxiv.org/abs/2308.07915",
        ),
    ))
    return tuple(declared)


def radius_bound_rows(
    codes: Iterable[BoundCode],
    error_rates: Iterable[str] = DEFAULT_ERROR_RATES,
) -> list[dict[str, str | int]]:
    rows: list[dict[str, str | int]] = []
    for code in codes:
        for error_rate_text in error_rates:
            error_rate = _validate_probability(error_rate_text)
            tail = radius_tail_probability(code.n, code.radius, error_rate)
            rows.append({
                "error_rate": error_rate_text,
                "code_id": code.code_id,
                "label": code.label,
                "family": code.family,
                "alphabet_dimension": code.modulus,
                "n": code.n,
                "k": code.k,
                "distance": code.distance,
                "radius": code.radius,
                "uncorrectable_weight_tail": decimal_text(tail),
                "metric_role": "rigorous_logical_failure_upper_bound",
                "noise_model": "iid_per_site_weight_only",
                "syndrome_model": "perfect",
                "source_url": code.source_url,
                "claim_scope": "code_capacity_bound_not_decoder_or_hardware_result",
            })
    return rows
