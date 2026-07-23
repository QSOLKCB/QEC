"""Deterministic all-weight decoder stress corpus."""

from __future__ import annotations

import hashlib
from decimal import Decimal
from math import comb

from .curves import decimal_text
from .prime import PrimeStabilizerModel, error_pattern_count

CORPUS_LIMIT_PER_WEIGHT = 2048
SELECTION_POLICY = "exact_or_evenly_spaced_ordinal_v1"


def _unrank_combination(width: int, weight: int, index: int) -> tuple[int, ...]:
    support = []
    start = 0
    for position in range(weight):
        for site in range(start, width):
            remaining = weight - position - 1
            count = comb(width - site - 1, remaining)
            if index < count:
                support.append(site)
                start = site + 1
                break
            index -= count
    return tuple(support)


def error_from_ordinal(
    model: PrimeStabilizerModel,
    weight: int,
    ordinal: int,
) -> tuple[int, ...]:
    """Map a canonical ordinal to one exact-weight generalized Pauli."""
    local = tuple(
        (x, z)
        for x in range(model.modulus)
        for z in range(model.modulus)
        if (x, z) != (0, 0)
    )
    local_space = len(local) ** weight
    support = _unrank_combination(model.n, weight, ordinal // local_space)
    powers_ordinal = ordinal % local_space
    powers = []
    for _ in range(weight):
        powers.append(local[powers_ordinal % len(local)])
        powers_ordinal //= len(local)
    powers.reverse()

    error = [0] * (2 * model.n)
    for site, (x_power, z_power) in zip(support, powers):
        error[site] = x_power
        error[model.n + site] = z_power
    return tuple(error)


def _ordinals(total: int, limit: int) -> tuple[int, ...]:
    if total <= limit:
        return tuple(range(total))
    return tuple(
        ((2 * index + 1) * total) // (2 * limit)
        for index in range(limit)
    )


def stress_rows(
    models: tuple[PrimeStabilizerModel, ...],
    *,
    limit_per_weight: int = CORPUS_LIMIT_PER_WEIGHT,
) -> list[dict[str, str | int]]:
    """Classify a hash-bound deterministic corpus at every physical weight."""
    if limit_per_weight < 1:
        raise ValueError("limit_per_weight must be positive")
    rows = []
    for model in models:
        for weight in range(model.n + 1):
            total = error_pattern_count(model.n, weight, model.modulus)
            ordinals = _ordinals(total, limit_per_weight)
            counts = {"corrected": 0, "rejected": 0, "miscorrected": 0}
            digest = hashlib.sha256()
            for ordinal in ordinals:
                error = error_from_ordinal(model, weight, ordinal)
                digest.update(ordinal.to_bytes(16, "big"))
                digest.update(bytes(error))
                counts[model.classify(error)] += 1

            tested = len(ordinals)
            failures = counts["rejected"] + counts["miscorrected"]
            rows.append({
                "code_id": model.code_id,
                "alphabet_dimension": model.modulus,
                "n": model.n,
                "k": model.k,
                "distance": model.distance,
                "decoder_radius": model.radius,
                "error_weight": weight,
                "total_error_patterns": total,
                "patterns_tested": tested,
                "coverage": "exhaustive" if tested == total else "deterministic_corpus",
                "selection_policy": SELECTION_POLICY,
                "corpus_sha256": digest.hexdigest(),
                "corrected": counts["corrected"],
                "rejected": counts["rejected"],
                "miscorrected": counts["miscorrected"],
                "conditional_failure_fraction": decimal_text(
                    Decimal(failures) / Decimal(tested)
                ),
                "statistical_claim": "none",
            })
    return rows
