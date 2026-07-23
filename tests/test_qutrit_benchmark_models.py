from __future__ import annotations

from decimal import Decimal
from math import comb

import pytest

from qec.benchmark.qutrit_battery.codes import benchmark_models
from qec.benchmark.qutrit_battery.curves import (
    decoded_failure_probability,
    radius_tail_probability,
)
from qec.benchmark.qutrit_battery.prime import paulis_of_weight
from qec.benchmark.qutrit_battery.stress import (
    error_from_ordinal,
    stress_rows,
)
from qec.decoder.qutrit import (
    ExactDecoder,
    Pauli,
    UncorrectableSyndrome,
    cyclic_five_qutrit_code,
    shor_nine_qutrit_code,
    ternary_golay_qutrit_code,
)

EXPECTED_PARAMETERS = {
    "qutrit_cyclic_5": (3, 5, 1, 3, 1),
    "qutrit_shor_9": (3, 9, 1, 3, 1),
    "qutrit_golay_11": (3, 11, 1, 5, 2),
    "five_qubit_5": (2, 5, 1, 3, 1),
    "steane_7": (2, 7, 1, 3, 1),
    "surface_rotated_d3": (2, 9, 1, 3, 1),
    "reed_muller_15": (2, 15, 1, 3, 1),
}


def test_declared_models_have_verified_parameters_and_exact_radius():
    models = benchmark_models()
    assert {model.code_id for model in models} == set(EXPECTED_PARAMETERS)
    for model in models:
        assert (
            model.modulus,
            model.n,
            model.k,
            model.distance,
            model.radius,
        ) == EXPECTED_PARAMETERS[model.code_id]
        for weight in range(model.radius + 1):
            assert all(
                model.classify(error) == "corrected"
                for error in paulis_of_weight(
                    model.n,
                    weight,
                    model.modulus,
                )
            )


@pytest.mark.parametrize(
    ("model_id", "factory"),
    [
        ("qutrit_cyclic_5", cyclic_five_qutrit_code),
        ("qutrit_shor_9", shor_nine_qutrit_code),
        ("qutrit_golay_11", ternary_golay_qutrit_code),
    ],
)
def test_benchmark_oracle_matches_production_qutrit_decoder(model_id, factory):
    model = next(item for item in benchmark_models() if item.code_id == model_id)
    code = factory()
    decoder = ExactDecoder(code, max_weight=model.radius)
    errors = []
    for weight in range(model.radius + 1):
        errors.extend(paulis_of_weight(model.n, weight, 3))
    for weight in range(model.radius + 1, min(model.n, model.radius + 2) + 1):
        total = comb(model.n, weight) * 8**weight
        errors.extend(
            error_from_ordinal(model, weight, ordinal)
            for ordinal in (
                ((2 * index + 1) * total) // 128
                for index in range(64)
            )
        )

    for vector in errors:
        error = Pauli(vector[:model.n], vector[model.n:])
        try:
            result = decoder.correct(error)
        except UncorrectableSyndrome:
            production = "rejected"
        else:
            production = "corrected" if result.success else "miscorrected"
        assert model.classify(vector) == production


def test_binary_reference_distances_are_independently_witnessed():
    for model in (item for item in benchmark_models() if item.modulus == 2):
        for weight in range(1, model.distance):
            for error in paulis_of_weight(model.n, weight, 2):
                assert any(model.syndrome(error)) or model.is_stabilizer(error)
        assert any(
            not any(model.syndrome(error)) and not model.is_stabilizer(error)
            for error in paulis_of_weight(model.n, model.distance, 2)
        )


def test_exact_decoded_curves_respect_the_guaranteed_radius_bound():
    for model in benchmark_models():
        counts = model.success_weight_counts(operation_limit=1_000_000)
        if counts is None:
            continue
        for error_rate in map(Decimal, ("1e-6", "1e-3", "1e-2", "0.1")):
            decoded = decoded_failure_probability(model, counts, error_rate)
            bound = radius_tail_probability(model.n, model.radius, error_rate)
            assert Decimal(0) <= decoded <= bound <= Decimal(1)


def test_stress_corpus_has_no_false_failure_inside_certified_radius():
    rows = stress_rows(benchmark_models(), limit_per_weight=32)
    for row in rows:
        if row["error_weight"] <= row["decoder_radius"]:
            assert row["corrected"] == row["patterns_tested"]
            assert row["rejected"] == 0
            assert row["miscorrected"] == 0


def test_error_ordinals_are_a_bijection_for_a_small_stratum():
    model = next(item for item in benchmark_models() if item.code_id == "five_qubit_5")
    expected = set(paulis_of_weight(model.n, 2, model.modulus))
    observed = {
        error_from_ordinal(model, 2, ordinal)
        for ordinal in range(len(expected))
    }
    assert observed == expected
