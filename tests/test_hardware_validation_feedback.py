from __future__ import annotations

import math

import pytest

from qec.analysis.hardware_validation_feedback import update_hardware_validation_config


def test_fail_case_reduces_thresholds_and_increases_agreement() -> None:
    result = update_hardware_validation_config(
        {
            "max_mean_relative_error": 0.25,
            "max_mean_absolute_error": 2.0,
            "min_mean_agreement_score": 0.95,
        },
        {"status": "FAIL"},
    )

    assert result == {
        "max_mean_relative_error": 0.2,
        "max_mean_absolute_error": 1.6,
        "min_mean_agreement_score": 1.0,
    }


def test_warn_case_reduces_thresholds_and_increases_agreement() -> None:
    result = update_hardware_validation_config(
        {
            "max_mean_relative_error": 1.0,
            "max_mean_absolute_error": 10.0,
            "min_mean_agreement_score": 0.8,
        },
        {"status": "WARN"},
    )

    assert result == {
        "max_mean_relative_error": 0.9,
        "max_mean_absolute_error": 9.0,
        "min_mean_agreement_score": 0.84,
    }


def test_pass_case_increases_thresholds_and_decreases_agreement() -> None:
    result = update_hardware_validation_config(
        {
            "max_mean_relative_error": 0.4,
            "max_mean_absolute_error": 0.6,
            "min_mean_agreement_score": 0.5,
        },
        {"status": "PASS"},
    )

    assert result == {
        "max_mean_relative_error": 0.42,
        "max_mean_absolute_error": 0.63,
        "min_mean_agreement_score": 0.49,
    }


def test_agreement_bounds_are_enforced() -> None:
    fail_capped = update_hardware_validation_config(
        {
            "max_mean_relative_error": 0.1,
            "max_mean_absolute_error": 0.1,
            "min_mean_agreement_score": 0.99,
        },
        {"status": "FAIL"},
    )
    pass_floored = update_hardware_validation_config(
        {
            "max_mean_relative_error": 0.1,
            "max_mean_absolute_error": 0.1,
            "min_mean_agreement_score": 0.001,
        },
        {"status": "PASS"},
    )

    # For failing runs, agreement is increased but capped at 1.0
    assert fail_capped["min_mean_agreement_score"] == 1.0
    # For passing runs with small positive agreement, agreement is decreased but floored at 0.0
    assert pass_floored["min_mean_agreement_score"] == 0.0


@pytest.mark.parametrize(
    "bad_value",
    [math.nan, math.inf, -math.inf, True, False],
)
def test_invalid_config_values_raise_value_error(bad_value: float | bool) -> None:
    config = {
        "max_mean_relative_error": 0.1,
        "max_mean_absolute_error": 0.2,
        "min_mean_agreement_score": 0.9,
    }
    for key in config:
        mutated = dict(config)
        mutated[key] = bad_value
        with pytest.raises(ValueError):
            update_hardware_validation_config(mutated, {"status": "WARN"})


def test_invalid_status_raises_value_error() -> None:
    with pytest.raises(ValueError, match="status"):
        update_hardware_validation_config(
            {
                "max_mean_relative_error": 0.1,
                "max_mean_absolute_error": 0.2,
                "min_mean_agreement_score": 0.9,
            },
            {"status": "UNKNOWN"},
        )


def test_deterministic_replay_same_input_same_output() -> None:
    config = {
        "max_mean_relative_error": 0.123456789012,
        "max_mean_absolute_error": 9.876543210987,
        "min_mean_agreement_score": 0.456789012345,
    }
    governance_result = {"status": "WARN"}

    first = update_hardware_validation_config(config, governance_result)
    second = update_hardware_validation_config(config, governance_result)

    assert first == second
