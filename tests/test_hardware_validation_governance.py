from __future__ import annotations

import math

import pytest

from qec.analysis import hardware_validation_governance as governance


SCENARIOS = [{"id": "s1", "nodes": ["n0"], "edges": []}]
HARDWARE = {"s1": {"latency": 1.0}}
BASE_CONFIG = {
    "max_mean_relative_error": 0.2,
    "max_mean_absolute_error": 1.0,
    "min_mean_agreement_score": 0.8,
}


def test_perfect_match_pass_and_certified(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        governance,
        "run_hardware_validation_bridge",
        lambda scenarios, hardware_measurements: {
            "mean_relative_error": 0.0,
            "mean_absolute_error": 0.0,
            "mean_agreement_score": 1.0,
        },
    )

    result = governance.run_hardware_validation_governance(
        SCENARIOS,
        HARDWARE,
        BASE_CONFIG,
    )

    assert result["status"] == "PASS"
    assert result["certified"] is True


def test_small_drift_warn(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        governance,
        "run_hardware_validation_bridge",
        lambda scenarios, hardware_measurements: {
            "mean_relative_error": 0.3,
            "mean_absolute_error": 1.4,
            "mean_agreement_score": 0.7,
        },
    )

    result = governance.run_hardware_validation_governance(
        SCENARIOS,
        HARDWARE,
        BASE_CONFIG,
    )

    assert result["status"] == "WARN"
    assert result["certified"] is False


def test_large_drift_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        governance,
        "run_hardware_validation_bridge",
        lambda scenarios, hardware_measurements: {
            "mean_relative_error": 0.45,
            "mean_absolute_error": 0.1,
            "mean_agreement_score": 0.79,
        },
    )

    result = governance.run_hardware_validation_governance(
        SCENARIOS,
        HARDWARE,
        BASE_CONFIG,
    )

    assert result["status"] == "FAIL"
    assert result["certified"] is False


def test_boundary_conditions_exact_thresholds_pass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        governance,
        "run_hardware_validation_bridge",
        lambda scenarios, hardware_measurements: {
            "mean_relative_error": 0.2,
            "mean_absolute_error": 1.0,
            "mean_agreement_score": 0.8,
        },
    )

    result = governance.run_hardware_validation_governance(
        SCENARIOS,
        HARDWARE,
        BASE_CONFIG,
    )

    assert result["status"] == "PASS"
    assert result["certified"] is True


@pytest.mark.parametrize(
    "config,match",
    [
        ({"max_mean_relative_error": -0.1, "max_mean_absolute_error": 1.0, "min_mean_agreement_score": 0.8}, "max_mean_relative_error must be >= 0"),
        ({"max_mean_relative_error": 0.1, "max_mean_absolute_error": -1.0, "min_mean_agreement_score": 0.8}, "max_mean_absolute_error must be >= 0"),
        ({"max_mean_relative_error": 0.1, "max_mean_absolute_error": 1.0, "min_mean_agreement_score": 1.1}, "min_mean_agreement_score must be between 0 and 1"),
        ({"max_mean_relative_error": 0.1, "max_mean_absolute_error": 1.0, "min_mean_agreement_score": "bad"}, "config value for 'min_mean_agreement_score' must be numeric"),
    ],
)
def test_invalid_config_raises_value_error(config: dict[str, object], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        governance.run_hardware_validation_governance(SCENARIOS, HARDWARE, config)


def test_deterministic_replay(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        governance,
        "run_hardware_validation_bridge",
        lambda scenarios, hardware_measurements: {
            "mean_relative_error": 0.2,
            "mean_absolute_error": 1.2,
            "mean_agreement_score": 0.7,
        },
    )

    first = governance.run_hardware_validation_governance(
        SCENARIOS,
        HARDWARE,
        BASE_CONFIG,
    )
    second = governance.run_hardware_validation_governance(
        SCENARIOS,
        HARDWARE,
        BASE_CONFIG,
    )

    assert first == second


def test_duplicate_scenario_ids_raise_value_error() -> None:
    with pytest.raises(ValueError, match="scenario ids must be unique"):
        governance.run_hardware_validation_governance(
            [
                {"id": "s1", "nodes": ["n0"], "edges": []},
                {"id": "s1", "nodes": ["n1"], "edges": []},
            ],
            HARDWARE,
            BASE_CONFIG,
        )


def test_missing_hardware_measurement_raises_value_error() -> None:
    with pytest.raises(ValueError, match="hardware_measurements missing scenario id"):
        governance.run_hardware_validation_governance(SCENARIOS, {}, BASE_CONFIG)


def test_bridge_nan_metric_raises_value_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        governance,
        "run_hardware_validation_bridge",
        lambda scenarios, hardware_measurements: {
            "mean_relative_error": math.nan,
            "mean_absolute_error": 0.0,
            "mean_agreement_score": 1.0,
        },
    )

    with pytest.raises(
        ValueError, match="bridge result value for 'mean_relative_error' must be finite"
    ):
        governance.run_hardware_validation_governance(SCENARIOS, HARDWARE, BASE_CONFIG)


def test_bridge_missing_key_raises_value_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        governance,
        "run_hardware_validation_bridge",
        lambda scenarios, hardware_measurements: {
            "mean_relative_error": 0.0,
            "mean_absolute_error": 0.0,
        },
    )

    with pytest.raises(
        ValueError,
        match="bridge result missing required key 'mean_agreement_score'",
    ):
        governance.run_hardware_validation_governance(SCENARIOS, HARDWARE, BASE_CONFIG)


def test_reordered_scenarios_produce_identical_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        governance,
        "run_hardware_validation_bridge",
        lambda scenarios, hardware_measurements: {
            "mean_relative_error": 0.1,
            "mean_absolute_error": 0.2,
            "mean_agreement_score": 0.9,
        },
    )
    scenarios_a = [
        {"id": "s2", "nodes": ["n1"], "edges": []},
        {"id": "s1", "nodes": ["n0"], "edges": []},
    ]
    scenarios_b = [
        {"id": "s1", "nodes": ["n0"], "edges": []},
        {"id": "s2", "nodes": ["n1"], "edges": []},
    ]
    hardware = {"s1": {"latency": 1.0}, "s2": {"latency": 2.0}}

    result_a = governance.run_hardware_validation_governance(
        scenarios_a,
        hardware,
        BASE_CONFIG,
    )
    result_b = governance.run_hardware_validation_governance(
        scenarios_b,
        hardware,
        BASE_CONFIG,
    )

    assert result_a == result_b
