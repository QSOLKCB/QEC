import copy
import json
from pathlib import Path

from qec.orchestration.governance_topology_forecast_stability_kernel import (
    TopologyForecastMetric,
    TopologyForecastScenario,
    build_topology_forecast_receipt,
    build_topology_forecast_scenario,
    compare_topology_forecast_replay,
    run_governance_topology_forecast_stability,
    summarize_topology_forecast_stability,
    validate_topology_forecast_stability,
)


MODULE_PATH = Path("src/qec/orchestration/governance_topology_forecast_stability_kernel.py")


def _base_scenario() -> TopologyForecastScenario:
    return build_topology_forecast_scenario(
        scenario_id="scenario.base",
        topology_stability_series=(
            {
                "topology_id": "t0",
                "coherence": 1.0,
                "alignment": 1.0,
                "pressure": 0.0,
                "continuity_ok": True,
                "replay_identity": "r",
            },
            {
                "topology_id": "t1",
                "coherence": 1.0,
                "alignment": 1.0,
                "pressure": 0.0,
                "continuity_ok": True,
                "replay_identity": "r",
            },
        ),
        replay_horizon_series=(
            {"horizon_id": "h1", "horizon_step": 1.0, "forecast_pressure": 0.0, "projection_delta": 0.0},
            {"horizon_id": "h2", "horizon_step": 2.0, "forecast_pressure": 0.0, "projection_delta": 0.0},
        ),
    )


def test_deterministic_repeated_forecasting():
    scenario = _base_scenario()
    first = run_governance_topology_forecast_stability(scenario=scenario)
    second = run_governance_topology_forecast_stability(scenario=scenario)
    assert first.to_dict() == second.to_dict()
    assert first.stable_hash() == second.stable_hash()


def test_stable_hash_reproducibility():
    scenario = _base_scenario()
    assert scenario.stable_hash() == _base_scenario().stable_hash()
    metric = TopologyForecastMetric("forecast_confidence_score", 5, 0.7)
    assert metric.stable_hash() == TopologyForecastMetric("forecast_confidence_score", 5, 0.7).stable_hash()


def test_advisory_band_stable_forecast():
    scenario = _base_scenario()
    kernel = run_governance_topology_forecast_stability(scenario=scenario)
    assert kernel.advisory_output == "stable_forecast"


def test_advisory_band_minor_forecast_variation():
    scenario = _base_scenario()
    metrics = tuple(
        TopologyForecastMetric(name, idx, 0.0 if name != "forecast_surface_pressure" else 0.10)
        for idx, name in enumerate(
            (
                "forecast_coherence_score",
                "horizon_projection_gradient",
                "forecast_surface_pressure",
                "topology_forecast_alignment",
                "replay_forecast_stability_score",
                "forecast_confidence_score",
            )
        )
    )
    receipt = build_topology_forecast_receipt(scenario=scenario, metrics=metrics, forecast_hash="abc")
    assert receipt.advisory_output == "minor_forecast_variation"


def test_advisory_band_moderate_forecast_instability():
    scenario = _base_scenario()
    metrics = tuple(
        TopologyForecastMetric(name, idx, 0.0 if name != "forecast_surface_pressure" else 0.30)
        for idx, name in enumerate(
            (
                "forecast_coherence_score",
                "horizon_projection_gradient",
                "forecast_surface_pressure",
                "topology_forecast_alignment",
                "replay_forecast_stability_score",
                "forecast_confidence_score",
            )
        )
    )
    receipt = build_topology_forecast_receipt(scenario=scenario, metrics=metrics, forecast_hash="abc")
    assert receipt.advisory_output == "moderate_forecast_instability"


def test_advisory_band_severe_forecast_instability():
    scenario = _base_scenario()
    metrics = tuple(
        TopologyForecastMetric(name, idx, 0.0 if name != "forecast_surface_pressure" else 0.70)
        for idx, name in enumerate(
            (
                "forecast_coherence_score",
                "horizon_projection_gradient",
                "forecast_surface_pressure",
                "topology_forecast_alignment",
                "replay_forecast_stability_score",
                "forecast_confidence_score",
            )
        )
    )
    receipt = build_topology_forecast_receipt(scenario=scenario, metrics=metrics, forecast_hash="abc")
    assert receipt.advisory_output == "severe_forecast_instability"


def test_canonical_json_round_trip():
    scenario = _base_scenario()
    parsed = json.loads(scenario.to_canonical_json())
    rebuilt = TopologyForecastScenario(
        scenario_id=parsed["scenario_id"],
        topology_stability_series=tuple(parsed["topology_stability_series"]),
        replay_horizon_series=tuple(parsed["replay_horizon_series"]),
    )
    assert scenario.to_canonical_json() == rebuilt.to_canonical_json()


def test_validator_never_raises_on_malformed_kernel():
    class BadKernel:
        @property
        def scenario(self):
            raise RuntimeError("boom")

    violations = validate_topology_forecast_stability(BadKernel())
    assert isinstance(violations, tuple)
    assert "missing_scenario" in violations


def test_malformed_input_normalizes_safely_and_reports_violations():
    scenario = build_topology_forecast_scenario(
        scenario_id=None,
        topology_stability_series=[{"topology_id": None, "coherence": "NaN", "alignment": -10}],
        replay_horizon_series=[{"horizon_id": None, "horizon_step": "bad", "forecast_pressure": "bad"}],
    )
    kernel = run_governance_topology_forecast_stability(scenario=scenario)
    assert kernel.scenario.scenario_id == ""
    assert "empty_scenario_id" in kernel.violations


def test_metric_bounds_and_fixed_order():
    kernel = run_governance_topology_forecast_stability(scenario=_base_scenario())
    expected = (
        "forecast_coherence_score",
        "horizon_projection_gradient",
        "forecast_surface_pressure",
        "topology_forecast_alignment",
        "replay_forecast_stability_score",
        "forecast_confidence_score",
    )
    assert tuple(m.metric_name for m in kernel.metrics) == expected
    for metric in kernel.metrics:
        assert 0.0 <= metric.metric_value <= 1.0


def test_replay_comparison_stability():
    scenario = _base_scenario()
    baseline = run_governance_topology_forecast_stability(scenario=scenario)
    replay = run_governance_topology_forecast_stability(scenario=scenario)
    comparison = compare_topology_forecast_replay(baseline, replay)
    assert comparison["is_stable_replay"] is True
    assert comparison["mismatches"] == ()
    assert all(delta == 0.0 for _, delta in comparison["metric_delta"])


def test_no_input_mutation():
    topology = [
        {"topology_id": "a", "coherence": 0.2, "alignment": 0.4, "pressure": 0.3, "continuity_ok": True, "replay_identity": "x"}
    ]
    horizons = [{"horizon_id": "h", "horizon_step": 2, "forecast_pressure": 0.3, "projection_delta": 0.2}]
    topology_before = copy.deepcopy(topology)
    horizons_before = copy.deepcopy(horizons)

    scenario = build_topology_forecast_scenario(
        scenario_id="immutability",
        topology_stability_series=topology,
        replay_horizon_series=horizons,
    )
    run_governance_topology_forecast_stability(scenario=scenario)

    assert topology == topology_before
    assert horizons == horizons_before


def test_decoder_untouched_confirmation():
    source = MODULE_PATH.read_text(encoding="utf-8")
    assert "qec.decoder" not in source
    assert "src/qec/decoder/" not in source


def test_receipt_determinism():
    kernel = run_governance_topology_forecast_stability(scenario=_base_scenario())
    again = build_topology_forecast_receipt(
        scenario=kernel.scenario,
        metrics=kernel.metrics,
        forecast_hash=kernel.forecast_hash,
    )
    assert again.to_dict() == kernel.receipt.to_dict()


def test_summary_content():
    kernel = run_governance_topology_forecast_stability(scenario=_base_scenario())
    summary = summarize_topology_forecast_stability(kernel)
    assert "scenario_id=scenario.base" in summary
    assert "advisory_output=stable_forecast" in summary
    assert "metrics:" in summary
    assert "violations:" in summary
