from __future__ import annotations

import copy
import json
import math

from qec.orchestration.latency_throughput_budget_ledger import (
    ADVISORY_STATES,
    METRIC_ORDER,
    BudgetMetric,
    LatencyThroughputBudgetLedger,
    LatencyThroughputSample,
    build_budget_receipt,
    build_latency_throughput_scenario,
    compare_budget_replay,
    run_latency_throughput_budget_ledger,
    summarize_latency_throughput_budget,
    validate_latency_throughput_budget_ledger,
)


def _scenario() -> dict:
    return {
        "timing_series": [
            {"sample_index": 1, "sample_id": "t1", "latency_ms": 8.0},
            {"sample_index": 0, "sample_id": "t0", "latency_ms": 8.0},
        ],
        "throughput_series": [
            {"sample_index": 1, "sample_id": "p1", "throughput_units": 110.0, "backlog_units": 1.0},
            {"sample_index": 0, "sample_id": "p0", "throughput_units": 120.0, "backlog_units": 1.0},
        ],
        "budget_requirements": {
            "latency_budget_ms": 10.0,
            "min_throughput_units": 100.0,
            "max_backlog_units": 10.0,
            "max_timing_variation_ms": 3.0,
        },
    }


def test_deterministic_repeated_ledger_runs() -> None:
    scenario = build_latency_throughput_scenario(_scenario())
    a = run_latency_throughput_budget_ledger(**scenario)
    b = run_latency_throughput_budget_ledger(**scenario)
    assert a == b


def test_stable_hash_reproducibility() -> None:
    scenario = build_latency_throughput_scenario(_scenario())
    hashes = {run_latency_throughput_budget_ledger(**scenario).stable_hash() for _ in range(4)}
    assert len(hashes) == 1


def test_canonical_json_round_trip() -> None:
    ledger = run_latency_throughput_budget_ledger(**build_latency_throughput_scenario(_scenario()))
    payload = ledger.to_canonical_json()
    parsed = json.loads(payload)
    rebuilt = LatencyThroughputBudgetLedger(
        ledger_version=parsed["ledger_version"],
        timing_series=tuple(LatencyThroughputSample(**item) for item in parsed["timing_series"]),
        throughput_series=tuple(LatencyThroughputSample(**item) for item in parsed["throughput_series"]),
        budget_requirements=parsed["budget_requirements"],
        budget_analysis=tuple(BudgetMetric(**item) for item in parsed["budget_analysis"]),
        advisory_state=parsed["advisory_state"],
        composite_budget_pressure=parsed["composite_budget_pressure"],
        budget_receipt=ledger.budget_receipt,
        normalization_notes=tuple(parsed["normalization_notes"]),
        ledger_hash=parsed["ledger_hash"],
    )
    assert rebuilt.to_canonical_json() == payload


def test_validator_never_raises() -> None:
    report = validate_latency_throughput_budget_ledger({"timing_series": object()})
    assert report["is_valid"] in (True, False)


def test_malformed_input_normalization() -> None:
    scenario = build_latency_throughput_scenario(
        {
            "timing_series": [{"sample_id": "bad", "latency_ms": "x"}],
            "throughput_series": [{"sample_id": "bad2", "throughput_units": "y", "backlog_units": "z"}],
            "budget_requirements": {"min_throughput_units": "oops"},
        }
    )
    assert scenario["timing_series"][0].latency_ms == 0.0
    assert scenario["throughput_series"][0].throughput_units == 0.0
    assert scenario["normalization_notes"]


def test_metric_bounds_and_fixed_order() -> None:
    ledger = run_latency_throughput_budget_ledger(**build_latency_throughput_scenario(_scenario()))
    assert tuple(metric.metric_name for metric in ledger.budget_analysis) == METRIC_ORDER
    assert all(0.0 <= metric.metric_value <= 1.0 for metric in ledger.budget_analysis)


def test_advisory_band_within_budget() -> None:
    ledger = run_latency_throughput_budget_ledger(**build_latency_throughput_scenario(_scenario()))
    assert ledger.advisory_state == "within_budget"


def test_advisory_band_near_budget_limit() -> None:
    s = _scenario()
    s["throughput_series"][0]["backlog_units"] = 2.5
    s["throughput_series"][1]["backlog_units"] = 2.5
    ledger = run_latency_throughput_budget_ledger(**build_latency_throughput_scenario(s))
    assert ledger.advisory_state == "near_budget_limit"


def test_advisory_band_budget_pressure() -> None:
    s = _scenario()
    s["timing_series"][0]["latency_ms"] = 11.0
    s["timing_series"][1]["latency_ms"] = 10.0
    s["throughput_series"][0]["throughput_units"] = 95.0
    s["throughput_series"][1]["throughput_units"] = 100.0
    ledger = run_latency_throughput_budget_ledger(**build_latency_throughput_scenario(s))
    assert ledger.advisory_state == "budget_pressure"


def test_advisory_band_budget_violation() -> None:
    s = _scenario()
    for item in s["timing_series"]:
        item["latency_ms"] = 100.0
    for item in s["throughput_series"]:
        item["throughput_units"] = 1.0
        item["backlog_units"] = 100.0
    ledger = run_latency_throughput_budget_ledger(**build_latency_throughput_scenario(s))
    assert ledger.advisory_state == "budget_violation"


def test_replay_comparison_stability() -> None:
    scenario = build_latency_throughput_scenario(_scenario())
    a = run_latency_throughput_budget_ledger(**scenario)
    b = run_latency_throughput_budget_ledger(**scenario)
    cmp = compare_budget_replay(a, b)
    assert cmp["replay_stable"] is True


def test_no_input_mutation() -> None:
    payload = _scenario()
    before = copy.deepcopy(payload)
    build_latency_throughput_scenario(payload)
    assert payload == before


def test_decoder_untouched_confirmation() -> None:
    import qec.orchestration.latency_throughput_budget_ledger as mod

    with open(mod.__file__, "r", encoding="utf-8") as handle:
        source = handle.read()
    assert "qec.decoder" not in source


def test_budget_receipt_determinism() -> None:
    ledger = run_latency_throughput_budget_ledger(**build_latency_throughput_scenario(_scenario()))
    a = build_budget_receipt(ledger)
    b = build_budget_receipt(ledger)
    assert a == b


def test_summary_content() -> None:
    ledger = run_latency_throughput_budget_ledger(**build_latency_throughput_scenario(_scenario()))
    summary = summarize_latency_throughput_budget(ledger)
    assert summary["advisory_state"] in ADVISORY_STATES
    assert summary["metrics"]


def test_malformed_float_handling_nan_inf_sentinels() -> None:
    scenario = build_latency_throughput_scenario(
        {
            "timing_series": [{"sample_id": "a", "latency_ms": math.nan}],
            "throughput_series": [{"sample_id": "b", "throughput_units": math.inf, "backlog_units": -math.inf}],
            "budget_requirements": {},
        }
    )
    notes = scenario["normalization_notes"]
    assert any(note.endswith(":nan") for note in notes)
    assert any(note.endswith(":pos_inf") for note in notes)
    assert any(note.endswith(":neg_inf") for note in notes)


def test_logical_valid_timing_violated_scenario() -> None:
    s = _scenario()
    for item in s["timing_series"]:
        item["latency_ms"] = 100.0
    for item in s["throughput_series"]:
        item["throughput_units"] = 0.1
        item["backlog_units"] = 100.0
    ledger = run_latency_throughput_budget_ledger(**build_latency_throughput_scenario(s))
    assert ledger.budget_receipt.logical_outputs_valid is True
    assert ledger.budget_receipt.timing_budget_exceeded is True


def test_empty_sample_handling() -> None:
    ledger = run_latency_throughput_budget_ledger()
    assert ledger.timing_series == ()
    assert ledger.throughput_series == ()


def test_deterministic_ordering_of_samples() -> None:
    scenario = build_latency_throughput_scenario(_scenario())
    assert tuple(sample.sample_id for sample in scenario["timing_series"]) == ("t0", "t1")
    assert tuple(sample.sample_id for sample in scenario["throughput_series"]) == ("p0", "p1")


def test_dataclass_samples_are_re_normalized_before_hashing() -> None:
    scenario = build_latency_throughput_scenario(_scenario())
    bad_sample = LatencyThroughputSample(
        sample_index=0,
        sample_id="nan-sample",
        latency_ms=math.nan,
        throughput_units=10.0,
        backlog_units=0.0,
    )
    ledger = run_latency_throughput_budget_ledger(
        timing_series=(bad_sample,),
        throughput_series=scenario["throughput_series"],
        budget_requirements=scenario["budget_requirements"],
    )
    assert ledger.timing_series[0].latency_ms == 0.0


def test_validator_preserves_integrity_fields_for_mapping_inputs() -> None:
    ledger = run_latency_throughput_budget_ledger(**build_latency_throughput_scenario(_scenario()))
    tampered = ledger.to_dict()
    tampered["ledger_hash"] = "00" * 32
    tampered["budget_receipt"]["receipt_hash"] = "11" * 32

    report = validate_latency_throughput_budget_ledger(tampered)
    assert report["is_valid"] is False
    assert "ledger hash drift" in report["violations"]
    assert "receipt hash drift" in report["violations"]


def test_compare_budget_replay_detects_content_differences() -> None:
    ledger = run_latency_throughput_budget_ledger(**build_latency_throughput_scenario(_scenario()))
    tampered = ledger.to_dict()
    tampered["timing_series"][0]["latency_ms"] = 99.0  # change actual content

    cmp = compare_budget_replay(ledger, tampered)
    assert cmp["replay_stable"] is False
    assert "ledger hash mismatch" in cmp["violations"]
