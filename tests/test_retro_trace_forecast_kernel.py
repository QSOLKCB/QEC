from __future__ import annotations

import pytest

from qec.analysis.canonical_hashing import sha256_hex
from qec.analysis.closed_loop_simulation_kernel import round12
from qec.analysis.retro_target_registry import build_retro_target
from qec.analysis.retro_trace_forecast_kernel import (
    RetroTraceForecastReceipt,
    RetroTraceForecastStep,
    RetroTraceForecastSummary,
    forecast_retro_trace,
)
from qec.analysis.retro_trace_intake_bridge import build_retro_trace


def _target_receipt():
    return build_retro_target(
        target_id="forecast-target",
        isa_family="z80",
        word_size=8,
        address_width=16,
        ram_budget=48 * 1024,
        rom_budget=32 * 1024,
        cycle_budget=3_500_000,
        display_budget={"width": 256, "height": 192, "colors": 16},
        audio_budget={"channels": 3, "sample_rate": 44_100},
        input_budget={"buttons": 2, "axes": 0},
        fpu_policy="none",
        provenance="hardware",
    )


def _stable_trace():
    return build_retro_trace(
        target_receipt=_target_receipt(),
        cpu_trace=tuple({"pc": 0x1000 + idx, "a": 0x10 + (idx % 16)} for idx in range(8)),
        memory_trace=({"address": 0x4000, "op": "read", "value": 0xAB},),
        timing_trace=tuple({"cycle": 100 + 20 * idx} for idx in range(8)),
        display_trace=({"scanline": 0, "event": "start"},),
        audio_trace=({"channel": 1, "pattern": "pulse"},),
        input_trace=tuple({"port": 1, "button": "A", "state": idx % 2} for idx in range(8)),
        metadata={"emulator": "retroarch", "rom_hash": "abc123", "version": "1.0.0"},
    )


def _unstable_trace():
    return build_retro_trace(
        target_receipt=_target_receipt(),
        cpu_trace=tuple({"pc": 0x2000 + idx, "a": idx % 256} for idx in range(24)),
        memory_trace=tuple(),
        timing_trace=({"cycle": 10}, {"cycle": 2500}),
        display_trace=tuple(),
        audio_trace=tuple(),
        input_trace=tuple(),
        metadata={"emulator": "retroarch", "rom_hash": "deadbeef", "version": "1.0.0"},
    )


def _trace_with_timing(cycles: tuple[int, ...]):
    return build_retro_trace(
        target_receipt=_target_receipt(),
        cpu_trace=tuple({"pc": 0x3000 + idx, "a": 0x20 + (idx % 8)} for idx in range(max(1, len(cycles)))),
        memory_trace=tuple(),
        timing_trace=tuple({"cycle": cycle} for cycle in cycles),
        display_trace=tuple(),
        audio_trace=tuple(),
        input_trace=tuple(),
        metadata={"emulator": "retroarch", "rom_hash": "timing", "version": "1.0.0"},
    )


def test_deterministic_replay_identical_hash() -> None:
    retro_trace = _stable_trace()
    a = forecast_retro_trace(retro_trace, horizon=12)
    b = forecast_retro_trace(retro_trace, horizon=12)
    assert a.stable_hash() == b.stable_hash()
    assert a.to_canonical_bytes() == b.to_canonical_bytes()


def test_identical_input_identical_forecast() -> None:
    trace_a = _stable_trace()
    trace_b = _stable_trace()
    left = forecast_retro_trace(trace_a, horizon=16)
    right = forecast_retro_trace(trace_b, horizon=16)
    assert left.to_canonical_json() == right.to_canonical_json()


def test_horizon_bounds_enforcement() -> None:
    with pytest.raises(ValueError, match=r"horizon must be int in \[1,256\]"):
        forecast_retro_trace(_stable_trace(), horizon=0)
    with pytest.raises(ValueError, match=r"horizon must be int in \[1,256\]"):
        forecast_retro_trace(_stable_trace(), horizon=257)


def test_monotonic_timing_projection() -> None:
    receipt = forecast_retro_trace(_stable_trace(), horizon=20)
    projected = [step.projected_timing for step in receipt.series.steps]
    assert projected == sorted(projected)
    assert all(projected[idx] > projected[idx - 1] for idx in range(1, len(projected)))


def test_classification_correctness() -> None:
    stable = forecast_retro_trace(_stable_trace(), horizon=10)
    unstable = forecast_retro_trace(_unstable_trace(), horizon=10)
    assert stable.summary.collapse_risk_classification == "STABLE"
    assert unstable.summary.collapse_risk_classification in {"DRIFT", "UNSTABLE"}
    assert unstable.summary.overall_stability_forecast < stable.summary.overall_stability_forecast


def test_forecast_snapshot_fixture_is_stable() -> None:
    receipt = forecast_retro_trace(_stable_trace(), horizon=12)
    expected_json = (
        '{"retro_trace_hash":"c9ff9f5fb8c2e84136c6ff9e14465573231a84fb658a8afd1e5f8c77cc5d6208","series":{"features":'
        '[["average_timing_delta",20.0],["event_rate",0.112033195021],["gradient_stability",1.0],["last_timing",240.0],'
        '["normalized_timing_gradient",0.142857142857],["ordering_integrity",1.0],["sparsity",0.111111111111],'
        '["timing_density",0.296296296296]],"horizon":12,"stable_hash":"950b77d0511d80d887934ba019b2963c430f81aea4528b04665ff25af16ae439",'
        '"steps":[{"projected_event_density":0.363018287997,"projected_timing":262,"stability_score":0.749035500231,'
        '"stable_hash":"9d988eddc6491e5a484852ce18ab2c74b47596e9efb94265d01e21f0eae03845","step_index":1},'
        '{"projected_event_density":0.429740279698,"projected_timing":285,"stability_score":0.744043798986,'
        '"stable_hash":"021faf725e775f40d50ef5d99779d277a2ef0371dc089ab50c84c63a608c3ac6","step_index":2},'
        '{"projected_event_density":0.496462271399,"projected_timing":309,"stability_score":0.739052097741,'
        '"stable_hash":"88e597b3b619c266282c88e282dd673f462f2fcee0b582e9a21ad9aa95d32c2d","step_index":3},'
        '{"projected_event_density":0.5631842631,"projected_timing":334,"stability_score":0.734060396496,'
        '"stable_hash":"e80fff29fab85d17f8201d57b59739ab791a9034f1ccb58a59f1ccab18d5f006","step_index":4},'
        '{"projected_event_density":0.629906254801,"projected_timing":360,"stability_score":0.729068695251,'
        '"stable_hash":"cb8f7b27fc3741ccf7634162ee602d0850658844a2159b96e8d4aa49e6b390ea","step_index":5},'
        '{"projected_event_density":0.696628246502,"projected_timing":387,"stability_score":0.715180728447,'
        '"stable_hash":"0c247bd187b1dbe4ff9ae84eac34f1d4343cf64773a36733a811de310fa7de09","step_index":6},'
        '{"projected_event_density":0.763350238203,"projected_timing":415,"stability_score":0.690172429692,'
        '"stable_hash":"b12771153412d678144d8188643272d31ad341b052438dd065bbe6243e4f8660","step_index":7},'
        '{"projected_event_density":0.830072229904,"projected_timing":444,"stability_score":0.665164130936,'
        '"stable_hash":"fef5f4ae71fe636a74d99b7cb17e38000f23d0250627996ed9da379d2fe599be","step_index":8},'
        '{"projected_event_density":0.896794221605,"projected_timing":474,"stability_score":0.640155832181,'
        '"stable_hash":"effe7cc60286682a3a703057608a35591654c639cb67f62459cc03bd5faea262","step_index":9},'
        '{"projected_event_density":0.963516213306,"projected_timing":505,"stability_score":0.615147533426,'
        '"stable_hash":"6ab6da0cb8e0c300d06efebd9397051d7832d41fa9052fe511b40a7001807cbc","step_index":10},'
        '{"projected_event_density":1.0,"projected_timing":537,"stability_score":0.594674965422,'
        '"stable_hash":"0c0736527ba8571d1ce31745cd868826ecbd8a8df49b959151e3700626c5e704","step_index":11},'
        '{"projected_event_density":1.0,"projected_timing":570,"stability_score":0.579674965422,'
        '"stable_hash":"6ce93b9c65a9e6d1afa964713e993666c260672ba19b3b9ec0bc4932e089103d","step_index":12}]},'
        '"stable_hash":"74c20945f214c4e5e564f945f4c29e4b9f965801abf4e846bfbc6e4ba05b7187","summary":'
        '{"collapse_risk_classification":"STABLE","overall_stability_forecast":0.682952589519,'
        '"stable_hash":"537db45c751e17829991631f07edada6a9532001e98171a8c3b2a356e39e302c"}}'
    )
    expected_hash = "74c20945f214c4e5e564f945f4c29e4b9f965801abf4e846bfbc6e4ba05b7187"
    assert receipt.to_canonical_json() == expected_json
    assert receipt.stable_hash() == expected_hash


def test_forecast_replay_certification_multi_run() -> None:
    retro_trace = _stable_trace()
    receipts = [forecast_retro_trace(retro_trace, horizon=20) for _ in range(75)]
    assert len({item.stable_hash() for item in receipts}) == 1
    assert len({item.to_canonical_json() for item in receipts}) == 1
    assert len({item.to_canonical_bytes() for item in receipts}) == 1


def test_forecast_handles_empty_timing_trace() -> None:
    trace = _trace_with_timing(())
    receipt = forecast_retro_trace(trace, horizon=8)
    replay = forecast_retro_trace(trace, horizon=8)
    projected = [step.projected_timing for step in receipt.series.steps]
    assert projected == sorted(projected)
    assert receipt.summary.collapse_risk_classification == replay.summary.collapse_risk_classification
    assert all(0.0 <= step.stability_score <= 1.0 for step in receipt.series.steps)


def test_forecast_handles_single_timing_event() -> None:
    receipt = forecast_retro_trace(_trace_with_timing((101,)), horizon=8)
    projected = [step.projected_timing for step in receipt.series.steps]
    assert projected == sorted(projected)
    assert all(projected[idx] > projected[idx - 1] for idx in range(1, len(projected)))
    assert receipt.summary.collapse_risk_classification in {"STABLE", "DRIFT", "UNSTABLE"}


def test_forecast_handles_repeated_identical_timing_events() -> None:
    receipt = forecast_retro_trace(_trace_with_timing((7, 7, 7, 7)), horizon=8)
    projected = [step.projected_timing for step in receipt.series.steps]
    assert projected == sorted(projected)
    assert all(projected[idx] > projected[idx - 1] for idx in range(1, len(projected)))


def test_forecast_horizon_boundary_values() -> None:
    one = forecast_retro_trace(_stable_trace(), horizon=1)
    maxed = forecast_retro_trace(_stable_trace(), horizon=256)
    assert len(one.series.steps) == 1
    assert len(maxed.series.steps) == 256


@pytest.mark.parametrize("invalid_horizon", [0, -1, 257, True, False])
def test_forecast_rejects_invalid_horizon_values(invalid_horizon: int) -> None:
    with pytest.raises(ValueError, match=r"horizon must be int in \[1,256\]"):
        forecast_retro_trace(_stable_trace(), horizon=invalid_horizon)


def test_forecast_near_boundary_float_stability() -> None:
    near_a = _trace_with_timing((100, 101, 102, 103, 104))
    near_b = _trace_with_timing((100, 101, 102, 103, 105))
    a = forecast_retro_trace(near_a, horizon=16)
    b = forecast_retro_trace(near_b, horizon=16)
    c = forecast_retro_trace(near_a, horizon=16)

    assert a.to_canonical_json() == c.to_canonical_json()
    assert a.stable_hash() == c.stable_hash()
    assert a.summary.collapse_risk_classification == c.summary.collapse_risk_classification
    assert b.summary.collapse_risk_classification in {"STABLE", "DRIFT", "UNSTABLE"}


def test_forecast_equivalent_inputs_produce_identical_hash() -> None:
    baseline = _stable_trace()
    equivalent = build_retro_trace(
        target_receipt=_target_receipt(),
        cpu_trace=tuple(reversed(tuple({"pc": 0x1000 + idx, "a": 0x10 + (idx % 16)} for idx in range(8)))),
        memory_trace=({"value": 0xAB, "op": "read", "address": 0x4000},),
        timing_trace=tuple(reversed(tuple({"cycle": 100 + 20 * idx} for idx in range(8)))),
        display_trace=({"event": "start", "scanline": 0},),
        audio_trace=({"pattern": "pulse", "channel": 1},),
        input_trace=tuple(reversed(tuple({"state": idx % 2, "button": "A", "port": 1} for idx in range(8)))),
        metadata={"version": "1.0.0", "rom_hash": "abc123", "emulator": "retroarch"},
    )
    left = forecast_retro_trace(baseline, horizon=24)
    right = forecast_retro_trace(equivalent, horizon=24)
    assert left.to_canonical_bytes() == right.to_canonical_bytes()
    assert left.stable_hash() == right.stable_hash()


def test_direct_receipt_reconstruction_rejects_non_canonical_values() -> None:
    receipt = forecast_retro_trace(_stable_trace(), horizon=4)
    step = RetroTraceForecastStep(
        step_index=1,
        projected_timing=10,
        projected_event_density=0.5000000000004,
        stability_score=0.6800000000004,
        _stable_hash=sha256_hex(
            {
                "step_index": 1,
                "projected_timing": 10,
                "projected_event_density": 0.5,
                "stability_score": 0.68,
            }
        ),
    )
    assert step.projected_event_density == 0.5
    assert step.stability_score == 0.68

    with pytest.raises(ValueError, match="stable_hash mismatch"):
        RetroTraceForecastStep(
            step_index=1,
            projected_timing=10,
            projected_event_density=0.5000000000004,
            stability_score=0.6800000000004,
            _stable_hash=sha256_hex(
                {
                    "step_index": 1,
                    "projected_timing": 10,
                    "projected_event_density": 0.5000000000004,
                    "stability_score": 0.6800000000004,
                }
            ),
        )

    with pytest.raises(ValueError, match="summary overall_stability_forecast mismatch"):
        RetroTraceForecastReceipt(
            retro_trace_hash=receipt.retro_trace_hash,
            series=receipt.series,
            summary=RetroTraceForecastSummary(
                overall_stability_forecast=receipt.summary.overall_stability_forecast + 1e-8,
                collapse_risk_classification=receipt.summary.collapse_risk_classification,
                _stable_hash=sha256_hex(
                    {
                        "overall_stability_forecast": round12(receipt.summary.overall_stability_forecast + 1e-8),
                        "collapse_risk_classification": receipt.summary.collapse_risk_classification,
                    }
                ),
            ),
            _stable_hash=receipt.stable_hash(),
        )
