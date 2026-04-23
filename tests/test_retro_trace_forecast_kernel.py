from __future__ import annotations

import pytest

from qec.analysis.retro_target_registry import build_retro_target
from qec.analysis.retro_trace_forecast_kernel import forecast_retro_trace
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
