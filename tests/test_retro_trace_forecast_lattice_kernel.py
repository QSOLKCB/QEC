from __future__ import annotations

import pytest

from qec.analysis.canonical_hashing import sha256_hex
from qec.analysis.retro_target_registry import build_retro_target
from qec.analysis.retro_trace_forecast_lattice_kernel import (
    LatticeCoordinate,
    RetroTraceForecastLatticeReceipt,
    RetroTraceLatticeCell,
    RetroTraceLatticeForecastStep,
    forecast_retro_trace_lattice,
)
from qec.analysis.retro_trace_intake_bridge import build_retro_trace


def _target_receipt():
    return build_retro_target(
        target_id="lattice-target",
        isa_family="z80",
        word_size=8,
        address_width=16,
        ram_budget=64 * 1024,
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
        timing_trace=tuple({"cycle": 120 + 40 * idx} for idx in range(8)),
        display_trace=({"scanline": 0, "event": "start"},),
        audio_trace=({"channel": 1, "pattern": "pulse"},),
        input_trace=tuple({"port": 1, "button": "A", "state": idx % 2} for idx in range(8)),
        metadata={"emulator": "retroarch", "rom_hash": "lattice-a", "version": "1.0.0"},
    )


def _sparse_trace():
    return build_retro_trace(
        target_receipt=_target_receipt(),
        cpu_trace=tuple(),
        memory_trace=tuple(),
        timing_trace=tuple(),
        display_trace=tuple(),
        audio_trace=tuple(),
        input_trace=tuple(),
        metadata={"emulator": "retroarch", "rom_hash": "empty", "version": "1.0.0"},
    )


def test_deterministic_replay_identical_hash_and_bytes() -> None:
    trace = _stable_trace()
    left = forecast_retro_trace_lattice(trace, horizon=12, lattice_mode="sierpinski_3")
    right = forecast_retro_trace_lattice(trace, horizon=12, lattice_mode="sierpinski_3")
    assert left.to_canonical_json() == right.to_canonical_json()
    assert left.to_canonical_bytes() == right.to_canonical_bytes()
    assert left.stable_hash() == right.stable_hash()


@pytest.mark.parametrize("mode", ["sierpinski_3", "neutral_atom_5", "rubik_8"])
def test_coordinate_bounds_per_mode(mode: str) -> None:
    max_bound = {"sierpinski_3": 2, "neutral_atom_5": 4, "rubik_8": 7}[mode]
    receipt = forecast_retro_trace_lattice(_stable_trace(), horizon=10, lattice_mode=mode)
    for step in receipt.series.steps:
        for cell in step.occupied_cells:
            assert 0 <= cell.coordinate.x <= max_bound
            assert 0 <= cell.coordinate.y <= max_bound
            assert 0 <= cell.coordinate.z <= max_bound


def test_invalid_lattice_mode_rejected() -> None:
    with pytest.raises(ValueError, match="lattice_mode must be one of"):
        forecast_retro_trace_lattice(_stable_trace(), horizon=6, lattice_mode="hex_9")


def test_monotonic_step_progression() -> None:
    receipt = forecast_retro_trace_lattice(_stable_trace(), horizon=24, lattice_mode="rubik_8")
    counts = [step.projected_occupancy_count for step in receipt.series.steps]
    assert counts == sorted(counts)


@pytest.mark.parametrize("mode", ["sierpinski_3", "neutral_atom_5", "rubik_8"])
def test_classification_and_bounds(mode: str) -> None:
    receipt = forecast_retro_trace_lattice(_stable_trace(), horizon=18, lattice_mode=mode)
    assert receipt.summary.collapse_risk_classification in {"STABLE", "DRIFT", "UNSTABLE"}
    assert 0.0 <= receipt.summary.occupancy_dispersion <= 1.0
    assert 0.0 <= receipt.summary.locality_risk <= 1.0
    assert 0.0 <= receipt.summary.overall_stability_forecast <= 1.0


def test_sparse_trace_edge_case() -> None:
    receipt = forecast_retro_trace_lattice(_sparse_trace(), horizon=8, lattice_mode="neutral_atom_5")
    assert len(receipt.series.steps) == 8
    assert all(step.projected_occupancy_count >= 1 for step in receipt.series.steps)
    for step in receipt.series.steps:
        assert sum(cell.occupancy_share for cell in step.occupied_cells) == pytest.approx(1.0)


def test_timing_features_change_on_timing_events_not_global_event_index() -> None:
    trace = build_retro_trace(
        target_receipt=_target_receipt(),
        cpu_trace=(
            {"pc": 0x1000, "a": 0x10},
            {"pc": 0x1001, "a": 0x11},
            {"pc": 0x1002, "a": 0x12},
        ),
        memory_trace=tuple(),
        timing_trace=({"cycle": 10}, {"cycle": 1000}, {"cycle": 1200}),
        display_trace=tuple(),
        audio_trace=tuple(),
        input_trace=tuple(),
        metadata={"emulator": "retroarch", "rom_hash": "timing-align", "version": "1.0.0"},
    )
    receipt = forecast_retro_trace_lattice(trace, horizon=1, lattice_mode="neutral_atom_5")
    first_step_cells = receipt.series.steps[0].occupied_cells
    observed_x = {cell.coordinate.x for cell in first_step_cells}
    assert len(observed_x) > 1


def test_equivalent_trace_inputs_produce_identical_receipts() -> None:
    baseline = _stable_trace()
    equivalent = build_retro_trace(
        target_receipt=_target_receipt(),
        cpu_trace=tuple(reversed(tuple({"pc": 0x1000 + idx, "a": 0x10 + (idx % 16)} for idx in range(8)))),
        memory_trace=({"value": 0xAB, "op": "read", "address": 0x4000},),
        timing_trace=tuple(reversed(tuple({"cycle": 120 + 40 * idx} for idx in range(8)))),
        display_trace=({"event": "start", "scanline": 0},),
        audio_trace=({"pattern": "pulse", "channel": 1},),
        input_trace=tuple(reversed(tuple({"state": idx % 2, "button": "A", "port": 1} for idx in range(8)))),
        metadata={"version": "1.0.0", "rom_hash": "lattice-a", "emulator": "retroarch"},
    )
    left = forecast_retro_trace_lattice(baseline, horizon=14, lattice_mode="neutral_atom_5")
    right = forecast_retro_trace_lattice(equivalent, horizon=14, lattice_mode="neutral_atom_5")
    assert left.to_canonical_bytes() == right.to_canonical_bytes()
    assert left.stable_hash() == right.stable_hash()


def test_duplicate_coordinate_rejected() -> None:
    coordinate = LatticeCoordinate(x=0, y=0, z=0)
    cell_payload = {
        "lattice_mode": "sierpinski_3",
        "coordinate": coordinate.to_dict(),
        "occupancy_share": 1.0,
        "locality_pressure": 0.0,
    }
    cell = RetroTraceLatticeCell(
        lattice_mode="sierpinski_3",
        coordinate=coordinate,
        occupancy_share=1.0,
        locality_pressure=0.0,
        _stable_hash=sha256_hex(cell_payload),
    )
    with pytest.raises(ValueError, match="duplicate occupied coordinates within step"):
        RetroTraceLatticeForecastStep(
            step_index=1,
            projected_occupancy_count=2,
            projected_density_score=0.2,
            projected_locality_pressure=0.1,
            projected_dispersion_score=0.1,
            stability_score=0.8,
            occupied_cells=(cell, cell),
            _stable_hash="a" * 64,
        )


def test_non_canonical_cell_order_rejected() -> None:
    c0 = LatticeCoordinate(x=0, y=0, z=1)
    c1 = LatticeCoordinate(x=0, y=0, z=0)
    p0 = {"lattice_mode": "sierpinski_3", "coordinate": c0.to_dict(), "occupancy_share": 0.5, "locality_pressure": 0.0}
    p1 = {"lattice_mode": "sierpinski_3", "coordinate": c1.to_dict(), "occupancy_share": 0.5, "locality_pressure": 0.0}
    cell0 = RetroTraceLatticeCell("sierpinski_3", c0, 0.5, 0.0, sha256_hex(p0))
    cell1 = RetroTraceLatticeCell("sierpinski_3", c1, 0.5, 0.0, sha256_hex(p1))
    with pytest.raises(ValueError, match="occupied_cells must use canonical ordering"):
        RetroTraceLatticeForecastStep(
            step_index=1,
            projected_occupancy_count=2,
            projected_density_score=0.2,
            projected_locality_pressure=0.0,
            projected_dispersion_score=0.1,
            stability_score=0.9,
            occupied_cells=(cell0, cell1),
            _stable_hash="a" * 64,
        )


@pytest.mark.parametrize("invalid_horizon", [True, False])
def test_rejects_bool_horizon(invalid_horizon: bool) -> None:
    with pytest.raises(ValueError, match=r"horizon must be int in \[1,256\]"):
        forecast_retro_trace_lattice(_stable_trace(), horizon=invalid_horizon, lattice_mode="rubik_8")


@pytest.mark.parametrize("field,value", [("x", True), ("y", False), ("z", True)])
def test_coordinate_rejects_bool_fields(field: str, value: bool) -> None:
    kwargs = {"x": 0, "y": 0, "z": 0}
    kwargs[field] = value
    with pytest.raises(ValueError, match=f"{field} must be int"):
        LatticeCoordinate(**kwargs)


@pytest.mark.parametrize("invalid_step", [True, False])
def test_step_rejects_bool_step_index(invalid_step: bool) -> None:
    with pytest.raises(ValueError, match="step_index must be positive int"):
        RetroTraceLatticeForecastStep(
            step_index=invalid_step,
            projected_occupancy_count=0,
            projected_density_score=0.0,
            projected_locality_pressure=0.0,
            projected_dispersion_score=0.0,
            stability_score=0.0,
            occupied_cells=tuple(),
            _stable_hash="a" * 64,
        )


def test_receipt_reconstruction_rejects_summary_mismatch() -> None:
    receipt = forecast_retro_trace_lattice(_stable_trace(), horizon=6, lattice_mode="sierpinski_3")
    with pytest.raises(ValueError, match="summary values mismatch recomputed canonical series values"):
        RetroTraceForecastLatticeReceipt(
            retro_trace_hash=receipt.retro_trace_hash,
            series=receipt.series,
            summary=receipt.summary.__class__(
                lattice_mode=receipt.summary.lattice_mode,
                dominant_region=receipt.summary.dominant_region,
                occupancy_dispersion=receipt.summary.occupancy_dispersion,
                locality_risk=receipt.summary.locality_risk,
                overall_stability_forecast=0.0,
                collapse_risk_classification="UNSTABLE",
                _stable_hash=sha256_hex(
                    {
                        "lattice_mode": receipt.summary.lattice_mode,
                        "dominant_region": receipt.summary.dominant_region,
                        "occupancy_dispersion": receipt.summary.occupancy_dispersion,
                        "locality_risk": receipt.summary.locality_risk,
                        "overall_stability_forecast": 0.0,
                        "collapse_risk_classification": "UNSTABLE",
                    }
                ),
            ),
            _stable_hash=receipt.stable_hash(),
        )


def test_snapshot_fixture_for_deterministic_forecast() -> None:
    receipt = forecast_retro_trace_lattice(_stable_trace(), horizon=4, lattice_mode="sierpinski_3")
    expected_hash = "955679df4ce94d86edc628a92486f2496a6185d349f980c89f88ca98be4a3afc"
    assert receipt.to_canonical_json().startswith('{"retro_trace_hash":"74ba968ac06131fceb51bc2b41bc02d51c8cec5e918adf133e7fe04d98ec9739"')
    assert receipt.stable_hash() == expected_hash


def test_replay_certification_many_rebuilds() -> None:
    trace = _stable_trace()
    receipts = [forecast_retro_trace_lattice(trace, horizon=10, lattice_mode="neutral_atom_5") for _ in range(75)]
    assert len({item.stable_hash() for item in receipts}) == 1
    assert len({item.to_canonical_json() for item in receipts}) == 1
