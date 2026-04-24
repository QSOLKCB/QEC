"""v147.3.2 — Geometry-Aware Retro Trace Forecast Lattice Kernel."""

from __future__ import annotations

from dataclasses import dataclass

from qec.analysis.canonical_hashing import canonical_bytes, canonical_json, sha256_hex
from qec.analysis.closed_loop_simulation_kernel import round12, validate_sha256_hex, validate_unit_interval
from qec.analysis.retro_trace_intake_bridge import RetroTraceReceipt

_JSONScalar = str | int | float | bool | None
_JSONValue = _JSONScalar | tuple["_JSONValue", ...] | dict[str, "_JSONValue"]

_MIN_HORIZON = 1
_MAX_HORIZON = 256
_SUPPORTED_MODES = ("sierpinski_3", "neutral_atom_5", "rubik_8")
_CLASS_STABLE = "STABLE"
_CLASS_DRIFT = "DRIFT"
_CLASS_UNSTABLE = "UNSTABLE"
_EVENT_TYPES = ("cpu", "memory", "timing", "display", "audio", "input")


def _validate_lattice_mode(mode: str) -> str:
    if not isinstance(mode, str) or mode not in _SUPPORTED_MODES:
        raise ValueError("lattice_mode must be one of sierpinski_3|neutral_atom_5|rubik_8")
    return mode


def _lattice_size(mode: str) -> int:
    if mode == "sierpinski_3":
        return 3
    if mode == "neutral_atom_5":
        return 5
    return 8


def _clamp01(value: float) -> float:
    if value <= 0.0:
        return 0.0
    if value >= 1.0:
        return 1.0
    return round12(value)


def _norm_timing_vector(retro_trace: RetroTraceReceipt) -> tuple[float, ...]:
    if retro_trace.trace_length == 0:
        return tuple()
    if not retro_trace.normalized_timing:
        return tuple(0.0 for _ in range(retro_trace.trace_length))
    max_cycle = float(max(1, retro_trace.normalized_timing[-1]))
    values: list[float] = []
    timing_index = 0
    latest_cycle = retro_trace.normalized_timing[0]
    for _event_index, event_type, _payload in retro_trace.event_sequence:
        if event_type == "timing" and timing_index < len(retro_trace.normalized_timing):
            latest_cycle = retro_trace.normalized_timing[timing_index]
            timing_index += 1
        values.append(_clamp01(float(latest_cycle) / max_cycle))
    return tuple(values)


def _type_code(event_type: str) -> int:
    return _EVENT_TYPES.index(event_type)


def _embed_coordinate(
    *,
    mode: str,
    size: int,
    event_index: int,
    event_type: str,
    normalized_timing: float,
    trace_length: int,
) -> tuple[int, int, int]:
    tcode = _type_code(event_type)
    tslot = int(round(normalized_timing * float(max(1, size - 1))))
    if mode == "sierpinski_3":
        x = (event_index + tcode + (trace_length % size)) % size
        y = ((event_index // size) + (tslot * 2) + tcode) % size
        z = ((event_index // (size * size)) + tslot + (2 * tcode)) % size
        return (x, y, z)
    if mode == "neutral_atom_5":
        center = size // 2
        radial = ((event_index + tslot + tcode + trace_length) % 3) - 1
        x = max(0, min(size - 1, center + radial))
        y = (center + ((event_index + tcode) % 5) - 2) % size
        z = (center + ((tslot + (event_index // 2)) % 5) - 2) % size
        return (x, y, z)
    block = (event_index + tcode + trace_length) % size
    face = ((event_index // size) + tslot + (tcode * 3)) % size
    slice_id = ((event_index // (size // 2)) + (2 * tslot) + tcode) % size
    return (block, face, slice_id)


def _neighbor_ratio(coords: tuple[int, int, int], occupied: dict[tuple[int, int, int], int]) -> float:
    x, y, z = coords
    neighbors = (
        (x - 1, y, z),
        (x + 1, y, z),
        (x, y - 1, z),
        (x, y + 1, z),
        (x, y, z - 1),
        (x, y, z + 1),
    )
    count = sum(1 for item in neighbors if item in occupied)
    return _clamp01(float(count) / 6.0)


def _dispersion(coords: tuple[tuple[int, int, int], ...], size: int) -> float:
    if len(coords) <= 1:
        return 0.0
    total = 0.0
    pairs = 0
    max_dist = float(max(1, 3 * (size - 1)))
    for idx in range(len(coords)):
        ax, ay, az = coords[idx]
        for jdx in range(idx + 1, len(coords)):
            bx, by, bz = coords[jdx]
            total += float(abs(ax - bx) + abs(ay - by) + abs(az - bz)) / max_dist
            pairs += 1
    return _clamp01(total / float(max(1, pairs)))


def _classify(stability: float) -> str:
    if stability >= 0.68:
        return _CLASS_STABLE
    if stability >= 0.50:
        return _CLASS_DRIFT
    return _CLASS_UNSTABLE


@dataclass(frozen=True)
class LatticeCoordinate:
    x: int
    y: int
    z: int

    def __post_init__(self) -> None:
        for field_name in ("x", "y", "z"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{field_name} must be int")
            if value < 0:
                raise ValueError(f"{field_name} must be non-negative")

    def to_dict(self) -> dict[str, _JSONValue]:
        return {"x": self.x, "y": self.y, "z": self.z}

    def to_canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return canonical_bytes(self.to_dict())

    def stable_hash(self) -> str:
        return sha256_hex(self.to_dict())


@dataclass(frozen=True)
class RetroTraceLatticeCell:
    lattice_mode: str
    coordinate: LatticeCoordinate
    occupancy_share: float
    locality_pressure: float
    _stable_hash: str

    def __post_init__(self) -> None:
        mode = _validate_lattice_mode(self.lattice_mode)
        object.__setattr__(self, "lattice_mode", mode)
        if not isinstance(self.coordinate, LatticeCoordinate):
            raise ValueError("coordinate must be LatticeCoordinate")
        bound = _lattice_size(mode) - 1
        if self.coordinate.x > bound or self.coordinate.y > bound or self.coordinate.z > bound:
            raise ValueError("coordinate out-of-bounds for lattice_mode")
        object.__setattr__(self, "occupancy_share", _clamp01(validate_unit_interval(self.occupancy_share, "occupancy_share")))
        object.__setattr__(
            self,
            "locality_pressure",
            _clamp01(validate_unit_interval(self.locality_pressure, "locality_pressure")),
        )
        if self._stable_hash != sha256_hex(self._payload_without_hash()):
            raise ValueError("stable_hash mismatch")

    def _payload_without_hash(self) -> dict[str, _JSONValue]:
        return {
            "lattice_mode": self.lattice_mode,
            "coordinate": self.coordinate.to_dict(),
            "occupancy_share": round12(self.occupancy_share),
            "locality_pressure": round12(self.locality_pressure),
        }

    def to_dict(self) -> dict[str, _JSONValue]:
        return {**self._payload_without_hash(), "stable_hash": self._stable_hash}

    def to_canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return canonical_bytes(self.to_dict())

    def stable_hash(self) -> str:
        return self._stable_hash


@dataclass(frozen=True)
class RetroTraceLatticeForecastStep:
    step_index: int
    projected_occupancy_count: int
    projected_density_score: float
    projected_locality_pressure: float
    projected_dispersion_score: float
    stability_score: float
    occupied_cells: tuple[RetroTraceLatticeCell, ...]
    _stable_hash: str

    def __post_init__(self) -> None:
        if isinstance(self.step_index, bool) or not isinstance(self.step_index, int) or self.step_index < 1:
            raise ValueError("step_index must be positive int")
        if (
            isinstance(self.projected_occupancy_count, bool)
            or not isinstance(self.projected_occupancy_count, int)
            or self.projected_occupancy_count < 0
        ):
            raise ValueError("projected_occupancy_count must be non-negative int")
        for name in (
            "projected_density_score",
            "projected_locality_pressure",
            "projected_dispersion_score",
            "stability_score",
        ):
            object.__setattr__(self, name, _clamp01(validate_unit_interval(getattr(self, name), name)))
        if not isinstance(self.occupied_cells, tuple):
            raise ValueError("occupied_cells must be tuple")
        if any(not isinstance(cell, RetroTraceLatticeCell) for cell in self.occupied_cells):
            raise ValueError("occupied_cells must contain RetroTraceLatticeCell")
        canonical = tuple(
            sorted(
                self.occupied_cells,
                key=lambda cell: (cell.coordinate.x, cell.coordinate.y, cell.coordinate.z),
            )
        )
        if self.occupied_cells != canonical:
            raise ValueError("occupied_cells must use canonical ordering")
        seen = set()
        for cell in self.occupied_cells:
            key = (cell.coordinate.x, cell.coordinate.y, cell.coordinate.z)
            if key in seen:
                raise ValueError("duplicate occupied coordinates within step")
            seen.add(key)
        if self.projected_occupancy_count != len(self.occupied_cells):
            raise ValueError("projected_occupancy_count must match occupied_cells length")
        if self._stable_hash != sha256_hex(self._payload_without_hash()):
            raise ValueError("stable_hash mismatch")

    def _payload_without_hash(self) -> dict[str, _JSONValue]:
        return {
            "step_index": self.step_index,
            "projected_occupancy_count": self.projected_occupancy_count,
            "projected_density_score": round12(self.projected_density_score),
            "projected_locality_pressure": round12(self.projected_locality_pressure),
            "projected_dispersion_score": round12(self.projected_dispersion_score),
            "stability_score": round12(self.stability_score),
            "occupied_cells": tuple(cell.to_dict() for cell in self.occupied_cells),
        }

    def to_dict(self) -> dict[str, _JSONValue]:
        return {**self._payload_without_hash(), "stable_hash": self._stable_hash}

    def to_canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return canonical_bytes(self.to_dict())

    def stable_hash(self) -> str:
        return self._stable_hash


@dataclass(frozen=True)
class RetroTraceLatticeForecastSeries:
    lattice_mode: str
    horizon: int
    steps: tuple[RetroTraceLatticeForecastStep, ...]
    _stable_hash: str

    def __post_init__(self) -> None:
        mode = _validate_lattice_mode(self.lattice_mode)
        object.__setattr__(self, "lattice_mode", mode)
        if isinstance(self.horizon, bool) or not isinstance(self.horizon, int) or not (_MIN_HORIZON <= self.horizon <= _MAX_HORIZON):
            raise ValueError(f"horizon must be int in [{_MIN_HORIZON},{_MAX_HORIZON}]")
        if not isinstance(self.steps, tuple) or len(self.steps) != self.horizon:
            raise ValueError("steps length must equal horizon")
        if any(not isinstance(step, RetroTraceLatticeForecastStep) for step in self.steps):
            raise ValueError("steps contains invalid item")
        for idx, step in enumerate(self.steps, start=1):
            if step.step_index != idx:
                raise ValueError("steps must use contiguous canonical ordering")
            if idx > 1 and step.projected_occupancy_count < self.steps[idx - 2].projected_occupancy_count:
                raise ValueError("projected_occupancy_count must be monotonic")
        if self._stable_hash != sha256_hex(self._payload_without_hash()):
            raise ValueError("stable_hash mismatch")

    def _payload_without_hash(self) -> dict[str, _JSONValue]:
        return {
            "lattice_mode": self.lattice_mode,
            "horizon": self.horizon,
            "steps": tuple(step.to_dict() for step in self.steps),
        }

    def to_dict(self) -> dict[str, _JSONValue]:
        return {**self._payload_without_hash(), "stable_hash": self._stable_hash}

    def to_canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return canonical_bytes(self.to_dict())

    def stable_hash(self) -> str:
        return self._stable_hash


def _dominant_region_for_step(step: RetroTraceLatticeForecastStep, size: int) -> str:
    half = size / 2.0
    region_totals: dict[str, float] = {}
    for cell in step.occupied_cells:
        region = (
            ("L" if cell.coordinate.x < half else "H")
            + ("L" if cell.coordinate.y < half else "H")
            + ("L" if cell.coordinate.z < half else "H")
        )
        region_totals[region] = region_totals.get(region, 0.0) + cell.occupancy_share
    if not region_totals:
        return "LLL"
    return sorted(region_totals.items(), key=lambda item: (-item[1], item[0]))[0][0]


def _recompute_summary_values(series: RetroTraceLatticeForecastSeries) -> tuple[str, float, float, float, str]:
    size = _lattice_size(series.lattice_mode)
    last_step = series.steps[-1]
    dominant_region = _dominant_region_for_step(last_step, size)
    occupancy_dispersion = _clamp01(sum(step.projected_dispersion_score for step in series.steps) / float(len(series.steps)))
    locality_risk = _clamp01(sum(step.projected_locality_pressure for step in series.steps) / float(len(series.steps)))
    overall_stability = _clamp01(sum(step.stability_score for step in series.steps) / float(len(series.steps)))
    return dominant_region, occupancy_dispersion, locality_risk, overall_stability, _classify(overall_stability)


@dataclass(frozen=True)
class RetroTraceLatticeForecastSummary:
    lattice_mode: str
    dominant_region: str
    occupancy_dispersion: float
    locality_risk: float
    overall_stability_forecast: float
    collapse_risk_classification: str
    _stable_hash: str

    def __post_init__(self) -> None:
        mode = _validate_lattice_mode(self.lattice_mode)
        object.__setattr__(self, "lattice_mode", mode)
        if not isinstance(self.dominant_region, str) or len(self.dominant_region) != 3 or any(c not in {"L", "H"} for c in self.dominant_region):
            raise ValueError("dominant_region must be 3-char region label using L/H")
        for name in ("occupancy_dispersion", "locality_risk", "overall_stability_forecast"):
            object.__setattr__(self, name, _clamp01(validate_unit_interval(getattr(self, name), name)))
        if self.collapse_risk_classification not in (_CLASS_STABLE, _CLASS_DRIFT, _CLASS_UNSTABLE):
            raise ValueError("invalid classification label")
        if self.collapse_risk_classification != _classify(self.overall_stability_forecast):
            raise ValueError("summary classification mismatch")
        if self._stable_hash != sha256_hex(self._payload_without_hash()):
            raise ValueError("stable_hash mismatch")

    def _payload_without_hash(self) -> dict[str, _JSONValue]:
        return {
            "lattice_mode": self.lattice_mode,
            "dominant_region": self.dominant_region,
            "occupancy_dispersion": round12(self.occupancy_dispersion),
            "locality_risk": round12(self.locality_risk),
            "overall_stability_forecast": round12(self.overall_stability_forecast),
            "collapse_risk_classification": self.collapse_risk_classification,
        }

    def to_dict(self) -> dict[str, _JSONValue]:
        return {**self._payload_without_hash(), "stable_hash": self._stable_hash}

    def to_canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return canonical_bytes(self.to_dict())

    def stable_hash(self) -> str:
        return self._stable_hash


@dataclass(frozen=True)
class RetroTraceForecastLatticeReceipt:
    retro_trace_hash: str
    series: RetroTraceLatticeForecastSeries
    summary: RetroTraceLatticeForecastSummary
    _stable_hash: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "retro_trace_hash", validate_sha256_hex(self.retro_trace_hash, "retro_trace_hash"))
        if not isinstance(self.series, RetroTraceLatticeForecastSeries):
            raise ValueError("series must be RetroTraceLatticeForecastSeries")
        if not isinstance(self.summary, RetroTraceLatticeForecastSummary):
            raise ValueError("summary must be RetroTraceLatticeForecastSummary")
        if self.summary.lattice_mode != self.series.lattice_mode:
            raise ValueError("summary lattice_mode mismatch")
        recomputed = _recompute_summary_values(self.series)
        expected = (
            self.summary.dominant_region,
            round12(self.summary.occupancy_dispersion),
            round12(self.summary.locality_risk),
            round12(self.summary.overall_stability_forecast),
            self.summary.collapse_risk_classification,
        )
        compare = (recomputed[0], round12(recomputed[1]), round12(recomputed[2]), round12(recomputed[3]), recomputed[4])
        if expected != compare:
            raise ValueError("summary values mismatch recomputed canonical series values")
        if self._stable_hash != sha256_hex(self._payload_without_hash()):
            raise ValueError("stable_hash mismatch")

    def _payload_without_hash(self) -> dict[str, _JSONValue]:
        return {
            "retro_trace_hash": self.retro_trace_hash,
            "series": self.series.to_dict(),
            "summary": self.summary.to_dict(),
        }

    def to_dict(self) -> dict[str, _JSONValue]:
        return {**self._payload_without_hash(), "stable_hash": self._stable_hash}

    def to_canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return canonical_bytes(self.to_dict())

    def stable_hash(self) -> str:
        return self._stable_hash


def forecast_retro_trace_lattice(
    retro_trace: RetroTraceReceipt,
    horizon: int,
    lattice_mode: str,
) -> RetroTraceForecastLatticeReceipt:
    if not isinstance(retro_trace, RetroTraceReceipt):
        raise ValueError("retro_trace must be RetroTraceReceipt")
    if isinstance(horizon, bool) or not isinstance(horizon, int) or not (_MIN_HORIZON <= horizon <= _MAX_HORIZON):
        raise ValueError(f"horizon must be int in [{_MIN_HORIZON},{_MAX_HORIZON}]")
    mode = _validate_lattice_mode(lattice_mode)

    retro_payload = retro_trace.to_dict()
    observed_hash = str(retro_payload.pop("stable_hash"))
    if observed_hash != sha256_hex(retro_payload):
        raise ValueError("retro_trace stable_hash mismatch")

    size = _lattice_size(mode)
    capacity = size * size * size
    normalized_timing = _norm_timing_vector(retro_trace)

    occupied_counts: dict[tuple[int, int, int], int] = {}
    events = retro_trace.event_sequence
    for event_index, event_type, _payload in events:
        timing_norm = normalized_timing[event_index] if event_index < len(normalized_timing) else 0.0
        coords = _embed_coordinate(
            mode=mode,
            size=size,
            event_index=event_index,
            event_type=event_type,
            normalized_timing=timing_norm,
            trace_length=retro_trace.trace_length,
        )
        occupied_counts[coords] = occupied_counts.get(coords, 0) + 1

    steps: list[RetroTraceLatticeForecastStep] = []
    for step_index in range(1, horizon + 1):
        synth_event_index = retro_trace.trace_length + step_index - 1
        synth_event_type = _EVENT_TYPES[(synth_event_index + len(events) + _SUPPORTED_MODES.index(mode)) % len(_EVENT_TYPES)]
        synth_timing = _clamp01(float(synth_event_index + 1) / float(max(1, retro_trace.trace_length + horizon)))
        synth_coords = _embed_coordinate(
            mode=mode,
            size=size,
            event_index=synth_event_index,
            event_type=synth_event_type,
            normalized_timing=synth_timing,
            trace_length=retro_trace.trace_length,
        )
        occupied_counts[synth_coords] = occupied_counts.get(synth_coords, 0) + 1

        total_events = max(1, sum(occupied_counts.values()))
        sorted_coords = tuple(sorted(occupied_counts.keys()))
        cells = []
        for coords in sorted_coords:
            occupancy_share = _clamp01(float(occupied_counts[coords]) / float(total_events))
            locality_pressure = _neighbor_ratio(coords, occupied_counts)
            cell_payload = {
                "lattice_mode": mode,
                "coordinate": {"x": coords[0], "y": coords[1], "z": coords[2]},
                "occupancy_share": round12(occupancy_share),
                "locality_pressure": round12(locality_pressure),
            }
            cells.append(
                RetroTraceLatticeCell(
                    lattice_mode=mode,
                    coordinate=LatticeCoordinate(x=coords[0], y=coords[1], z=coords[2]),
                    occupancy_share=occupancy_share,
                    locality_pressure=locality_pressure,
                    _stable_hash=sha256_hex(cell_payload),
                )
            )

        projected_occupancy_count = len(sorted_coords)
        projected_density_score = _clamp01(float(projected_occupancy_count) / float(capacity))
        projected_locality_pressure = _clamp01(
            sum(cell.locality_pressure * cell.occupancy_share for cell in cells)
        )
        projected_dispersion_score = _dispersion(sorted_coords, size)
        stability_score = _clamp01(
            0.45 * (1.0 - projected_density_score)
            + 0.30 * (1.0 - projected_locality_pressure)
            + 0.25 * (1.0 - projected_dispersion_score)
        )

        step_payload = {
            "step_index": step_index,
            "projected_occupancy_count": projected_occupancy_count,
            "projected_density_score": round12(projected_density_score),
            "projected_locality_pressure": round12(projected_locality_pressure),
            "projected_dispersion_score": round12(projected_dispersion_score),
            "stability_score": round12(stability_score),
            "occupied_cells": tuple(cell.to_dict() for cell in cells),
        }
        steps.append(
            RetroTraceLatticeForecastStep(
                step_index=step_index,
                projected_occupancy_count=projected_occupancy_count,
                projected_density_score=projected_density_score,
                projected_locality_pressure=projected_locality_pressure,
                projected_dispersion_score=projected_dispersion_score,
                stability_score=stability_score,
                occupied_cells=tuple(cells),
                _stable_hash=sha256_hex(step_payload),
            )
        )

    series_payload = {
        "lattice_mode": mode,
        "horizon": horizon,
        "steps": tuple(step.to_dict() for step in steps),
    }
    series = RetroTraceLatticeForecastSeries(
        lattice_mode=mode,
        horizon=horizon,
        steps=tuple(steps),
        _stable_hash=sha256_hex(series_payload),
    )

    dominant_region, occupancy_dispersion, locality_risk, overall_stability, classification = _recompute_summary_values(series)
    summary_payload = {
        "lattice_mode": mode,
        "dominant_region": dominant_region,
        "occupancy_dispersion": round12(occupancy_dispersion),
        "locality_risk": round12(locality_risk),
        "overall_stability_forecast": round12(overall_stability),
        "collapse_risk_classification": classification,
    }
    summary = RetroTraceLatticeForecastSummary(
        lattice_mode=mode,
        dominant_region=dominant_region,
        occupancy_dispersion=occupancy_dispersion,
        locality_risk=locality_risk,
        overall_stability_forecast=overall_stability,
        collapse_risk_classification=classification,
        _stable_hash=sha256_hex(summary_payload),
    )

    receipt_payload = {
        "retro_trace_hash": retro_trace.stable_hash,
        "series": series.to_dict(),
        "summary": summary.to_dict(),
    }
    return RetroTraceForecastLatticeReceipt(
        retro_trace_hash=retro_trace.stable_hash,
        series=series,
        summary=summary,
        _stable_hash=sha256_hex(receipt_payload),
    )


__all__ = [
    "LatticeCoordinate",
    "RetroTraceLatticeCell",
    "RetroTraceLatticeForecastStep",
    "RetroTraceLatticeForecastSeries",
    "RetroTraceLatticeForecastSummary",
    "RetroTraceForecastLatticeReceipt",
    "forecast_retro_trace_lattice",
]
