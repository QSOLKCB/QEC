# SPDX-License-Identifier: MPL-2.0
"""Deterministic data model for the virtual Strowger syndrome exchange."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Final, Mapping

from qec.sonify.canonical import canonical_sha256, require_int, require_nonempty_text


class ExchangeMode(str, Enum):
    AUTOMATIC = "automatic"
    SUPERVISED = "supervised"
    MANUAL = "manual"


class DeviceState(str, Enum):
    HOME = "home"
    SEIZED = "seized"
    RECEIVING_PULSES = "receiving_pulses"
    VERTICAL_STEPPING = "vertical_stepping"
    TRUNK_HUNTING = "trunk_hunting"
    ROTARY_STEPPING = "rotary_stepping"
    CONTACT_TEST = "contact_test"
    CONNECTED = "connected"
    VERIFYING = "verifying"
    COMMITTED = "committed"
    BUSY = "busy"
    RELEASING = "releasing"
    QUARANTINED = "quarantined"
    FAULT = "fault"


class TrunkState(str, Enum):
    FREE = "free"
    BUSY = "busy"
    QUARANTINED = "quarantined"


class RouteOutcome(str, Enum):
    COMMITTED = "committed"
    NO_LINEFINDER_AVAILABLE = "no_linefinder_available"
    ALL_TRUNKS_BUSY = "all_trunks_busy"
    TONE_MISMATCH = "tone_mismatch"
    SELECTOR_FAULT = "selector_fault"
    CONNECTOR_FAULT = "connector_fault"
    OPERATOR_RELEASED = "operator_released"


_CLAIM_BOUNDARY_VALUES: Final = {
    "classical_routing_only": True,
    "decoder_replacement": False,
    "quantum_hardware_claim": False,
    "tone_proves_decoder_correctness": False,
    "operator_may_force_accept": False,
    "receipt_proves": "deterministic_route_and_declared_verification_events",
    "receipt_does_not_prove": "physical_truth_or_quantum_advantage",
}
CLAIM_BOUNDARY: Final[Mapping[str, object]] = MappingProxyType(
    _CLAIM_BOUNDARY_VALUES
)


def _exact_object(
    payload: object,
    *,
    label: str,
    fields: set[str],
) -> dict[str, object]:
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be an object")
    if set(payload) != fields:
        raise ValueError(f"{label} must contain exactly the canonical fields")
    return payload


def _exact_list(payload: object, *, label: str) -> list[object]:
    if not isinstance(payload, list):
        raise ValueError(f"{label} must be a list")
    return payload


@dataclass(frozen=True)
class StageConfig:
    """One selector stage in a mixed-radix exchange."""

    name: str
    radix: int
    trunks: int = 10

    def __post_init__(self) -> None:
        require_nonempty_text(self.name, "name")
        require_int(self.radix, "radix", minimum=2, maximum=256)
        require_int(self.trunks, "trunks", minimum=1, maximum=4096)

    @classmethod
    def from_dict(cls, payload: object) -> "StageConfig":
        record = _exact_object(
            payload,
            label="stage config",
            fields={"name", "radix", "trunks"},
        )
        return cls(
            name=record["name"],
            radix=record["radix"],
            trunks=record["trunks"],
        )

    def as_dict(self) -> dict[str, object]:
        return {"name": self.name, "radix": self.radix, "trunks": self.trunks}


@dataclass(frozen=True)
class ExchangeConfig:
    """Complete exchange topology."""

    linefinders: int
    selectors: tuple[StageConfig, ...]
    connector_vertical_radix: int
    connector_rotary_radix: int
    route_tone_base_hz: int = 320
    dark_reference_hz: int = 90
    tone_tolerance_hz: int = 0

    def __post_init__(self) -> None:
        require_int(self.linefinders, "linefinders", minimum=1, maximum=4096)
        if not self.selectors:
            raise ValueError("selectors must contain at least one stage")
        if len({stage.name for stage in self.selectors}) != len(self.selectors):
            raise ValueError("selector stage names must be unique")
        require_int(
            self.connector_vertical_radix,
            "connector_vertical_radix",
            minimum=2,
            maximum=256,
        )
        require_int(
            self.connector_rotary_radix,
            "connector_rotary_radix",
            minimum=2,
            maximum=4096,
        )
        require_int(
            self.route_tone_base_hz,
            "route_tone_base_hz",
            minimum=20,
            maximum=20000,
        )
        require_int(
            self.dark_reference_hz,
            "dark_reference_hz",
            minimum=1,
            maximum=20000,
        )
        require_int(
            self.tone_tolerance_hz,
            "tone_tolerance_hz",
            minimum=0,
            maximum=1000,
        )

    @classmethod
    def from_dict(cls, payload: object) -> "ExchangeConfig":
        record = _exact_object(
            payload,
            label="exchange config",
            fields={
                "linefinders",
                "selectors",
                "connector_vertical_radix",
                "connector_rotary_radix",
                "route_tone_base_hz",
                "dark_reference_hz",
                "tone_tolerance_hz",
            },
        )
        selectors = _exact_list(record["selectors"], label="selectors")
        return cls(
            linefinders=record["linefinders"],
            selectors=tuple(StageConfig.from_dict(item) for item in selectors),
            connector_vertical_radix=record["connector_vertical_radix"],
            connector_rotary_radix=record["connector_rotary_radix"],
            route_tone_base_hz=record["route_tone_base_hz"],
            dark_reference_hz=record["dark_reference_hz"],
            tone_tolerance_hz=record["tone_tolerance_hz"],
        )

    @property
    def route_radices(self) -> tuple[int, ...]:
        return tuple(stage.radix for stage in self.selectors) + (
            self.connector_vertical_radix,
            self.connector_rotary_radix,
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "linefinders": self.linefinders,
            "selectors": [stage.as_dict() for stage in self.selectors],
            "connector_vertical_radix": self.connector_vertical_radix,
            "connector_rotary_radix": self.connector_rotary_radix,
            "route_tone_base_hz": self.route_tone_base_hz,
            "dark_reference_hz": self.dark_reference_hz,
            "tone_tolerance_hz": self.tone_tolerance_hz,
        }


@dataclass(frozen=True)
class RouteRequest:
    """A correction-routing request expressed as mixed-radix digits."""

    request_id: str
    digits: tuple[int, ...]
    epoch: int
    destination: str

    def __post_init__(self) -> None:
        require_nonempty_text(self.request_id, "request_id")
        require_nonempty_text(self.destination, "destination")
        require_int(self.epoch, "epoch", minimum=0)
        if not self.digits:
            raise ValueError("digits must not be empty")
        for index, digit in enumerate(self.digits):
            require_int(digit, f"digits[{index}]", minimum=0)

    @classmethod
    def from_dict(cls, payload: object) -> "RouteRequest":
        record = _exact_object(
            payload,
            label="route request",
            fields={"request_id", "digits", "epoch", "destination"},
        )
        digits = _exact_list(record["digits"], label="request digits")
        return cls(
            request_id=record["request_id"],
            digits=tuple(digits),
            epoch=record["epoch"],
            destination=record["destination"],
        )

    def validate_against(self, config: ExchangeConfig) -> None:
        if len(self.digits) != len(config.route_radices):
            raise ValueError("route digit count does not match exchange topology")
        for index, (digit, radix) in enumerate(zip(self.digits, config.route_radices)):
            if digit >= radix:
                raise ValueError(f"digits[{index}] must be < radix {radix}")

    def as_dict(self) -> dict[str, object]:
        return {
            "request_id": self.request_id,
            "digits": list(self.digits),
            "epoch": self.epoch,
            "destination": self.destination,
        }

    def sha256(self) -> str:
        return canonical_sha256(self.as_dict())


@dataclass(frozen=True)
class FaultPlan:
    """Deterministic fault injection for research and demonstration."""

    missed_pulses: tuple[tuple[int, int], ...] = ()
    duplicate_pulses: tuple[tuple[int, int], ...] = ()
    stuck_selectors: tuple[int, ...] = ()
    tone_offsets_hz: tuple[int, int, int] = (0, 0, 0)

    def __post_init__(self) -> None:
        for name, pairs in (
            ("missed_pulses", self.missed_pulses),
            ("duplicate_pulses", self.duplicate_pulses),
        ):
            if len(set(pairs)) != len(pairs):
                raise ValueError(f"{name} must contain unique stage/pulse pairs")
            for stage, pulse in pairs:
                require_int(stage, f"{name}.stage", minimum=0)
                require_int(pulse, f"{name}.pulse", minimum=1)
        if len(set(self.stuck_selectors)) != len(self.stuck_selectors):
            raise ValueError("stuck_selectors must be unique")
        for stage in self.stuck_selectors:
            require_int(stage, "stuck_selector", minimum=0)
        if len(self.tone_offsets_hz) != 3:
            raise ValueError("tone_offsets_hz must contain three offsets")
        for value in self.tone_offsets_hz:
            require_int(value, "tone_offset_hz", minimum=-20000, maximum=20000)

    @classmethod
    def from_dict(cls, payload: object) -> "FaultPlan":
        record = _exact_object(
            payload,
            label="fault plan",
            fields={
                "missed_pulses",
                "duplicate_pulses",
                "stuck_selectors",
                "tone_offsets_hz",
            },
        )

        def pairs(name: str) -> tuple[tuple[int, int], ...]:
            values = _exact_list(record[name], label=name)
            result: list[tuple[int, int]] = []
            for index, pair in enumerate(values):
                pair_list = _exact_list(pair, label=f"{name}[{index}]")
                if len(pair_list) != 2:
                    raise ValueError(f"{name}[{index}] must contain two integers")
                result.append((pair_list[0], pair_list[1]))
            return tuple(result)

        stuck = _exact_list(record["stuck_selectors"], label="stuck_selectors")
        offsets = _exact_list(record["tone_offsets_hz"], label="tone_offsets_hz")
        if len(offsets) != 3:
            raise ValueError("tone_offsets_hz must contain three integers")
        return cls(
            missed_pulses=pairs("missed_pulses"),
            duplicate_pulses=pairs("duplicate_pulses"),
            stuck_selectors=tuple(stuck),
            tone_offsets_hz=(offsets[0], offsets[1], offsets[2]),
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "missed_pulses": [list(pair) for pair in self.missed_pulses],
            "duplicate_pulses": [list(pair) for pair in self.duplicate_pulses],
            "stuck_selectors": list(self.stuck_selectors),
            "tone_offsets_hz": list(self.tone_offsets_hz),
        }
