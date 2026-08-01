"""Canonical, capability-gated NEXUS invocation contracts."""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Final

from .source import NexusSourceIdentity, source_profile

OPERATIONS: Final = (
    "verify",
    "trace",
    "ternary",
    "receipt",
    "fibonacci",
    "verify-parallel",
)


def _decimal_text(
    value: str,
    field: str,
    *,
    positive: bool = False,
    nonnegative: bool = False,
) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be non-empty decimal text")
    try:
        number = Decimal(value)
    except InvalidOperation as exc:
        raise ValueError(f"{field} must be decimal text") from exc
    if not number.is_finite():
        raise ValueError(f"{field} must be finite")
    if positive and number <= 0:
        raise ValueError(f"{field} must be positive")
    if nonnegative and number < 0:
        raise ValueError(f"{field} must be non-negative")
    return value


@dataclass(frozen=True)
class NexusConfig:
    logical: int = 16_777_216
    rendered: int = 1_024
    particles: int = 512
    radius: str = "0.56"
    phase: str = "0"
    turns: str = "1.5"

    def __post_init__(self) -> None:
        for name in ("logical", "rendered", "particles"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive exact integer")
        if self.rendered < 3:
            raise ValueError("rendered must be at least 3")
        if self.rendered > self.logical:
            raise ValueError("rendered may not exceed logical")
        _decimal_text(self.radius, "radius", positive=True)
        _decimal_text(self.phase, "phase")
        _decimal_text(self.turns, "turns", nonnegative=True)

    def cli_args(self) -> list[str]:
        return [
            "--logical",
            str(self.logical),
            "--rendered",
            str(self.rendered),
            "--particles",
            str(self.particles),
            "--radius",
            self.radius,
            "--phase",
            self.phase,
            "--turns",
            self.turns,
        ]

    def as_dict(self) -> dict[str, object]:
        return {
            "logical": self.logical,
            "rendered": self.rendered,
            "particles": self.particles,
            "radius": self.radius,
            "phase": self.phase,
            "turns": self.turns,
        }


@dataclass(frozen=True)
class NexusInvocation:
    operation: str
    profile: str = "v4.0.0"
    config: NexusConfig = NexusConfig()
    channel: int | None = None
    steps: int | None = None
    samples: int | None = None
    workers: int | None = None
    base_frequency_hz: str | None = None

    def __post_init__(self) -> None:
        if self.operation not in OPERATIONS:
            raise ValueError(f"unsupported NEXUS operation: {self.operation}")
        source = self.source
        if not source.supports(self.operation):
            raise ValueError(f"NEXUS {self.profile} does not support {self.operation}")
        if self.operation in {"trace", "ternary", "fibonacci"}:
            if (
                type(self.channel) is not int
                or self.channel < 0
                or self.channel >= self.config.rendered
            ):
                raise ValueError("channel must be an in-range exact integer")
            if type(self.steps) is not int or self.steps < 2:
                raise ValueError("steps must be an exact integer >= 2")
        elif self.channel is not None or self.steps is not None:
            raise ValueError(
                "channel and steps are only valid for trace-like operations"
            )
        if self.operation == "receipt":
            if type(self.samples) is not int or self.samples <= 0:
                raise ValueError("samples must be a positive exact integer")
        elif self.samples is not None:
            raise ValueError("samples is only valid for receipt")
        if self.operation == "verify-parallel":
            if type(self.workers) is not int or not 1 <= self.workers <= 256:
                raise ValueError("workers must be an exact integer in [1, 256]")
        elif self.workers is not None:
            raise ValueError("workers is only valid for verify-parallel")
        if self.operation == "ternary":
            if self.base_frequency_hz is None:
                raise ValueError("ternary requires base_frequency_hz")
            _decimal_text(
                self.base_frequency_hz,
                "base_frequency_hz",
                positive=True,
            )
        elif self.base_frequency_hz is not None:
            raise ValueError("base_frequency_hz is only valid for ternary")

    @property
    def source(self) -> NexusSourceIdentity:
        return source_profile(self.profile)

    def argv(self, binary: str | Path) -> list[str]:
        args = [str(binary), self.operation]
        if self.operation == "verify-parallel":
            args.append(str(self.workers))
        elif self.operation in {"trace", "fibonacci"}:
            args.extend((str(self.channel), str(self.steps)))
        elif self.operation == "ternary":
            args.extend(
                (
                    str(self.channel),
                    str(self.steps),
                    str(self.base_frequency_hz),
                )
            )
        elif self.operation == "receipt":
            args.append(str(self.samples))
        args.extend(self.config.cli_args())
        return args

    def as_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "operation": self.operation,
            "profile": self.profile,
            "config": self.config.as_dict(),
        }
        for name in (
            "channel",
            "steps",
            "samples",
            "workers",
            "base_frequency_hz",
        ):
            value = getattr(self, name)
            if value is not None:
                payload[name] = value
        return payload
