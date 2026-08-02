# SPDX-License-Identifier: MPL-2.0
"""Deterministic route-tone signatures and verification."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib

from .model import ExchangeConfig


@dataclass(frozen=True)
class ToneSignature:
    route_hz: int
    check_hz: int
    dark_reference_hz: int

    def as_dict(self) -> dict[str, int]:
        return {
            "route_hz": self.route_hz,
            "check_hz": self.check_hz,
            "dark_reference_hz": self.dark_reference_hz,
        }


@dataclass(frozen=True)
class ToneObservation:
    route_hz: int
    check_hz: int
    dark_reference_hz: int

    def as_dict(self) -> dict[str, int]:
        return {
            "route_hz": self.route_hz,
            "check_hz": self.check_hz,
            "dark_reference_hz": self.dark_reference_hz,
        }


def derive_tone_signature(
    *,
    config: ExchangeConfig,
    digits: tuple[int, ...],
    trunk_path: tuple[int, ...],
    destination: str,
) -> ToneSignature:
    identity = (
        ",".join(str(value) for value in digits)
        + "|"
        + ",".join(str(value) for value in trunk_path)
        + "|"
        + destination
    ).encode("utf-8")
    digest = hashlib.sha256(identity).digest()
    route_hz = config.route_tone_base_hz + int.from_bytes(digest[:2], "big") % 1200
    check_hz = config.route_tone_base_hz + int.from_bytes(digest[2:4], "big") % 1200
    return ToneSignature(
        route_hz=route_hz,
        check_hz=check_hz,
        dark_reference_hz=config.dark_reference_hz,
    )


def observe_with_offsets(
    expected: ToneSignature, offsets: tuple[int, int, int]
) -> ToneObservation:
    return ToneObservation(
        route_hz=expected.route_hz + offsets[0],
        check_hz=expected.check_hz + offsets[1],
        dark_reference_hz=expected.dark_reference_hz + offsets[2],
    )


def verify_tones(
    expected: ToneSignature,
    observed: ToneObservation,
    *,
    tolerance_hz: int,
) -> dict[str, object]:
    residuals = {
        "route_hz": observed.route_hz - expected.route_hz,
        "check_hz": observed.check_hz - expected.check_hz,
        "dark_reference_hz": (
            observed.dark_reference_hz - expected.dark_reference_hz
        ),
    }
    matches = {
        name: abs(residual) <= tolerance_hz
        for name, residual in residuals.items()
    }
    return {
        "verified": all(matches.values()),
        "matches": matches,
        "residuals_hz": residuals,
        "tolerance_hz": tolerance_hz,
    }
