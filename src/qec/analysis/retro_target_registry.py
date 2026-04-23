# SPDX-License-Identifier: MIT
"""v147.0 — Retro Target Registry.

Deterministic analysis-layer registry for retro-constrained compute targets.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Dict, Mapping, Tuple

RETRO_TARGET_REGISTRY_VERSION = "v147.0"

_ALLOWED_ISA_FAMILIES: Tuple[str, ...] = (
    "68k",
    "intel_4004",
    "intel_8008",
    "z80",
    "m6502",
    "tracker_audio",
    "display_raster",
)

_ALLOWED_WORD_SIZES: Tuple[int, ...] = (4, 8, 16, 32)
_ALLOWED_FPU_POLICIES: Tuple[str, ...] = ("none", "emulated", "coprocessor", "native")
_ALLOWED_PROVENANCE: Tuple[str, ...] = ("hardware", "emulator", "hybrid")

_CLASSIFICATION_ORDER: Tuple[str, ...] = (
    "word_starved",
    "memory_starved",
    "no_fpu",
    "fixed_point_bound",
    "scanline_bound",
    "tracker_bound",
    "input_sparse",
)


class RetroTargetValidationError(ValueError):
    """Raised when retro target payload violates deterministic schema rules."""


def _canonical_json(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def _stable_hash(data: Any) -> str:
    return hashlib.sha256(_canonical_json(data).encode("utf-8")).hexdigest()


def _require_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RetroTargetValidationError(f"{field} must be an integer")
    return int(value)


def _require_non_empty_text(value: Any, *, field: str) -> str:
    if not isinstance(value, str):
        raise RetroTargetValidationError(f"{field} must be a non-empty string")
    text = value.strip()
    if not text:
        raise RetroTargetValidationError(f"{field} must be a non-empty string")
    return text


def _canonicalize_budget_primitive(value: Any, *, field: str) -> Any:
    if value is None:
        return None
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            raise RetroTargetValidationError(f"{field} must not contain NaN/inf")
        return float(value)
    if isinstance(value, str):
        return value
    raise RetroTargetValidationError(f"{field} must contain only JSON-serializable primitives")


def _canonicalize_budget_mapping(value: Any, *, field: str) -> Dict[str, Any]:
    if not isinstance(value, Mapping):
        raise RetroTargetValidationError(f"{field} must be a dict")
    result: Dict[str, Any] = {}
    for raw_key in sorted(value.keys(), key=lambda x: str(x)):
        key = str(raw_key)
        if key in result:
            raise RetroTargetValidationError(f"{field} contains duplicate canonical key: {key!r}")
        item = value[raw_key]
        if isinstance(item, (Mapping, list, tuple, set, frozenset)):
            raise RetroTargetValidationError(f"{field}.{key} must not contain nested unordered/container types")
        result[key] = _canonicalize_budget_primitive(item, field=f"{field}.{key}")
    return result


def _clamp_round_01(value: float) -> float:
    if math.isnan(value) or math.isinf(value):
        raise RetroTargetValidationError("metric must be finite")
    if value <= 0.0:
        return 0.0
    if value >= 1.0:
        return 1.0
    return float(round(value, 12))


def _ratio_pressure(numerator: float, baseline: float) -> float:
    denom = numerator + baseline
    if denom <= 0.0:
        return 0.0
    return _clamp_round_01(float(numerator / denom))


def _numeric_budget_mass(budget: Mapping[str, Any]) -> float:
    total = 0.0
    for key in sorted(budget.keys()):
        value = budget[key]
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            total += float(value)
    return float(total)


def _compute_constraint_metrics(descriptor: "RetroTargetDescriptor") -> Dict[str, float]:
    arithmetic_constraint_pressure = _ratio_pressure(32.0 / float(descriptor.word_size), 16.0)
    memory_constraint_pressure = _ratio_pressure(
        (1_048_576.0 / float(descriptor.ram_budget + descriptor.rom_budget + 1)),
        1.0,
    )
    timing_constraint_pressure = _ratio_pressure(10_000_000.0 / float(descriptor.cycle_budget), 1.0)

    display_mass = _numeric_budget_mass(descriptor.display_budget)
    audio_mass = _numeric_budget_mass(descriptor.audio_budget)
    input_mass = _numeric_budget_mass(descriptor.input_budget)

    display_constraint_pressure = _ratio_pressure(1_000.0 / float(display_mass + 1.0), 1.0)
    audio_constraint_pressure = _ratio_pressure(1_000.0 / float(audio_mass + 1.0), 1.0)

    replay_raw = 1.0
    if descriptor.provenance == "emulator":
        replay_raw -= 0.20
    elif descriptor.provenance == "hybrid":
        replay_raw -= 0.10
    if descriptor.fpu_policy == "emulated":
        replay_raw -= 0.05
    if descriptor.word_size < 16:
        replay_raw -= 0.05
    if input_mass <= 2.0:
        replay_raw -= 0.05
    replay_surface_clarity = _clamp_round_01(replay_raw)

    metrics = {
        "arithmetic_constraint_pressure": _clamp_round_01(arithmetic_constraint_pressure),
        "memory_constraint_pressure": _clamp_round_01(memory_constraint_pressure),
        "timing_constraint_pressure": _clamp_round_01(timing_constraint_pressure),
        "display_constraint_pressure": _clamp_round_01(display_constraint_pressure),
        "audio_constraint_pressure": _clamp_round_01(audio_constraint_pressure),
        "replay_surface_clarity": _clamp_round_01(replay_surface_clarity),
    }
    return {k: float(metrics[k]) for k in sorted(metrics.keys())}


def _derive_labels(descriptor: "RetroTargetDescriptor", metrics: Mapping[str, float]) -> Tuple[str, ...]:
    labels = []
    ram_total = descriptor.ram_budget + descriptor.rom_budget
    input_mass = _numeric_budget_mass(descriptor.input_budget)

    if descriptor.word_size <= 8:
        labels.append("word_starved")
    if ram_total < 65_536:
        labels.append("memory_starved")
    if descriptor.fpu_policy == "none":
        labels.append("no_fpu")
    if descriptor.fpu_policy in ("none", "emulated"):
        labels.append("fixed_point_bound")
    if descriptor.isa_family == "display_raster" or metrics["display_constraint_pressure"] >= 0.6:
        labels.append("scanline_bound")
    if descriptor.isa_family == "tracker_audio" or metrics["audio_constraint_pressure"] >= 0.6:
        labels.append("tracker_bound")
    if input_mass <= 2.0:
        labels.append("input_sparse")

    ordered = tuple(label for label in _CLASSIFICATION_ORDER if label in labels)
    if not ordered:
        return ("balanced_retro",)
    return ordered


@dataclass(frozen=True)
class RetroTargetDescriptor:
    target_id: str
    isa_family: str
    word_size: int
    address_width: int
    ram_budget: int
    rom_budget: int
    cycle_budget: int
    display_budget: Dict[str, Any]
    audio_budget: Dict[str, Any]
    input_budget: Dict[str, Any]
    fpu_policy: str
    provenance: str

    def __post_init__(self) -> None:
        target_id = _require_non_empty_text(self.target_id, field="target_id")
        isa_family = _require_non_empty_text(self.isa_family, field="isa_family")
        if isa_family not in _ALLOWED_ISA_FAMILIES:
            raise RetroTargetValidationError(f"isa_family must be one of {_ALLOWED_ISA_FAMILIES}")

        word_size = _require_int(self.word_size, field="word_size")
        if word_size not in _ALLOWED_WORD_SIZES:
            raise RetroTargetValidationError(f"word_size must be one of {_ALLOWED_WORD_SIZES}")

        address_width = _require_int(self.address_width, field="address_width")
        if address_width <= 0:
            raise RetroTargetValidationError("address_width must be > 0")

        ram_budget = _require_int(self.ram_budget, field="ram_budget")
        if ram_budget < 0:
            raise RetroTargetValidationError("ram_budget must be >= 0")

        rom_budget = _require_int(self.rom_budget, field="rom_budget")
        if rom_budget < 0:
            raise RetroTargetValidationError("rom_budget must be >= 0")

        cycle_budget = _require_int(self.cycle_budget, field="cycle_budget")
        if cycle_budget <= 0:
            raise RetroTargetValidationError("cycle_budget must be > 0")

        fpu_policy = _require_non_empty_text(self.fpu_policy, field="fpu_policy")
        if fpu_policy not in _ALLOWED_FPU_POLICIES:
            raise RetroTargetValidationError(f"fpu_policy must be one of {_ALLOWED_FPU_POLICIES}")

        provenance = _require_non_empty_text(self.provenance, field="provenance")
        if provenance not in _ALLOWED_PROVENANCE:
            raise RetroTargetValidationError(f"provenance must be one of {_ALLOWED_PROVENANCE}")

        object.__setattr__(self, "target_id", target_id)
        object.__setattr__(self, "isa_family", isa_family)
        object.__setattr__(self, "word_size", word_size)
        object.__setattr__(self, "address_width", address_width)
        object.__setattr__(self, "ram_budget", ram_budget)
        object.__setattr__(self, "rom_budget", rom_budget)
        object.__setattr__(self, "cycle_budget", cycle_budget)
        object.__setattr__(self, "display_budget", _canonicalize_budget_mapping(self.display_budget, field="display_budget"))
        object.__setattr__(self, "audio_budget", _canonicalize_budget_mapping(self.audio_budget, field="audio_budget"))
        object.__setattr__(self, "input_budget", _canonicalize_budget_mapping(self.input_budget, field="input_budget"))
        object.__setattr__(self, "fpu_policy", fpu_policy)
        object.__setattr__(self, "provenance", provenance)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target_id": self.target_id,
            "isa_family": self.isa_family,
            "word_size": int(self.word_size),
            "address_width": int(self.address_width),
            "ram_budget": int(self.ram_budget),
            "rom_budget": int(self.rom_budget),
            "cycle_budget": int(self.cycle_budget),
            "display_budget": dict(self.display_budget),
            "audio_budget": dict(self.audio_budget),
            "input_budget": dict(self.input_budget),
            "fpu_policy": self.fpu_policy,
            "provenance": self.provenance,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")

    def stable_hash(self) -> str:
        return _stable_hash(self.to_dict())


@dataclass(frozen=True)
class RetroTargetReceipt:
    descriptor: RetroTargetDescriptor
    constraint_metrics: Dict[str, float]
    classification_labels: Tuple[str, ...]
    registry_version: str
    stable_hash: str

    def __post_init__(self) -> None:
        metrics: Dict[str, float] = {}
        if not isinstance(self.constraint_metrics, Mapping):
            raise RetroTargetValidationError("constraint_metrics must be a dict")
        for key in sorted(self.constraint_metrics.keys()):
            value = self.constraint_metrics[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise RetroTargetValidationError(f"constraint_metrics.{key} must be float")
            numeric = float(value)
            if math.isnan(numeric) or math.isinf(numeric):
                raise RetroTargetValidationError(f"constraint_metrics.{key} must be finite")
            metrics[str(key)] = _clamp_round_01(numeric)

        labels_raw = self.classification_labels
        if not isinstance(labels_raw, tuple):
            raise RetroTargetValidationError("classification_labels must be tuple")
        labels = tuple(str(item) for item in labels_raw)

        object.__setattr__(self, "constraint_metrics", {k: float(metrics[k]) for k in sorted(metrics.keys())})
        object.__setattr__(self, "classification_labels", labels)

        if len(self.stable_hash) != 64 or self.stable_hash.lower() != self.stable_hash:
            raise RetroTargetValidationError("stable_hash must be lowercase SHA-256 hex")

    def _hash_payload(self) -> Dict[str, Any]:
        return {
            "descriptor": self.descriptor.to_dict(),
            "constraint_metrics": {k: float(self.constraint_metrics[k]) for k in sorted(self.constraint_metrics.keys())},
            "classification_labels": list(self.classification_labels),
            "registry_version": self.registry_version,
        }

    def to_dict(self) -> Dict[str, Any]:
        payload = self._hash_payload()
        payload["stable_hash"] = self.stable_hash
        return payload

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")


@dataclass(frozen=True)
class RetroTargetRegistry:
    targets: Tuple[RetroTargetReceipt, ...]
    registry_version: str
    stable_hash: str

    def __post_init__(self) -> None:
        if not isinstance(self.targets, tuple):
            raise RetroTargetValidationError("targets must be tuple")
        sorted_targets = tuple(sorted(self.targets, key=lambda x: x.descriptor.target_id))
        seen = set()
        for target in sorted_targets:
            target_id = target.descriptor.target_id
            if target_id in seen:
                raise RetroTargetValidationError(f"duplicate target_id: {target_id!r}")
            seen.add(target_id)
        object.__setattr__(self, "targets", sorted_targets)
        if len(self.stable_hash) != 64 or self.stable_hash.lower() != self.stable_hash:
            raise RetroTargetValidationError("stable_hash must be lowercase SHA-256 hex")

    def _hash_payload(self) -> Dict[str, Any]:
        return {
            "targets": [target.to_dict() for target in self.targets],
            "registry_version": self.registry_version,
        }

    def to_dict(self) -> Dict[str, Any]:
        payload = self._hash_payload()
        payload["stable_hash"] = self.stable_hash
        return payload

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")


def build_retro_target(
    target_id: str,
    isa_family: str,
    word_size: int,
    address_width: int,
    ram_budget: int,
    rom_budget: int,
    cycle_budget: int,
    display_budget: Mapping[str, Any],
    audio_budget: Mapping[str, Any],
    input_budget: Mapping[str, Any],
    fpu_policy: str,
    provenance: str,
) -> RetroTargetReceipt:
    descriptor = RetroTargetDescriptor(
        target_id=target_id,
        isa_family=isa_family,
        word_size=word_size,
        address_width=address_width,
        ram_budget=ram_budget,
        rom_budget=rom_budget,
        cycle_budget=cycle_budget,
        display_budget=display_budget,
        audio_budget=audio_budget,
        input_budget=input_budget,
        fpu_policy=fpu_policy,
        provenance=provenance,
    )

    metrics = _compute_constraint_metrics(descriptor)
    labels = _derive_labels(descriptor, metrics)
    payload = {
        "descriptor": descriptor.to_dict(),
        "constraint_metrics": metrics,
        "classification_labels": list(labels),
        "registry_version": RETRO_TARGET_REGISTRY_VERSION,
    }
    receipt_hash = _stable_hash(payload)
    return RetroTargetReceipt(
        descriptor=descriptor,
        constraint_metrics=metrics,
        classification_labels=labels,
        registry_version=RETRO_TARGET_REGISTRY_VERSION,
        stable_hash=receipt_hash,
    )


def build_retro_target_registry(targets: Tuple[RetroTargetReceipt, ...]) -> RetroTargetRegistry:
    sorted_targets = tuple(sorted(targets, key=lambda x: x.descriptor.target_id))
    payload = {
        "targets": [target.to_dict() for target in sorted_targets],
        "registry_version": RETRO_TARGET_REGISTRY_VERSION,
    }
    registry_hash = _stable_hash(payload)
    return RetroTargetRegistry(
        targets=sorted_targets,
        registry_version=RETRO_TARGET_REGISTRY_VERSION,
        stable_hash=registry_hash,
    )
