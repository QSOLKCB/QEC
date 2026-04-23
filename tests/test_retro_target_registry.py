# SPDX-License-Identifier: MIT
"""Deterministic tests for v147.0 retro target registry."""

from __future__ import annotations

import json

import pytest

from qec.analysis.retro_target_registry import (
    RETRO_TARGET_REGISTRY_VERSION,
    RetroTargetDescriptor,
    RetroTargetReceipt,
    RetroTargetRegistry,
    RetroTargetValidationError,
    build_retro_target,
    build_retro_target_registry,
)


def _payload(**overrides: object) -> dict:
    base = {
        "target_id": "z80-home-micro",
        "isa_family": "z80",
        "word_size": 8,
        "address_width": 16,
        "ram_budget": 48 * 1024,
        "rom_budget": 32 * 1024,
        "cycle_budget": 3_500_000,
        "display_budget": {"width": 256, "height": 192, "colors": 16},
        "audio_budget": {"channels": 3, "sample_rate": 44_100},
        "input_budget": {"buttons": 2, "axes": 0},
        "fpu_policy": "none",
        "provenance": "hardware",
    }
    base.update(overrides)
    return base


def _build(**overrides: object):
    return build_retro_target(**_payload(**overrides))


def test_deterministic_replay_same_input_same_hash() -> None:
    a = _build()
    b = _build()
    assert a.stable_hash == b.stable_hash
    assert a.to_canonical_bytes() == b.to_canonical_bytes()


def test_invalid_schema_rejection() -> None:
    with pytest.raises(RetroTargetValidationError, match="isa_family"):
        _build(isa_family="arm")

    with pytest.raises(RetroTargetValidationError, match="ram_budget"):
        _build(ram_budget=-1)

    with pytest.raises(RetroTargetValidationError, match="display_budget"):
        _build(display_budget=[("width", 320)])


def test_metric_bounds_enforced() -> None:
    receipt = _build()
    for value in receipt.constraint_metrics.values():
        assert 0.0 <= value <= 1.0
        assert isinstance(value, float)


def test_classification_is_deterministic_ordered_tuple() -> None:
    a = _build()
    b = _build()
    assert isinstance(a.classification_labels, tuple)
    assert a.classification_labels == b.classification_labels


def test_canonical_serialization_identical_across_rebuild() -> None:
    a = _build()
    b = _build()
    assert a.to_canonical_json() == b.to_canonical_json()


def test_hash_stability_serialize_deserialize_reserialize() -> None:
    receipt = _build()
    parsed = json.loads(receipt.to_canonical_json())
    rebuilt = build_retro_target(**parsed["descriptor"])
    assert rebuilt.stable_hash == receipt.stable_hash
    assert rebuilt.to_canonical_json() == receipt.to_canonical_json()


def test_duplicate_target_rejection_registry() -> None:
    receipt = _build()
    with pytest.raises(RetroTargetValidationError, match="duplicate target_id"):
        RetroTargetRegistry(
            targets=(receipt, receipt),
            registry_version=RETRO_TARGET_REGISTRY_VERSION,
            stable_hash="0" * 64,
        )


def test_registry_sorts_targets_deterministically() -> None:
    b = _build(target_id="b")
    a = _build(target_id="a")
    registry = build_retro_target_registry((b, a))
    assert tuple(item.descriptor.target_id for item in registry.targets) == ("a", "b")


def test_mutation_safety_descriptor_frozen() -> None:
    receipt = _build()
    with pytest.raises(Exception):
        receipt.descriptor.word_size = 16


def test_canonical_json_round_trip() -> None:
    receipt = _build()
    j1 = receipt.to_canonical_json()
    j2 = json.dumps(json.loads(j1), sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
    assert j1 == j2


def test_hash_tamper_detection() -> None:
    receipt = _build()
    tampered_metrics = dict(receipt.constraint_metrics)
    tampered_metrics["memory_constraint_pressure"] = 0.111
    with pytest.raises(RetroTargetValidationError, match="stable_hash mismatch"):
        RetroTargetReceipt(
            descriptor=RetroTargetDescriptor(**receipt.descriptor.to_dict()),
            constraint_metrics=tampered_metrics,
            classification_labels=receipt.classification_labels,
            registry_version=receipt.registry_version,
            stable_hash=receipt.stable_hash,
        )


def test_invalid_classification_label_rejected() -> None:
    receipt = _build()
    with pytest.raises(RetroTargetValidationError, match="invalid classification_label"):
        RetroTargetReceipt(
            descriptor=receipt.descriptor,
            constraint_metrics=receipt.constraint_metrics,
            classification_labels=("not_a_real_label",),
            registry_version=receipt.registry_version,
            stable_hash="0" * 64,
        )
