# SPDX-License-Identifier: MPL-2.0
"""Strict bounded canonical helpers for the additive acoustic experiment."""
from __future__ import annotations

import re
from types import MappingProxyType
from qec.sonify.canonical import canonical_json, canonical_sha256

CLAIM_BOUNDARY = MappingProxyType({
    "maximum_claim_level": "classical_channel_verified",
    "synthetic_pcm_only": True,
    "physical_quantum_protection": False,
    "quantum_state_evolution": False,
    "source_hashes_authenticate_physical_provenance": False,
    "receiver_acceptance_is_independent_truth": False,
    "released_decoders_and_transport_contracts_modified": False,
})


def seal(name: str, **fields) -> dict:
    value = {"schema": "qec.acoustic-" + name + ".v1", "contract_version": "177.0", **fields}
    return {**value, "sha256": canonical_sha256(value)}


def exact(actual: object, expected: dict) -> None:
    try:
        if type(actual) is not dict or canonical_json(actual) != canonical_json(expected):
            raise ValueError("INVALID_INPUT")
    except (TypeError, UnicodeError, RecursionError) as exc:
        raise ValueError("INVALID_INPUT") from exc


def integer(value: object, maximum: int, minimum: int = 0) -> None:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError("INVALID_INPUT")


def sequence(value: object, maximum: int) -> tuple:
    if type(value) not in (tuple, list) or len(value) > maximum:
        raise ValueError("INVALID_INPUT")
    return tuple(value)


def digest(value: object) -> None:
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("INVALID_INPUT")


def read(cls, value: object, fields: tuple[str, ...], transforms=None):
    try:
        if type(value) is not dict:
            raise ValueError("INVALID_INPUT")
        transforms = transforms or {}
        result = cls(*(transforms.get(k, lambda v: v)(value[k]) for k in fields))
        exact(value, result.as_dict())
        return result
    except (KeyError, TypeError, ValueError, AttributeError, OverflowError, RecursionError) as exc:
        raise ValueError("INVALID_INPUT") from exc
