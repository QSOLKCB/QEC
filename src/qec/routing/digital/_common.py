# SPDX-License-Identifier: MPL-2.0
"""Strict canonical helpers private to the unreleased digital contracts."""
from __future__ import annotations

import hashlib
from types import MappingProxyType

from qec.sonify.canonical import canonical_json, canonical_sha256, validate_sha256

MAX_SLOTS = 16
MAX_FRAMES = 1024
MAX_COMMANDS = 64
MAX_TICK = 2**31 - 1
IDLE_SYMBOL = 256
CLAIM_BOUNDARY = MappingProxyType({
    "classical_software_model_only": True,
    "bounded_closed_batch": True,
    "single_request_transport_only": True,
    "initial_occupancy_is_caller_declared": True,
    "logical_time_only": True,
    "payload_mutation_permitted": False,
    "decoder_output_mutation_permitted": False,
    "decoder_identity_is_caller_declared": True,
    "authenticated_provenance": False,
    "physical_network_or_quantum_evidence": False,
    "external_network_authority": False,
    "receipt_proves": "declared_frames_allocation_alignment_interchange_and_exact_payload_reassembly",
    "release_status": "implemented_unreleased",
})


def integer(value: object, maximum: int, minimum: int = 0) -> None:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError("INVALID_INPUT")


def text(value: object) -> None:
    if type(value) is not str or not 1 <= len(value) <= 128:
        raise ValueError("INVALID_INPUT")
    try:
        value.encode("utf-8")
    except UnicodeError as exc:
        raise ValueError("INVALID_INPUT") from exc


def sequence(value: object, maximum: int) -> tuple:
    if type(value) not in (list, tuple) or len(value) > maximum:
        raise ValueError("INVALID_INPUT")
    return tuple(value)


def digest(value: object) -> str:
    try:
        if type(value) is not str:
            raise ValueError("INVALID_INPUT")
        return validate_sha256(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("INVALID_INPUT") from exc


def payload_hash(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def hex_bytes(value: object, maximum: int) -> bytes:
    if type(value) is not str or len(value) > maximum * 2 or len(value) % 2:
        raise ValueError("INVALID_INPUT")
    try:
        result = bytes.fromhex(value)
        if result.hex() != value:
            raise ValueError("INVALID_INPUT")
        return result
    except ValueError as exc:
        raise ValueError("INVALID_INPUT") from exc


def seal(name: str, version: str, **fields: object) -> dict:
    unsigned = {"schema": "qec.digital-" + name + ".v1", "contract_version": version, **fields}
    return {**unsigned, "sha256": canonical_sha256(unsigned)}


def exact(actual: object, expected: dict) -> None:
    try:
        if type(actual) is not dict or canonical_json(actual) != canonical_json(expected):
            raise ValueError("INVALID_INPUT")
    except (TypeError, UnicodeError, RecursionError) as exc:
        raise ValueError("INVALID_INPUT") from exc


def pins(**pairs: tuple[object, str]) -> dict:
    for expected, actual in pairs.values():
        if expected is not None and digest(expected) != actual:
            raise ValueError("INVALID_INPUT")
    return {key: pair[0] is not None for key, pair in pairs.items()}
