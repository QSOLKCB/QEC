"""
QEC Code Zoo — Deterministic Code-Family Registry (v136.8.2).

Canonical constructors and metadata for supported QEC code families:
- repetition
- surface
- toric
- qldpc

Design invariants
-----------------
* frozen dataclasses only
* deterministic construction — same input always produces identical CodeSpec
* stable registry ordering — sorted by (family, distance, code_id)
* canonical JSON + SHA-256 registry hashing
* no decoder imports
* no hidden randomness
* no new dependencies (stdlib only)
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping, Tuple


# ---------------------------------------------------------------------------
# Registry version
# ---------------------------------------------------------------------------

REGISTRY_VERSION = "v136.8.2"


# ---------------------------------------------------------------------------
# Frozen dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CodeSpec:
    """Immutable specification for a single QEC code instance."""

    code_id: str
    family: str
    distance: int
    logical_qubits: int
    physical_qubits: int
    stabilizer_count: int
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class CodeZooRegistry:
    """Immutable registry of QEC code specifications."""

    codes: Tuple[CodeSpec, ...]
    registry_version: str
    state_hash: str


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_code_spec(spec: CodeSpec) -> bool:
    """Validate a CodeSpec against invariant rules.

    Returns True if valid. Raises ValueError if invalid.
    """
    if not isinstance(spec.code_id, str) or not spec.code_id:
        raise ValueError(f"code_id must be a non-empty string, got {spec.code_id!r}")
    if not isinstance(spec.family, str) or not spec.family:
        raise ValueError(f"family must be a non-empty string, got {spec.family!r}")
    if not isinstance(spec.distance, int) or spec.distance < 1:
        raise ValueError(f"distance must be an integer >= 1, got {spec.distance!r}")
    if not isinstance(spec.logical_qubits, int) or spec.logical_qubits < 1:
        raise ValueError(
            f"logical_qubits must be an integer >= 1, got {spec.logical_qubits!r}"
        )
    if not isinstance(spec.physical_qubits, int) or spec.physical_qubits < 1:
        raise ValueError(
            f"physical_qubits must be an integer >= 1, got {spec.physical_qubits!r}"
        )
    if not isinstance(spec.stabilizer_count, int) or spec.stabilizer_count < 0:
        raise ValueError(
            f"stabilizer_count must be an integer >= 0, got {spec.stabilizer_count!r}"
        )
    if not isinstance(spec.metadata, Mapping):
        raise ValueError(
            f"metadata must be a Mapping, got {type(spec.metadata).__name__}"
        )
    return True


def validate_registry(registry: CodeZooRegistry) -> bool:
    """Validate a CodeZooRegistry against invariant rules.

    Returns True if valid. Raises ValueError if invalid.
    """
    if not isinstance(registry.codes, tuple):
        raise ValueError(
            f"codes must be a tuple, got {type(registry.codes).__name__}"
        )
    if not isinstance(registry.registry_version, str) or not registry.registry_version:
        raise ValueError(
            f"registry_version must be a non-empty string, "
            f"got {registry.registry_version!r}"
        )
    if not isinstance(registry.state_hash, str) or len(registry.state_hash) != 64:
        raise ValueError(
            f"state_hash must be a 64-char hex string, got {registry.state_hash!r}"
        )
    try:
        int(registry.state_hash, 16)
    except ValueError:
        raise ValueError(
            f"state_hash must be a valid hex string, got {registry.state_hash!r}"
        )

    # Validate each spec
    for spec in registry.codes:
        validate_code_spec(spec)

    # Validate ordering: must be sorted by (family, distance, code_id)
    sort_keys = [(s.family, s.distance, s.code_id) for s in registry.codes]
    if sort_keys != sorted(sort_keys):
        raise ValueError("Registry codes must be sorted by (family, distance, code_id)")

    # Validate hash matches
    expected_hash = compute_code_registry_hash(registry)
    if registry.state_hash != expected_hash:
        raise ValueError(
            f"state_hash mismatch: expected {expected_hash}, got {registry.state_hash}"
        )

    return True


# ---------------------------------------------------------------------------
# Code constructors
# ---------------------------------------------------------------------------


def build_repetition_code(distance: int) -> CodeSpec:
    """Build a repetition code specification.

    physical_qubits = distance
    logical_qubits = 1
    stabilizer_count = distance - 1
    """
    if not isinstance(distance, int) or distance < 1:
        raise ValueError(f"distance must be an integer >= 1, got {distance!r}")
    return CodeSpec(
        code_id=f"repetition_d{distance}",
        family="repetition",
        distance=distance,
        logical_qubits=1,
        physical_qubits=distance,
        stabilizer_count=distance - 1,
        metadata={"type": "css", "dimension": 1},
    )


def build_surface_code(distance: int) -> CodeSpec:
    """Build a surface code specification.

    physical_qubits = distance * distance
    logical_qubits = 1
    stabilizer_count = physical_qubits - 1
    """
    if not isinstance(distance, int) or distance < 1:
        raise ValueError(f"distance must be an integer >= 1, got {distance!r}")
    physical = distance * distance
    return CodeSpec(
        code_id=f"surface_d{distance}",
        family="surface",
        distance=distance,
        logical_qubits=1,
        physical_qubits=physical,
        stabilizer_count=physical - 1,
        metadata={"type": "css", "dimension": 2},
    )


def build_toric_code(distance: int) -> CodeSpec:
    """Build a toric code specification.

    physical_qubits = 2 * distance * distance
    logical_qubits = 2
    stabilizer_count = physical_qubits - 2
    """
    if not isinstance(distance, int) or distance < 1:
        raise ValueError(f"distance must be an integer >= 1, got {distance!r}")
    physical = 2 * distance * distance
    return CodeSpec(
        code_id=f"toric_d{distance}",
        family="toric",
        distance=distance,
        logical_qubits=2,
        physical_qubits=physical,
        stabilizer_count=physical - 2,
        metadata={"type": "css", "dimension": 2, "topology": "torus"},
    )


def build_qldpc_code(n: int, k: int, d: int) -> CodeSpec:
    """Build a quantum LDPC code specification.

    Uses explicit provided parameters.
    """
    if not isinstance(n, int) or n < 1:
        raise ValueError(f"n must be an integer >= 1, got {n!r}")
    if not isinstance(k, int) or k < 1:
        raise ValueError(f"k must be an integer >= 1, got {k!r}")
    if not isinstance(d, int) or d < 1:
        raise ValueError(f"d must be an integer >= 1, got {d!r}")
    if k > n:
        raise ValueError(f"k ({k}) must be <= n ({n})")
    return CodeSpec(
        code_id=f"qldpc_n{n}_k{k}_d{d}",
        family="qldpc",
        distance=d,
        logical_qubits=k,
        physical_qubits=n,
        stabilizer_count=n - k,
        metadata={"type": "qldpc", "rate": k / n},
    )


# ---------------------------------------------------------------------------
# Registry operations
# ---------------------------------------------------------------------------


def register_code(
    spec: CodeSpec, registry: CodeZooRegistry | None = None,
) -> CodeZooRegistry:
    """Register a CodeSpec into a registry, returning a new registry.

    Maintains sorted order by (family, distance, code_id).
    Recomputes the state hash.
    """
    validate_code_spec(spec)

    if registry is None:
        codes = (spec,)
    else:
        # Check for duplicate code_id
        for existing in registry.codes:
            if existing.code_id == spec.code_id:
                raise ValueError(f"Duplicate code_id: {spec.code_id!r}")
        codes = tuple(
            sorted(
                registry.codes + (spec,),
                key=lambda s: (s.family, s.distance, s.code_id),
            )
        )

    # Build temporary registry to compute hash
    tmp = CodeZooRegistry(codes=codes, registry_version=REGISTRY_VERSION, state_hash="")
    state_hash = compute_code_registry_hash(tmp)
    return CodeZooRegistry(
        codes=codes,
        registry_version=REGISTRY_VERSION,
        state_hash=state_hash,
    )


def build_default_code_zoo() -> CodeZooRegistry:
    """Build the default code zoo with canonical code families.

    Includes repetition (d=3,5,7), surface (d=3,5), toric (d=3,5),
    and one qldpc code.

    Deterministic: same call always produces identical registry.
    """
    specs = [
        build_repetition_code(3),
        build_repetition_code(5),
        build_repetition_code(7),
        build_surface_code(3),
        build_surface_code(5),
        build_toric_code(3),
        build_toric_code(5),
        build_qldpc_code(n=30, k=8, d=6),
    ]
    codes = tuple(sorted(specs, key=lambda s: (s.family, s.distance, s.code_id)))
    tmp = CodeZooRegistry(codes=codes, registry_version=REGISTRY_VERSION, state_hash="")
    state_hash = compute_code_registry_hash(tmp)
    return CodeZooRegistry(
        codes=codes,
        registry_version=REGISTRY_VERSION,
        state_hash=state_hash,
    )


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------


def _code_spec_to_canonical_dict(spec: CodeSpec) -> dict:
    """Convert a CodeSpec to a canonical dict for serialization."""
    return {
        "code_id": spec.code_id,
        "distance": spec.distance,
        "family": spec.family,
        "logical_qubits": spec.logical_qubits,
        "metadata": dict(sorted(spec.metadata.items())),
        "physical_qubits": spec.physical_qubits,
        "stabilizer_count": spec.stabilizer_count,
    }


def _canonical_json(obj: Any) -> str:
    """Produce canonical JSON: sorted keys, compact separators."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def compute_code_registry_hash(registry: CodeZooRegistry) -> str:
    """Compute a deterministic SHA-256 hash of a CodeZooRegistry.

    Uses canonical JSON serialization of codes + registry_version.
    Same registry always produces the same hash.
    """
    canonical = {
        "codes": [_code_spec_to_canonical_dict(s) for s in registry.codes],
        "registry_version": registry.registry_version,
    }
    payload = _canonical_json(canonical)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Snapshot integration
# ---------------------------------------------------------------------------


def build_snapshot_from_code_registry(
    registry: CodeZooRegistry, policy_id: str,
) -> "ControllerSnapshot":
    """Build a ControllerSnapshot from a CodeZooRegistry.

    Integrates with ``qec.ai.controller_snapshot_schema``.

    Parameters
    ----------
    registry
        A validated CodeZooRegistry.
    policy_id
        Deterministic policy identifier string.
    """
    from qec.ai.controller_snapshot_schema import (
        SCHEMA_VERSION,
        ControllerSnapshot,
        _canonical_json,
    )

    canonical = {
        "codes": [_code_spec_to_canonical_dict(s) for s in registry.codes],
        "registry_version": registry.registry_version,
        "type": "CodeZooRegistry",
    }
    payload = _canonical_json(canonical)
    state_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()

    # Evidence score: ratio of codes with distance >= 3 to total codes
    n_total = len(registry.codes)
    if n_total > 0:
        n_good = sum(1 for s in registry.codes if s.distance >= 3)
        evidence = n_good / n_total
    else:
        evidence = 0.0

    invariant_passed = (
        len(registry.codes) > 0
        and registry.state_hash == compute_code_registry_hash(registry)
    )

    return ControllerSnapshot(
        state_hash=state_hash,
        policy_id=policy_id,
        evidence_score=round(evidence, 15),
        invariant_passed=invariant_passed,
        timestamp_index=len(registry.codes),
        schema_version=SCHEMA_VERSION,
        payload_json=payload,
    )
