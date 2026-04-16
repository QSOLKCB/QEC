# SPDX-License-Identifier: MIT
"""v138.1.2 — Multi-Code Orchestration Simulator.

Deterministic supervisory simulator that coordinates multiple simulation lanes
under a canonical orchestration manifest. This module is additive-only and
explicitly does not execute decoder logic.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Dict, Mapping, Sequence, Tuple

MULTI_CODE_ORCHESTRATION_SIMULATOR_VERSION = "v138.1.2"

SUPPORTED_CODE_FAMILIES: Tuple[str, ...] = (
    "surface_code",
    "qldpc",
    "qutrit_ldpc",
    "bosonic_concat",
    "synthetic_control",
)
SUPPORTED_ORCHESTRATION_POLICIES: Tuple[str, ...] = (
    "strict_deterministic",
    "benchmark_supervisor",
)
SUPPORTED_LANE_EXECUTION_POLICIES: Tuple[str, ...] = (
    "simulate_only",
    "deterministic_stub",
)

_HEX_64 = frozenset("0123456789abcdef")


class MultiCodeOrchestrationValidationError(ValueError):
    """Raised when orchestration data violates deterministic schema."""



def _canonical_json(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)



def _stable_hash(data: Any) -> str:
    return hashlib.sha256(_canonical_json(data).encode("utf-8")).hexdigest()



def _is_hex_sha256(value: str) -> bool:
    return len(value) == 64 and all(ch in _HEX_64 for ch in value)



def _normalize_text(value: Any, *, field: str) -> str:
    if value is None:
        raise MultiCodeOrchestrationValidationError(f"{field} must be non-empty")
    text = str(value).strip()
    if not text:
        raise MultiCodeOrchestrationValidationError(f"{field} must be non-empty")
    return text



def _normalize_hash(value: Any, *, field: str) -> str:
    result = _normalize_text(value, field=field).lower()
    if not _is_hex_sha256(result):
        raise MultiCodeOrchestrationValidationError(f"{field} must be 64-char lowercase SHA-256 hex")
    return result



def _canonicalize_value(value: Any, *, field: str) -> Any:
    if value is None or isinstance(value, (bool, str, int)):
        return value
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            raise MultiCodeOrchestrationValidationError(f"{field} contains non-canonical numeric value")
        return float(value)
    if isinstance(value, Mapping):
        normalized: Dict[str, Any] = {}
        for raw_key in sorted(value.keys(), key=lambda x: str(x)):
            key = str(raw_key)
            if key in normalized:
                raise MultiCodeOrchestrationValidationError(f"{field} contains duplicate canonical key: {key!r}")
            normalized[key] = _canonicalize_value(value[raw_key], field=f"{field}.{key}")
        return normalized
    if isinstance(value, (list, tuple)):
        return [_canonicalize_value(item, field=field) for item in value]
    raise MultiCodeOrchestrationValidationError(f"{field} contains unsupported type: {type(value).__name__}")



def _lane_sort_key(lane: "SimulationLane") -> Tuple[Any, ...]:
    return (int(lane.priority_rank), lane.lane_id)



@dataclass(frozen=True)
class SimulationLane:
    lane_id: str
    code_family: str
    topology_family: str
    package_hash: str
    priority_rank: int
    execution_policy: str
    metadata: Mapping[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "lane_id": self.lane_id,
            "code_family": self.code_family,
            "topology_family": self.topology_family,
            "package_hash": self.package_hash,
            "priority_rank": int(self.priority_rank),
            "execution_policy": self.execution_policy,
            "metadata": _canonicalize_value(dict(self.metadata), field="lane.metadata"),
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def stable_hash(self) -> str:
        return _stable_hash(self.to_dict())


@dataclass(frozen=True)
class MultiCodeOrchestrationManifest:
    orchestration_version: str
    experiment_id: str
    orchestration_policy: str
    lane_order: Tuple[str, ...]
    upstream_package_hashes: Tuple[str, ...]
    seed: int | str
    policy_flags: Tuple[str, ...]
    notes: Tuple[str, ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "orchestration_version": self.orchestration_version,
            "experiment_id": self.experiment_id,
            "orchestration_policy": self.orchestration_policy,
            "lane_order": list(self.lane_order),
            "upstream_package_hashes": list(self.upstream_package_hashes),
            "seed": self.seed,
            "policy_flags": list(self.policy_flags),
            "notes": list(self.notes),
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def stable_hash(self) -> str:
        return _stable_hash(self.to_dict())


@dataclass(frozen=True)
class SimulationLaneResult:
    lane_id: str
    package_hash: str
    execution_hash: str
    status: str
    topology_stability_score: float
    replay_identity: str
    metadata: Mapping[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "lane_id": self.lane_id,
            "package_hash": self.package_hash,
            "execution_hash": self.execution_hash,
            "status": self.status,
            "topology_stability_score": float(self.topology_stability_score),
            "replay_identity": self.replay_identity,
            "metadata": _canonicalize_value(dict(self.metadata), field="result.metadata"),
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def stable_hash(self) -> str:
        return _stable_hash(self.to_dict())


@dataclass(frozen=True)
class MultiCodeOrchestrationReceipt:
    manifest_hash: str
    lane_set_hash: str
    execution_hash: str
    validation_passed: bool
    lane_count: int
    receipt_hash: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "manifest_hash": self.manifest_hash,
            "lane_set_hash": self.lane_set_hash,
            "execution_hash": self.execution_hash,
            "validation_passed": self.validation_passed,
            "lane_count": int(self.lane_count),
            "receipt_hash": self.receipt_hash,
        }

    def to_hash_payload_dict(self) -> Dict[str, Any]:
        return {
            "manifest_hash": self.manifest_hash,
            "lane_set_hash": self.lane_set_hash,
            "execution_hash": self.execution_hash,
            "lane_count": int(self.lane_count),
            "validation_passed": bool(self.validation_passed),
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def stable_hash(self) -> str:
        return _stable_hash(self.to_hash_payload_dict())


@dataclass(frozen=True)
class MultiCodeOrchestrationSimulation:
    manifest: MultiCodeOrchestrationManifest
    lanes: Tuple[SimulationLane, ...]
    results: Tuple[SimulationLaneResult, ...]
    receipt: MultiCodeOrchestrationReceipt

    def to_dict(self) -> Dict[str, Any]:
        return {
            "manifest": self.manifest.to_dict(),
            "lanes": [lane.to_dict() for lane in self.lanes],
            "results": [result.to_dict() for result in self.results],
            "receipt": self.receipt.to_dict(),
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def stable_hash(self) -> str:
        return _stable_hash(self.to_dict())


@dataclass(frozen=True)
class MultiCodeOrchestrationValidationReport:
    valid: bool
    errors: Tuple[str, ...]



def _lane_from_any(raw: SimulationLane | Mapping[str, Any]) -> SimulationLane:
    if isinstance(raw, SimulationLane):
        raw = raw.to_dict()
    if not isinstance(raw, Mapping):
        raise MultiCodeOrchestrationValidationError("lane must be mapping or SimulationLane")

    code_family = _normalize_text(raw.get("code_family"), field="lane.code_family")
    if code_family not in SUPPORTED_CODE_FAMILIES:
        raise MultiCodeOrchestrationValidationError(f"unsupported lane.code_family: {code_family!r}")

    execution_policy = _normalize_text(raw.get("execution_policy"), field="lane.execution_policy")
    if execution_policy not in SUPPORTED_LANE_EXECUTION_POLICIES:
        raise MultiCodeOrchestrationValidationError(f"unsupported lane.execution_policy: {execution_policy!r}")

    priority_rank = raw.get("priority_rank")
    if isinstance(priority_rank, bool) or not isinstance(priority_rank, int):
        raise MultiCodeOrchestrationValidationError("lane.priority_rank must be int")

    return SimulationLane(
        lane_id=_normalize_text(raw.get("lane_id"), field="lane.lane_id"),
        code_family=code_family,
        topology_family=_normalize_text(raw.get("topology_family"), field="lane.topology_family"),
        package_hash=_normalize_hash(raw.get("package_hash"), field="lane.package_hash"),
        priority_rank=int(priority_rank),
        execution_policy=execution_policy,
        metadata=_canonicalize_value(dict(raw.get("metadata", {})), field="lane.metadata"),
    )



def _manifest_from_any(raw: MultiCodeOrchestrationManifest | Mapping[str, Any]) -> MultiCodeOrchestrationManifest:
    if isinstance(raw, MultiCodeOrchestrationManifest):
        raw = raw.to_dict()
    if not isinstance(raw, Mapping):
        raise MultiCodeOrchestrationValidationError("manifest must be mapping or MultiCodeOrchestrationManifest")

    seed = raw.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, (int, str)):
        raise MultiCodeOrchestrationValidationError("manifest.seed must be int or str")

    orchestration_policy = _normalize_text(raw.get("orchestration_policy"), field="manifest.orchestration_policy")
    if orchestration_policy not in SUPPORTED_ORCHESTRATION_POLICIES:
        raise MultiCodeOrchestrationValidationError(
            f"unsupported manifest.orchestration_policy: {orchestration_policy!r}"
        )

    orchestration_version = _normalize_text(
        raw.get("orchestration_version", MULTI_CODE_ORCHESTRATION_SIMULATOR_VERSION),
        field="manifest.orchestration_version",
    )
    if orchestration_version != MULTI_CODE_ORCHESTRATION_SIMULATOR_VERSION:
        raise MultiCodeOrchestrationValidationError(
            "unsupported manifest.orchestration_version: "
            f"{orchestration_version!r}; expected "
            f"{MULTI_CODE_ORCHESTRATION_SIMULATOR_VERSION!r}"
        )

    return MultiCodeOrchestrationManifest(
        orchestration_version=orchestration_version,
        experiment_id=_normalize_text(raw.get("experiment_id"), field="manifest.experiment_id"),
        orchestration_policy=orchestration_policy,
        lane_order=tuple(_normalize_text(item, field="manifest.lane_order") for item in tuple(raw.get("lane_order", ()))),
        upstream_package_hashes=tuple(
            sorted(_normalize_hash(item, field="manifest.upstream_package_hashes") for item in tuple(raw.get("upstream_package_hashes", ())))
        ),
        seed=str(seed) if isinstance(seed, str) else int(seed),
        policy_flags=tuple(sorted(_normalize_text(item, field="manifest.policy_flags") for item in tuple(raw.get("policy_flags", ())))),
        notes=tuple(sorted(_normalize_text(item, field="manifest.notes") for item in tuple(raw.get("notes", ())))),
    )



def simulate_lane_execution(
    *,
    lane: SimulationLane,
    manifest: MultiCodeOrchestrationManifest,
    manifest_hash: str,
    seeded_policy_inputs: Mapping[str, Any] | None = None,
) -> SimulationLaneResult:
    """Compute deterministic simulation-only lane execution artifacts."""
    policy_inputs = _canonicalize_value(dict(seeded_policy_inputs or {}), field="seeded_policy_inputs")

    lane_execution_payload = {
        "manifest_hash": manifest_hash,
        "seed": manifest.seed,
        "policy": manifest.orchestration_policy,
        "lane_id": lane.lane_id,
        "package_hash": lane.package_hash,
        "code_family": lane.code_family,
        "topology_family": lane.topology_family,
        "execution_policy": lane.execution_policy,
        "lane_metadata_hash": _stable_hash(lane.metadata),
        "policy_inputs": policy_inputs,
    }
    execution_hash = _stable_hash(lane_execution_payload)
    score_int = int(execution_hash[:16], 16)
    topology_stability_score = float(score_int / float(0xFFFFFFFFFFFFFFFF))

    replay_identity = orchestration_replay_identity(
        manifest_hash=manifest_hash,
        lane_id=lane.lane_id,
        package_hash=lane.package_hash,
        lane_execution_hash=execution_hash,
    )
    return SimulationLaneResult(
        lane_id=lane.lane_id,
        package_hash=lane.package_hash,
        execution_hash=execution_hash,
        status="simulated",
        topology_stability_score=topology_stability_score,
        replay_identity=replay_identity,
        metadata={
            "policy_input_hash": _stable_hash(policy_inputs),
            "lane_hash": lane.stable_hash(),
            "manifest_policy": manifest.orchestration_policy,
        },
    )



def build_multi_code_orchestration(
    *,
    manifest: MultiCodeOrchestrationManifest | Mapping[str, Any],
    lanes: Sequence[SimulationLane | Mapping[str, Any]],
    package_hashes: Sequence[str] = (),
    seeded_policy_inputs: Mapping[str, Any] | None = None,
    allow_duplicate_package_hashes: bool = False,
) -> MultiCodeOrchestrationSimulation:
    """Build deterministic multi-code orchestration simulation object."""
    normalized_lanes = tuple(sorted((_lane_from_any(lane) for lane in lanes), key=_lane_sort_key))
    canonical_lane_order = tuple(lane.lane_id for lane in normalized_lanes)

    normalized_manifest = _manifest_from_any(manifest)
    normalized_package_hashes = tuple(_normalize_hash(item, field="package_hashes") for item in package_hashes)
    normalized_manifest = MultiCodeOrchestrationManifest(
        orchestration_version=normalized_manifest.orchestration_version,
        experiment_id=normalized_manifest.experiment_id,
        orchestration_policy=normalized_manifest.orchestration_policy,
        lane_order=canonical_lane_order,
        upstream_package_hashes=tuple(
            sorted(
                {
                    *normalized_manifest.upstream_package_hashes,
                    *normalized_package_hashes,
                }
            )
        ),
        seed=normalized_manifest.seed,
        policy_flags=normalized_manifest.policy_flags,
        notes=normalized_manifest.notes,
    )

    manifest_hash = normalized_manifest.stable_hash()
    lane_set_hash = _stable_hash([lane.to_dict() for lane in normalized_lanes])

    results = tuple(
        simulate_lane_execution(
            lane=lane,
            manifest=normalized_manifest,
            manifest_hash=manifest_hash,
            seeded_policy_inputs=seeded_policy_inputs,
        )
        for lane in normalized_lanes
    )

    execution_hash = _stable_hash(
        {
            "manifest_hash": manifest_hash,
            "lane_set_hash": lane_set_hash,
            "lane_execution_hashes": [result.execution_hash for result in results],
            "lane_policies": [lane.execution_policy for lane in normalized_lanes],
            "package_lineage": list(normalized_manifest.upstream_package_hashes),
        }
    )

    receipt_base = MultiCodeOrchestrationReceipt(
        manifest_hash=manifest_hash,
        lane_set_hash=lane_set_hash,
        execution_hash=execution_hash,
        validation_passed=True,
        lane_count=len(normalized_lanes),
        receipt_hash="",
    )
    receipt_hash = _stable_hash(receipt_base.to_hash_payload_dict())
    simulation = MultiCodeOrchestrationSimulation(
        manifest=normalized_manifest,
        lanes=normalized_lanes,
        results=results,
        receipt=MultiCodeOrchestrationReceipt(
            manifest_hash=manifest_hash,
            lane_set_hash=lane_set_hash,
            execution_hash=execution_hash,
            validation_passed=True,
            lane_count=len(normalized_lanes),
            receipt_hash=receipt_hash,
        ),
    )

    report = validate_multi_code_orchestration(
        simulation,
        allow_duplicate_package_hashes=allow_duplicate_package_hashes,
    )
    if not report.valid:
        raise MultiCodeOrchestrationValidationError("; ".join(report.errors))
    return simulation



def validate_multi_code_orchestration(
    simulation: MultiCodeOrchestrationSimulation,
    *,
    allow_duplicate_package_hashes: bool = False,
) -> MultiCodeOrchestrationValidationReport:
    """Validate canonical orchestration schema + hash consistency."""
    errors = []

    lane_ids = [lane.lane_id for lane in simulation.lanes]
    if len(set(lane_ids)) != len(lane_ids):
        errors.append("duplicate lane ids are not allowed")

    package_hashes = [lane.package_hash for lane in simulation.lanes]
    if (not allow_duplicate_package_hashes) and len(set(package_hashes)) != len(package_hashes):
        errors.append("duplicate lane package hashes are not allowed")

    if simulation.manifest.lane_order != tuple(lane_ids):
        errors.append("manifest.lane_order must match canonical lane ordering")

    expected_lanes = tuple(sorted(simulation.lanes, key=_lane_sort_key))
    if expected_lanes != simulation.lanes:
        errors.append("lanes must be pre-sorted by priority_rank then lane_id")

    if simulation.manifest.orchestration_policy not in SUPPORTED_ORCHESTRATION_POLICIES:
        errors.append("manifest.orchestration_policy is invalid")

    for idx, lane in enumerate(simulation.lanes):
        if lane.code_family not in SUPPORTED_CODE_FAMILIES:
            errors.append(f"lanes[{idx}].code_family is invalid")
        if lane.execution_policy not in SUPPORTED_LANE_EXECUTION_POLICIES:
            errors.append(f"lanes[{idx}].execution_policy is invalid")
        if not _is_hex_sha256(lane.package_hash):
            errors.append(f"lanes[{idx}].package_hash must be 64-char lowercase SHA-256 hex")
        try:
            _canonicalize_value(dict(lane.metadata), field=f"lanes[{idx}].metadata")
        except MultiCodeOrchestrationValidationError as exc:
            errors.append(str(exc))

    for idx, value in enumerate(simulation.manifest.upstream_package_hashes):
        if not _is_hex_sha256(value):
            errors.append(f"manifest.upstream_package_hashes[{idx}] must be 64-char lowercase SHA-256 hex")

    hash_fields = (
        ("receipt.manifest_hash", simulation.receipt.manifest_hash),
        ("receipt.lane_set_hash", simulation.receipt.lane_set_hash),
        ("receipt.execution_hash", simulation.receipt.execution_hash),
        ("receipt.receipt_hash", simulation.receipt.receipt_hash),
    )
    for field, hash_value in hash_fields:
        if not _is_hex_sha256(hash_value):
            errors.append(f"{field} must be 64-char lowercase SHA-256 hex")

    for idx, result in enumerate(simulation.results):
        if result.status.strip() == "":
            errors.append(f"results[{idx}].status must be non-empty")
        if not _is_hex_sha256(result.package_hash):
            errors.append(f"results[{idx}].package_hash must be 64-char lowercase SHA-256 hex")
        elif result.package_hash != simulation.lanes[idx].package_hash:
            errors.append(f"results[{idx}].package_hash must match lanes[{idx}].package_hash")
        if not _is_hex_sha256(result.execution_hash):
            errors.append(f"results[{idx}].execution_hash must be 64-char lowercase SHA-256 hex")
        if math.isnan(result.topology_stability_score) or math.isinf(result.topology_stability_score):
            errors.append(f"results[{idx}].topology_stability_score must be finite")
        try:
            _canonicalize_value(dict(result.metadata), field=f"results[{idx}].metadata")
        except MultiCodeOrchestrationValidationError as exc:
            errors.append(str(exc))

    if tuple(result.lane_id for result in simulation.results) != tuple(lane_ids):
        errors.append("results must align to lane ordering")

    expected_manifest_hash = simulation.manifest.stable_hash()
    if simulation.receipt.manifest_hash != expected_manifest_hash:
        errors.append("receipt.manifest_hash mismatch")

    expected_lane_set_hash = _stable_hash([lane.to_dict() for lane in simulation.lanes])
    if simulation.receipt.lane_set_hash != expected_lane_set_hash:
        errors.append("receipt.lane_set_hash mismatch")

    expected_execution_hash = _stable_hash(
        {
            "manifest_hash": simulation.receipt.manifest_hash,
            "lane_set_hash": simulation.receipt.lane_set_hash,
            "lane_execution_hashes": [result.execution_hash for result in simulation.results],
            "lane_policies": [lane.execution_policy for lane in simulation.lanes],
            "package_lineage": list(simulation.manifest.upstream_package_hashes),
        }
    )
    if simulation.receipt.execution_hash != expected_execution_hash:
        errors.append("receipt.execution_hash mismatch")

    expected_receipt_hash = simulation.receipt.stable_hash()
    if simulation.receipt.receipt_hash != expected_receipt_hash:
        errors.append("receipt.receipt_hash mismatch")

    if simulation.receipt.lane_count != len(simulation.lanes):
        errors.append("receipt.lane_count mismatch")

    ordered_errors = tuple(sorted(errors))
    return MultiCodeOrchestrationValidationReport(valid=not ordered_errors, errors=ordered_errors)



def orchestration_replay_identity(
    *,
    manifest_hash: str,
    lane_id: str,
    package_hash: str,
    lane_execution_hash: str,
) -> str:
    """Expose deterministic replay identity for orchestration lineage handoff."""
    return _stable_hash(
        {
            "manifest_hash": _normalize_hash(manifest_hash, field="manifest_hash"),
            "lane_id": _normalize_text(lane_id, field="lane_id"),
            "package_hash": _normalize_hash(package_hash, field="package_hash"),
            "lane_execution_hash": _normalize_hash(lane_execution_hash, field="lane_execution_hash"),
            "schema": MULTI_CODE_ORCHESTRATION_SIMULATOR_VERSION,
        }
    )


__all__ = [
    "MULTI_CODE_ORCHESTRATION_SIMULATOR_VERSION",
    "SUPPORTED_CODE_FAMILIES",
    "SUPPORTED_ORCHESTRATION_POLICIES",
    "SUPPORTED_LANE_EXECUTION_POLICIES",
    "MultiCodeOrchestrationValidationError",
    "SimulationLane",
    "MultiCodeOrchestrationManifest",
    "SimulationLaneResult",
    "MultiCodeOrchestrationReceipt",
    "MultiCodeOrchestrationSimulation",
    "MultiCodeOrchestrationValidationReport",
    "build_multi_code_orchestration",
    "validate_multi_code_orchestration",
    "simulate_lane_execution",
    "orchestration_replay_identity",
]
