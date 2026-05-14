from __future__ import annotations

from dataclasses import dataclass
from importlib import metadata as importlib_metadata
from importlib import util as importlib_util
import hashlib
import json
import re
from typing import Any

_SCHEMA_VERSION = "HEAVY_DEPENDENCY_DISCOVERY_V1"
_DISCOVERY_MODE = "STATIC_HEAVY_DEPENDENCY_INVENTORY"
_SOURCE_POLICY = "AUTHORITATIVE_UPSTREAM_REQUIRED"
_MAX_DEPENDENCY_NAME_LENGTH = 64
_MAX_MODULE_NAME_LENGTH = 128
_MAX_VERSION_LENGTH = 128
_MAX_DEPENDENCY_TARGETS = 32

_ALLOWED_CATEGORIES = {
    "NUMERIC_CORE",
    "SCIENTIFIC_CORE",
    "DATAFRAME",
    "VISUALIZATION",
    "QUANTUM_SIMULATION",
    "QUANTUM_BACKEND",
    "AUDIO_MIDI",
    "INTERNAL_QEC",
    "EXTERNAL_QEC",
}
_ALLOWED_AVAILABILITY_STATUSES = {
    "AVAILABLE",
    "UNAVAILABLE",
    "NOT_PROBED",
    "BLOCKED_BY_POLICY",
    "INTERNAL_AVAILABLE",
}
_ALLOWED_PROBE_MODES = {"EXPLICIT", "IMPORTLIB_FIND_SPEC", "IMPORT_METADATA", "INTERNAL_MODULE"}
_ALLOWED_SOURCE_POLICIES = {
    "INTERNAL_REPOSITORY",
    "AUTHORITATIVE_UPSTREAM_REQUIRED",
    "NORMALIZED_UPSTREAM_APPROVED",
    "UNNORMALIZED_EXTERNAL_BLOCKED",
}
_HASH_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class HeavyDependencyTarget:
    dependency_name: str
    import_name: str
    category: str
    source_policy: str
    optional: bool
    notes: tuple[str, ...]
    target_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "dependency_name": self.dependency_name,
            "import_name": self.import_name,
            "category": self.category,
            "source_policy": self.source_policy,
            "optional": self.optional,
            "notes": list(self.notes),
            "target_hash": self.target_hash,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")


@dataclass(frozen=True)
class HeavyDependencyProbeResult:
    dependency_name: str
    import_name: str
    availability_status: str
    probe_mode: str
    version: str | None
    version_source: str | None
    normalized_source: bool
    policy_status: str
    probe_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "dependency_name": self.dependency_name,
            "import_name": self.import_name,
            "availability_status": self.availability_status,
            "probe_mode": self.probe_mode,
            "version": self.version,
            "version_source": self.version_source,
            "normalized_source": self.normalized_source,
            "policy_status": self.policy_status,
            "probe_hash": self.probe_hash,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")


@dataclass(frozen=True)
class HeavyDependencyDiscoveryManifest:
    schema_version: str
    discovery_mode: str
    source_policy: str
    targets: tuple[HeavyDependencyTarget, ...]
    probe_results: tuple[HeavyDependencyProbeResult, ...]
    target_count: int
    available_count: int
    unavailable_count: int
    not_probed_count: int
    blocked_by_policy_count: int
    internal_available_count: int
    first_target_hash: str
    final_target_hash: str
    first_probe_hash: str
    final_probe_hash: str
    heavy_dependency_discovery_manifest_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "discovery_mode": self.discovery_mode,
            "source_policy": self.source_policy,
            "targets": [t.to_dict() for t in self.targets],
            "probe_results": [p.to_dict() for p in self.probe_results],
            "target_count": self.target_count,
            "available_count": self.available_count,
            "unavailable_count": self.unavailable_count,
            "not_probed_count": self.not_probed_count,
            "blocked_by_policy_count": self.blocked_by_policy_count,
            "internal_available_count": self.internal_available_count,
            "first_target_hash": self.first_target_hash,
            "final_target_hash": self.final_target_hash,
            "first_probe_hash": self.first_probe_hash,
            "final_probe_hash": self.final_probe_hash,
            "heavy_dependency_discovery_manifest_hash": self.heavy_dependency_discovery_manifest_hash,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")


def _canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _hash_payload(payload: Any) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _validate_hash_format(value: str) -> None:
    if not isinstance(value, str) or _HASH_RE.fullmatch(value) is None:
        raise ValueError("INVALID_HASH_FORMAT")


def _base_target_payload(target: HeavyDependencyTarget) -> dict[str, Any]:
    return {
        "dependency_name": target.dependency_name,
        "import_name": target.import_name,
        "category": target.category,
        "source_policy": target.source_policy,
        "optional": target.optional,
        "notes": list(target.notes),
    }


def _base_probe_payload(result: HeavyDependencyProbeResult) -> dict[str, Any]:
    return {
        "dependency_name": result.dependency_name,
        "import_name": result.import_name,
        "availability_status": result.availability_status,
        "probe_mode": result.probe_mode,
        "version": result.version,
        "version_source": result.version_source,
        "normalized_source": result.normalized_source,
        "policy_status": result.policy_status,
    }


def _registry_defs() -> tuple[tuple[str, str, str, str, bool, tuple[str, ...]], ...]:
    return (
        ("numpy", "numpy", "NUMERIC_CORE", "AUTHORITATIVE_UPSTREAM_REQUIRED", False, ()),
        ("scipy", "scipy", "SCIENTIFIC_CORE", "AUTHORITATIVE_UPSTREAM_REQUIRED", True, ()),
        ("pandas", "pandas", "DATAFRAME", "AUTHORITATIVE_UPSTREAM_REQUIRED", True, ()),
        ("matplotlib", "matplotlib", "VISUALIZATION", "AUTHORITATIVE_UPSTREAM_REQUIRED", True, ()),
        ("qutip", "qutip", "QUANTUM_SIMULATION", "AUTHORITATIVE_UPSTREAM_REQUIRED", True, ()),
        ("qiskit", "qiskit", "QUANTUM_BACKEND", "AUTHORITATIVE_UPSTREAM_REQUIRED", True, ()),
        ("qiskit_aer", "qiskit_aer", "QUANTUM_BACKEND", "AUTHORITATIVE_UPSTREAM_REQUIRED", True, ()),
        ("stim", "stim", "QUANTUM_SIMULATION", "AUTHORITATIVE_UPSTREAM_REQUIRED", True, ()),
        ("pymatching", "pymatching", "QUANTUM_SIMULATION", "AUTHORITATIVE_UPSTREAM_REQUIRED", True, ()),
        ("mido", "mido", "AUDIO_MIDI", "AUTHORITATIVE_UPSTREAM_REQUIRED", True, ()),
        ("qldpc_internal", "qldpc.css_code", "INTERNAL_QEC", "INTERNAL_REPOSITORY", False, ("in-repo deterministic surface",)),
        ("qldpc_external", "qldpc", "EXTERNAL_QEC", "UNNORMALIZED_EXTERNAL_BLOCKED", True, ("requires normalization",)),
    )


def get_heavy_dependency_targets() -> tuple[HeavyDependencyTarget, ...]:
    return tuple(build_heavy_dependency_target(*d) for d in _registry_defs())


def build_heavy_dependency_target(
    dependency_name: str,
    import_name: str,
    category: str,
    source_policy: str,
    optional: bool,
    notes: tuple[str, ...] = (),
) -> HeavyDependencyTarget:
    if dependency_name not in {d[0] for d in _registry_defs()}:
        raise ValueError("INVALID_DEPENDENCY_NAME")
    if category not in _ALLOWED_CATEGORIES:
        raise ValueError("INVALID_CATEGORY")
    if source_policy not in _ALLOWED_SOURCE_POLICIES:
        raise ValueError("INVALID_SOURCE_POLICY")
    if not isinstance(optional, bool):
        raise ValueError("INVALID_INPUT")
    if not isinstance(dependency_name, str) or len(dependency_name) > _MAX_DEPENDENCY_NAME_LENGTH:
        raise ValueError("INVALID_DEPENDENCY_NAME")
    if not isinstance(import_name, str) or len(import_name) > _MAX_MODULE_NAME_LENGTH:
        raise ValueError("INVALID_INPUT")
    if not isinstance(notes, tuple) or any((not isinstance(n, str) for n in notes)):
        raise ValueError("INVALID_INPUT")
    target = HeavyDependencyTarget(dependency_name, import_name, category, source_policy, optional, notes, "")
    target_hash = _hash_payload(_base_target_payload(target))
    return HeavyDependencyTarget(
        dependency_name=target.dependency_name,
        import_name=target.import_name,
        category=target.category,
        source_policy=target.source_policy,
        optional=target.optional,
        notes=target.notes,
        target_hash=target_hash,
    )


def build_probe_result(dependency_name: str, availability_status: str, *, version: str | None = None,
                       version_source: str | None = None, probe_mode: str = "EXPLICIT",
                       normalized_source: bool = False, policy_status: str | None = None) -> HeavyDependencyProbeResult:
    targets = {t.dependency_name: t for t in get_heavy_dependency_targets()}
    if dependency_name not in targets:
        raise ValueError("INVALID_DEPENDENCY_NAME")
    if availability_status not in _ALLOWED_AVAILABILITY_STATUSES:
        raise ValueError("INVALID_AVAILABILITY_STATUS")
    if probe_mode not in _ALLOWED_PROBE_MODES:
        raise ValueError("INVALID_PROBE_MODE")
    if policy_status is None:
        policy_status = targets[dependency_name].source_policy
    if policy_status not in _ALLOWED_SOURCE_POLICIES:
        raise ValueError("INVALID_SOURCE_POLICY")
    if version is not None and (not isinstance(version, str) or len(version) > _MAX_VERSION_LENGTH):
        raise ValueError("INVALID_INPUT")
    if version_source is not None and not isinstance(version_source, str):
        raise ValueError("INVALID_INPUT")
    if not isinstance(normalized_source, bool):
        raise ValueError("INVALID_INPUT")
    result = HeavyDependencyProbeResult(
        dependency_name=dependency_name,
        import_name=targets[dependency_name].import_name,
        availability_status=availability_status,
        probe_mode=probe_mode,
        version=version,
        version_source=version_source,
        normalized_source=normalized_source,
        policy_status=policy_status,
        probe_hash="",
    )
    return HeavyDependencyProbeResult(
        dependency_name=result.dependency_name,
        import_name=result.import_name,
        availability_status=result.availability_status,
        probe_mode=result.probe_mode,
        version=result.version,
        version_source=result.version_source,
        normalized_source=result.normalized_source,
        policy_status=result.policy_status,
        probe_hash=_hash_payload(_base_probe_payload(result)),
    )


def build_heavy_dependency_discovery_manifest(probe_results: tuple[HeavyDependencyProbeResult, ...] | list[HeavyDependencyProbeResult]) -> HeavyDependencyDiscoveryManifest:
    targets = get_heavy_dependency_targets()
    if len(targets) > _MAX_DEPENDENCY_TARGETS:
        raise ValueError("INVALID_INPUT")
    if not isinstance(probe_results, (tuple, list)):
        raise ValueError("INVALID_INPUT")
    expected_order = [t.dependency_name for t in targets]
    if len(probe_results) != len(expected_order):
        raise ValueError("DISCOVERY_COUNT_MISMATCH")
    seen: set[str] = set()
    for i, p in enumerate(probe_results):
        validate_heavy_dependency_probe_result(p)
        if p.dependency_name in seen:
            raise ValueError("DUPLICATE_DEPENDENCY")
        seen.add(p.dependency_name)
        if p.dependency_name != expected_order[i]:
            raise ValueError("DEPENDENCY_ORDER_MISMATCH")
    missing = [name for name in expected_order if name not in seen]
    if missing:
        raise ValueError("MISSING_DEPENDENCY")
    counts = {k: 0 for k in _ALLOWED_AVAILABILITY_STATUSES}
    for p in probe_results:
        counts[p.availability_status] += 1
    manifest = HeavyDependencyDiscoveryManifest(
        schema_version=_SCHEMA_VERSION,
        discovery_mode=_DISCOVERY_MODE,
        source_policy=_SOURCE_POLICY,
        targets=targets,
        probe_results=tuple(probe_results),
        target_count=len(targets),
        available_count=counts["AVAILABLE"],
        unavailable_count=counts["UNAVAILABLE"],
        not_probed_count=counts["NOT_PROBED"],
        blocked_by_policy_count=counts["BLOCKED_BY_POLICY"],
        internal_available_count=counts["INTERNAL_AVAILABLE"],
        first_target_hash=targets[0].target_hash,
        final_target_hash=targets[-1].target_hash,
        first_probe_hash=probe_results[0].probe_hash,
        final_probe_hash=probe_results[-1].probe_hash,
        heavy_dependency_discovery_manifest_hash="",
    )
    payload = manifest.to_dict()
    payload.pop("heavy_dependency_discovery_manifest_hash")
    return HeavyDependencyDiscoveryManifest(
        schema_version=manifest.schema_version,
        discovery_mode=manifest.discovery_mode,
        source_policy=manifest.source_policy,
        targets=manifest.targets,
        probe_results=manifest.probe_results,
        target_count=manifest.target_count,
        available_count=manifest.available_count,
        unavailable_count=manifest.unavailable_count,
        not_probed_count=manifest.not_probed_count,
        blocked_by_policy_count=manifest.blocked_by_policy_count,
        internal_available_count=manifest.internal_available_count,
        first_target_hash=manifest.first_target_hash,
        final_target_hash=manifest.final_target_hash,
        first_probe_hash=manifest.first_probe_hash,
        final_probe_hash=manifest.final_probe_hash,
        heavy_dependency_discovery_manifest_hash=_hash_payload(payload),
    )


def build_default_unprobed_manifest() -> HeavyDependencyDiscoveryManifest:
    probes = [build_probe_result(t.dependency_name, "NOT_PROBED") for t in get_heavy_dependency_targets()]
    return build_heavy_dependency_discovery_manifest(probes)


def probe_current_environment() -> HeavyDependencyDiscoveryManifest:
    probes: list[HeavyDependencyProbeResult] = []
    for target in get_heavy_dependency_targets():
        if target.dependency_name == "qldpc_external":
            probes.append(build_probe_result("qldpc_external", "BLOCKED_BY_POLICY", probe_mode="EXPLICIT"))
            continue
        spec = importlib_util.find_spec(target.import_name)
        if spec is None:
            probes.append(build_probe_result(target.dependency_name, "UNAVAILABLE", probe_mode="IMPORTLIB_FIND_SPEC"))
            continue
        if target.dependency_name == "qldpc_internal":
            probes.append(build_probe_result("qldpc_internal", "INTERNAL_AVAILABLE", probe_mode="INTERNAL_MODULE"))
            continue
        try:
            version = importlib_metadata.version(target.dependency_name)
            probes.append(build_probe_result(target.dependency_name, "AVAILABLE", version=version, version_source="importlib.metadata.version", probe_mode="IMPORT_METADATA"))
        except importlib_metadata.PackageNotFoundError:
            probes.append(build_probe_result(target.dependency_name, "AVAILABLE", version=None, version_source=None, probe_mode="IMPORTLIB_FIND_SPEC"))
    return build_heavy_dependency_discovery_manifest(probes)


def validate_heavy_dependency_target(target: HeavyDependencyTarget) -> bool:
    if not isinstance(target, HeavyDependencyTarget):
        raise ValueError("INVALID_INPUT")
    _validate_hash_format(target.target_hash)
    rebuilt = build_heavy_dependency_target(target.dependency_name, target.import_name, target.category, target.source_policy, target.optional, target.notes)
    if rebuilt.target_hash != target.target_hash:
        raise ValueError("HASH_MISMATCH")
    return True


def validate_heavy_dependency_probe_result(result: HeavyDependencyProbeResult) -> bool:
    if not isinstance(result, HeavyDependencyProbeResult):
        raise ValueError("INVALID_INPUT")
    _validate_hash_format(result.probe_hash)
    rebuilt = build_probe_result(
        result.dependency_name,
        result.availability_status,
        version=result.version,
        version_source=result.version_source,
        probe_mode=result.probe_mode,
        normalized_source=result.normalized_source,
        policy_status=result.policy_status,
    )
    if rebuilt.probe_hash != result.probe_hash:
        raise ValueError("HASH_MISMATCH")
    return True


def validate_heavy_dependency_discovery_manifest(manifest: HeavyDependencyDiscoveryManifest) -> bool:
    if not isinstance(manifest, HeavyDependencyDiscoveryManifest):
        raise ValueError("INVALID_INPUT")
    _validate_hash_format(manifest.heavy_dependency_discovery_manifest_hash)
    int_fields = [manifest.target_count, manifest.available_count, manifest.unavailable_count, manifest.not_probed_count, manifest.blocked_by_policy_count, manifest.internal_available_count]
    if any((type(v) is not int for v in int_fields)):
        raise ValueError("INVALID_INPUT")
    rebuilt = build_heavy_dependency_discovery_manifest(manifest.probe_results)
    if manifest.to_dict() != rebuilt.to_dict():
        if manifest.heavy_dependency_discovery_manifest_hash != rebuilt.heavy_dependency_discovery_manifest_hash:
            raise ValueError("HASH_MISMATCH")
        raise ValueError("DISCOVERY_COUNT_MISMATCH")
    return True


def validate_manifest_matches_probe_results(
    manifest: HeavyDependencyDiscoveryManifest,
    probe_results: tuple[HeavyDependencyProbeResult, ...] | list[HeavyDependencyProbeResult],
) -> bool:
    if not isinstance(manifest, HeavyDependencyDiscoveryManifest):
        raise ValueError("INVALID_INPUT")
    expected = build_heavy_dependency_discovery_manifest(probe_results)
    if manifest.to_dict() != expected.to_dict():
        raise ValueError("DISCOVERY_MANIFEST_MISMATCH")
    return True
