from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from typing import Any, Iterable

from .dependency_hotpath_receipts import (
    DependencyHotPathCandidate,
    DependencyImportAndHotPathReceipt,
    DependencyImportSite,
    validate_dependency_import_and_hotpath_receipt,
)
from .heavy_dependency_discovery import (
    HeavyDependencyDiscoveryManifest,
    get_heavy_dependency_targets,
    validate_heavy_dependency_discovery_manifest,
)

_SCHEMA_VERSION = "BACKEND_INVARIANT_CANDIDATE_V1"
_CANDIDATE_MODE = "STATIC_BACKEND_INVARIANT_CANDIDATE_DISCOVERY"
_MAX_INVARIANT_EVIDENCE = 4096
_MAX_INVARIANT_CANDIDATES = 2048
_MAX_SOURCE_PATH_LENGTH = 256
_MAX_REASON_LENGTH = 256
_MAX_INVARIANT_NAME_LENGTH = 128
_HASH_RE = re.compile(r"^[0-9a-f]{64}$")

_ALLOWED_EVIDENCE_KINDS = {"IMPORT_SITE_EVIDENCE", "HOTPATH_CANDIDATE_EVIDENCE", "DISCOVERY_STATUS_EVIDENCE", "POLICY_STATUS_EVIDENCE"}
_ALLOWED_INVARIANT_KINDS = {
    "IMPORT_PLACEMENT_INVARIANT", "TOP_LEVEL_IMPORT_BOUNDARY_INVARIANT", "REPEATED_IMPORT_SURFACE_INVARIANT",
    "QUANTUM_BACKEND_BOUNDARY_INVARIANT", "SPARSE_DENSE_BOUNDARY_INVARIANT", "PLOTTING_RENDER_BOUNDARY_INVARIANT",
    "DATAFRAME_SCHEMA_BOUNDARY_INVARIANT", "AUDIO_MIDI_BOUNDARY_INVARIANT", "INTERNAL_QEC_SURFACE_INVARIANT",
    "POLICY_BLOCKED_EXTERNAL_INVARIANT", "UNAVAILABLE_BACKEND_INVARIANT", "AVAILABLE_BACKEND_SURFACE_INVARIANT",
}
_ALLOWED_REVIEW_CLASSES = {
    "DISCOVERY_ONLY", "NEEDS_EQUIVALENCE_RECEIPT", "NEEDS_BENCHMARK_RECEIPT", "NEEDS_POLICY_NORMALIZATION",
    "BLOCKED_BY_POLICY", "SAFE_FOR_OPTIMIZATION_CONTRACT_REVIEW",
}
_ALLOWED_STATUSES = {"CANDIDATE_IDENTIFIED", "CANDIDATE_BLOCKED", "CANDIDATE_REQUIRES_REVIEW"}
_ALLOWED_REQUIRED_RECEIPTS = {
    "CrossBackendEquivalenceReceipt", "OptimizedQECBenchmarkReceipt", "UpstreamSourceNormalizationReceipt", "OptimizationContract", "NONE",
}

_REGISTRY = frozenset(x.dependency_name for x in get_heavy_dependency_targets())


@dataclass(frozen=True)
class BackendInvariantEvidence:
    evidence_index: int
    dependency_name: str
    evidence_kind: str
    source_path: str | None
    line_number: int | None
    import_site_hash: str | None
    hotpath_candidate_hash: str | None
    probe_hash: str | None
    reason: str
    evidence_hash: str

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")


@dataclass(frozen=True)
class BackendInvariantCandidate:
    candidate_index: int
    dependency_name: str
    invariant_name: str
    invariant_kind: str
    invariant_status: str
    review_class: str
    evidence_hashes: tuple[str, ...]
    source_paths: tuple[str, ...]
    reason: str
    required_next_receipt: str
    candidate_hash: str

    def to_dict(self) -> dict[str, Any]:
        d = self.__dict__.copy()
        d["evidence_hashes"] = list(self.evidence_hashes)
        d["source_paths"] = list(self.source_paths)
        return d

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")


@dataclass(frozen=True)
class BackendInvariantCandidateReceipt:
    schema_version: str
    candidate_mode: str
    discovery_manifest_hash: str
    dependency_hotpath_receipt_hash: str
    evidence_count: int
    candidate_count: int
    blocked_candidate_count: int
    equivalence_required_count: int
    benchmark_required_count: int
    policy_normalization_required_count: int
    evidence: tuple[BackendInvariantEvidence, ...]
    candidates: tuple[BackendInvariantCandidate, ...]
    first_evidence_hash: str
    final_evidence_hash: str
    first_candidate_hash: str
    final_candidate_hash: str
    backend_invariant_candidate_receipt_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version, "candidate_mode": self.candidate_mode,
            "discovery_manifest_hash": self.discovery_manifest_hash, "dependency_hotpath_receipt_hash": self.dependency_hotpath_receipt_hash,
            "evidence_count": self.evidence_count, "candidate_count": self.candidate_count,
            "blocked_candidate_count": self.blocked_candidate_count, "equivalence_required_count": self.equivalence_required_count,
            "benchmark_required_count": self.benchmark_required_count, "policy_normalization_required_count": self.policy_normalization_required_count,
            "evidence": [e.to_dict() for e in self.evidence], "candidates": [c.to_dict() for c in self.candidates],
            "first_evidence_hash": self.first_evidence_hash, "final_evidence_hash": self.final_evidence_hash,
            "first_candidate_hash": self.first_candidate_hash, "final_candidate_hash": self.final_candidate_hash,
            "backend_invariant_candidate_receipt_hash": self.backend_invariant_candidate_receipt_hash,
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


def _base_evidence_payload(e: BackendInvariantEvidence) -> dict[str, Any]:
    d = e.to_dict()
    d.pop("evidence_hash")
    return d


def _base_candidate_payload(c: BackendInvariantCandidate) -> dict[str, Any]:
    d = c.to_dict()
    d.pop("candidate_hash")
    return d


def build_backend_invariant_evidence(**kwargs: Any) -> BackendInvariantEvidence:
    payload = dict(kwargs)
    payload.pop("evidence_hash", None)
    e = BackendInvariantEvidence(evidence_hash="", **payload)
    validate_backend_invariant_evidence(e, allow_blank_hash=True)
    return BackendInvariantEvidence(**{**e.to_dict(), "evidence_hash": _hash_payload(_base_evidence_payload(e))})


def build_backend_invariant_candidate(**kwargs: Any) -> BackendInvariantCandidate:
    payload = dict(kwargs)
    payload.pop("candidate_hash", None)
    c = BackendInvariantCandidate(candidate_hash="", **payload)
    validate_backend_invariant_candidate(c, allow_blank_hash=True)
    return BackendInvariantCandidate(**{**c.to_dict(), "evidence_hashes": tuple(c.evidence_hashes), "source_paths": tuple(sorted(c.source_paths)), "candidate_hash": _hash_payload(_base_candidate_payload(c))})


def validate_backend_invariant_evidence(evidence: BackendInvariantEvidence, allow_blank_hash: bool = False) -> bool:
    if not isinstance(evidence, BackendInvariantEvidence):
        raise ValueError("INVALID_INPUT")
    if not isinstance(evidence.evidence_index, int) or isinstance(evidence.evidence_index, bool) or evidence.evidence_index < 0:
        raise ValueError("INVALID_INPUT")
    if evidence.dependency_name not in _REGISTRY:
        raise ValueError("INVALID_DEPENDENCY_NAME")
    if evidence.evidence_kind not in _ALLOWED_EVIDENCE_KINDS:
        raise ValueError("INVALID_EVIDENCE_KIND")
    if evidence.source_path is not None and (not isinstance(evidence.source_path, str) or not evidence.source_path or len(evidence.source_path) > _MAX_SOURCE_PATH_LENGTH or "\\" in evidence.source_path):
        raise ValueError("INVALID_INPUT")
    if evidence.line_number is not None and (not isinstance(evidence.line_number, int) or isinstance(evidence.line_number, bool) or evidence.line_number <= 0):
        raise ValueError("INVALID_INPUT")
    for h in (evidence.import_site_hash, evidence.hotpath_candidate_hash, evidence.probe_hash):
        if h is not None:
            _validate_hash_format(h)
    if not isinstance(evidence.reason, str) or len(evidence.reason) > _MAX_REASON_LENGTH:
        raise ValueError("INVALID_INPUT")
    expected = _hash_payload(_base_evidence_payload(evidence))
    if evidence.evidence_hash == "" and allow_blank_hash:
        return True
    _validate_hash_format(evidence.evidence_hash)
    if evidence.evidence_hash != expected:
        raise ValueError("HASH_MISMATCH")
    return True


def validate_backend_invariant_candidate(candidate: BackendInvariantCandidate, allow_blank_hash: bool = False) -> bool:
    if not isinstance(candidate, BackendInvariantCandidate):
        raise ValueError("INVALID_INPUT")
    if not isinstance(candidate.candidate_index, int) or isinstance(candidate.candidate_index, bool) or candidate.candidate_index < 0:
        raise ValueError("INVALID_INPUT")
    if candidate.dependency_name not in _REGISTRY:
        raise ValueError("INVALID_DEPENDENCY_NAME")
    if not isinstance(candidate.invariant_name, str) or not candidate.invariant_name or len(candidate.invariant_name) > _MAX_INVARIANT_NAME_LENGTH:
        raise ValueError("INVALID_INPUT")
    if candidate.invariant_kind not in _ALLOWED_INVARIANT_KINDS:
        raise ValueError("INVALID_INVARIANT_KIND")
    if candidate.invariant_status not in _ALLOWED_STATUSES:
        raise ValueError("INVALID_INVARIANT_STATUS")
    if candidate.review_class not in _ALLOWED_REVIEW_CLASSES:
        raise ValueError("INVALID_REVIEW_CLASS")
    if candidate.required_next_receipt not in _ALLOWED_REQUIRED_RECEIPTS:
        raise ValueError("INVALID_REQUIRED_NEXT_RECEIPT")
    if not isinstance(candidate.reason, str) or len(candidate.reason) > _MAX_REASON_LENGTH:
        raise ValueError("INVALID_INPUT")
    for h in candidate.evidence_hashes:
        _validate_hash_format(h)
    if tuple(sorted(candidate.source_paths)) != tuple(candidate.source_paths):
        raise ValueError("INVALID_INPUT")
    expected = _hash_payload(_base_candidate_payload(candidate))
    if candidate.candidate_hash == "" and allow_blank_hash:
        return True
    _validate_hash_format(candidate.candidate_hash)
    if candidate.candidate_hash != expected:
        raise ValueError("HASH_MISMATCH")
    return True


def build_backend_invariant_candidate_receipt(
    discovery_manifest: HeavyDependencyDiscoveryManifest,
    hotpath_receipt: DependencyImportAndHotPathReceipt,
    evidence: Iterable[BackendInvariantEvidence],
    candidates: Iterable[BackendInvariantCandidate],
) -> BackendInvariantCandidateReceipt:
    validate_heavy_dependency_discovery_manifest(discovery_manifest)
    validate_dependency_import_and_hotpath_receipt(hotpath_receipt)
    es = tuple(sorted(tuple(evidence), key=lambda x: x.evidence_index))
    cs = tuple(sorted(tuple(candidates), key=lambda x: x.candidate_index))
    if len(es) > _MAX_INVARIANT_EVIDENCE or len(cs) > _MAX_INVARIANT_CANDIDATES:
        raise ValueError("INVALID_INPUT")
    for e in es:
        validate_backend_invariant_evidence(e)
    for c in cs:
        validate_backend_invariant_candidate(c)
    if tuple(e.evidence_index for e in es) != tuple(range(len(es))):
        raise ValueError("EVIDENCE_ORDER_MISMATCH")
    if tuple(c.candidate_index for c in cs) != tuple(range(len(cs))):
        raise ValueError("CANDIDATE_ORDER_MISMATCH")
    receipt = BackendInvariantCandidateReceipt(
        schema_version=_SCHEMA_VERSION,
        candidate_mode=_CANDIDATE_MODE,
        discovery_manifest_hash=discovery_manifest.heavy_dependency_discovery_manifest_hash,
        dependency_hotpath_receipt_hash=hotpath_receipt.dependency_hotpath_receipt_hash,
        evidence_count=len(es),
        candidate_count=len(cs),
        blocked_candidate_count=sum(1 for c in cs if c.invariant_status == "CANDIDATE_BLOCKED"),
        equivalence_required_count=sum(1 for c in cs if c.required_next_receipt == "CrossBackendEquivalenceReceipt"),
        benchmark_required_count=sum(1 for c in cs if c.required_next_receipt == "OptimizedQECBenchmarkReceipt"),
        policy_normalization_required_count=sum(1 for c in cs if c.required_next_receipt == "UpstreamSourceNormalizationReceipt"),
        evidence=es,
        candidates=cs,
        first_evidence_hash=es[0].evidence_hash if es else "",
        final_evidence_hash=es[-1].evidence_hash if es else "",
        first_candidate_hash=cs[0].candidate_hash if cs else "",
        final_candidate_hash=cs[-1].candidate_hash if cs else "",
        backend_invariant_candidate_receipt_hash="",
    )
    payload = receipt.to_dict()
    payload.pop("backend_invariant_candidate_receipt_hash")
    return BackendInvariantCandidateReceipt(
        **{
            **receipt.to_dict(),
            "evidence": es,
            "candidates": cs,
            "backend_invariant_candidate_receipt_hash": _hash_payload(payload),
        }
    )


def validate_backend_invariant_candidate_receipt(receipt: BackendInvariantCandidateReceipt) -> bool:
    if not isinstance(receipt, BackendInvariantCandidateReceipt):
        raise ValueError("INVALID_INPUT")
    if receipt.schema_version != _SCHEMA_VERSION:
        raise ValueError("INVALID_SCHEMA_VERSION")
    if receipt.candidate_mode != _CANDIDATE_MODE:
        raise ValueError("INVALID_CANDIDATE_MODE")
    _validate_hash_format(receipt.discovery_manifest_hash)
    _validate_hash_format(receipt.dependency_hotpath_receipt_hash)
    if receipt.evidence_count != len(receipt.evidence) or receipt.candidate_count != len(receipt.candidates):
        raise ValueError("INVARIANT_COUNT_MISMATCH")
    for e in receipt.evidence:
        validate_backend_invariant_evidence(e)
    for c in receipt.candidates:
        validate_backend_invariant_candidate(c)
    if tuple(e.evidence_index for e in receipt.evidence) != tuple(range(len(receipt.evidence))):
        raise ValueError("EVIDENCE_ORDER_MISMATCH")
    if tuple(c.candidate_index for c in receipt.candidates) != tuple(range(len(receipt.candidates))):
        raise ValueError("CANDIDATE_ORDER_MISMATCH")
    if receipt.first_evidence_hash != (receipt.evidence[0].evidence_hash if receipt.evidence else ""):
        raise ValueError("HASH_MISMATCH")
    if receipt.final_evidence_hash != (receipt.evidence[-1].evidence_hash if receipt.evidence else ""):
        raise ValueError("HASH_MISMATCH")
    if receipt.first_candidate_hash != (receipt.candidates[0].candidate_hash if receipt.candidates else ""):
        raise ValueError("HASH_MISMATCH")
    if receipt.final_candidate_hash != (receipt.candidates[-1].candidate_hash if receipt.candidates else ""):
        raise ValueError("HASH_MISMATCH")
    if receipt.blocked_candidate_count != sum(1 for c in receipt.candidates if c.invariant_status == "CANDIDATE_BLOCKED"):
        raise ValueError("INVARIANT_COUNT_MISMATCH")
    if receipt.equivalence_required_count != sum(1 for c in receipt.candidates if c.required_next_receipt == "CrossBackendEquivalenceReceipt"):
        raise ValueError("INVARIANT_COUNT_MISMATCH")
    if receipt.benchmark_required_count != sum(1 for c in receipt.candidates if c.required_next_receipt == "OptimizedQECBenchmarkReceipt"):
        raise ValueError("INVARIANT_COUNT_MISMATCH")
    if receipt.policy_normalization_required_count != sum(1 for c in receipt.candidates if c.required_next_receipt == "UpstreamSourceNormalizationReceipt"):
        raise ValueError("INVARIANT_COUNT_MISMATCH")
    _validate_hash_format(receipt.backend_invariant_candidate_receipt_hash)
    payload = receipt.to_dict()
    payload.pop("backend_invariant_candidate_receipt_hash")
    if _hash_payload(payload) != receipt.backend_invariant_candidate_receipt_hash:
        raise ValueError("HASH_MISMATCH")
    return True


def derive_backend_invariant_candidates(
    discovery_manifest: HeavyDependencyDiscoveryManifest,
    hotpath_receipt: DependencyImportAndHotPathReceipt,
) -> BackendInvariantCandidateReceipt:
    validate_heavy_dependency_discovery_manifest(discovery_manifest)
    validate_dependency_import_and_hotpath_receipt(hotpath_receipt)
    ev: list[BackendInvariantEvidence] = []
    # P1 fix: Skip non-registry imports when deriving evidence
    for s in hotpath_receipt.import_sites:
        if s.dependency_name not in _REGISTRY:
            continue
        ev.append(build_backend_invariant_evidence(
            evidence_index=len(ev),
            dependency_name=s.dependency_name,
            evidence_kind="IMPORT_SITE_EVIDENCE",
            source_path=s.source_path,
            line_number=s.line_number,
            import_site_hash=s.import_site_hash,
            hotpath_candidate_hash=None,
            probe_hash=None,
            reason=f"import:{s.import_kind}:{s.import_placement}",
        ))
    for h in hotpath_receipt.hotpath_candidates:
        ev.append(build_backend_invariant_evidence(
            evidence_index=len(ev),
            dependency_name=h.dependency_name,
            evidence_kind="HOTPATH_CANDIDATE_EVIDENCE",
            source_path=h.source_path,
            line_number=h.line_number,
            import_site_hash=None,
            hotpath_candidate_hash=h.candidate_hash,
            probe_hash=None,
            reason=h.reason,
        ))
    # Iterate discovery_manifest.probe_results directly for determinism
    for p in discovery_manifest.probe_results:
        ev.append(build_backend_invariant_evidence(
            evidence_index=len(ev),
            dependency_name=p.dependency_name,
            evidence_kind="DISCOVERY_STATUS_EVIDENCE",
            source_path=None,
            line_number=None,
            import_site_hash=None,
            hotpath_candidate_hash=None,
            probe_hash=p.probe_hash,
            reason=p.availability_status,
        ))
        ev.append(build_backend_invariant_evidence(
            evidence_index=len(ev),
            dependency_name=p.dependency_name,
            evidence_kind="POLICY_STATUS_EVIDENCE",
            source_path=None,
            line_number=None,
            import_site_hash=None,
            hotpath_candidate_hash=None,
            probe_hash=p.probe_hash,
            reason=p.policy_status,
        ))
    cands: list[BackendInvariantCandidate] = []

    def add(dep: str, name: str, kind: str, status: str, review: str, nxt: str, reason: str, srcs: list[str], ehs: list[str]) -> None:
        cands.append(build_backend_invariant_candidate(
            candidate_index=0,
            dependency_name=dep,
            invariant_name=name,
            invariant_kind=kind,
            invariant_status=status,
            review_class=review,
            evidence_hashes=tuple(sorted(set(ehs))),
            source_paths=tuple(sorted(set(srcs))),
            reason=reason,
            required_next_receipt=nxt,
        ))

    ev_by_dep: dict[str, list[BackendInvariantEvidence]] = {}
    for e in ev:
        ev_by_dep.setdefault(e.dependency_name, []).append(e)
    # Iterate discovery_manifest.probe_results directly for determinism
    for p in discovery_manifest.probe_results:
        dep = p.dependency_name
        ehs = [x.evidence_hash for x in ev_by_dep.get(dep, [])]
        src = [x.source_path for x in ev_by_dep.get(dep, []) if x.source_path]
        if p.availability_status in {"AVAILABLE", "INTERNAL_AVAILABLE"}:
            add(
                dep,
                f"{dep}_available_surface",
                "AVAILABLE_BACKEND_SURFACE_INVARIANT",
                "CANDIDATE_IDENTIFIED",
                "DISCOVERY_ONLY",
                "CrossBackendEquivalenceReceipt",
                "available backend surface",
                src,
                ehs,
            )
        elif p.availability_status in {"UNAVAILABLE", "NOT_PROBED"}:
            add(
                dep,
                f"{dep}_unavailable_surface",
                "UNAVAILABLE_BACKEND_INVARIANT",
                "CANDIDATE_REQUIRES_REVIEW",
                "DISCOVERY_ONLY",
                "NONE",
                "unavailable or not probed",
                src,
                ehs,
            )
        elif p.availability_status == "BLOCKED_BY_POLICY":
            add(
                dep,
                f"{dep}_policy_blocked",
                "POLICY_BLOCKED_EXTERNAL_INVARIANT",
                "CANDIDATE_BLOCKED",
                "BLOCKED_BY_POLICY",
                "UpstreamSourceNormalizationReceipt",
                "blocked by policy",
                src,
                ehs,
            )
    for h in hotpath_receipt.hotpath_candidates:
        ehs = [x.evidence_hash for x in ev_by_dep.get(h.dependency_name, [])]
        src = [x.source_path for x in ev_by_dep.get(h.dependency_name, []) if x.source_path]
        if h.candidate_kind == "MODULE_TOP_LEVEL_HEAVY_IMPORT":
            add(
                h.dependency_name,
                f"{h.dependency_name}_top_level_boundary",
                "TOP_LEVEL_IMPORT_BOUNDARY_INVARIANT",
                "CANDIDATE_REQUIRES_REVIEW",
                "NEEDS_BENCHMARK_RECEIPT",
                "OptimizedQECBenchmarkReceipt",
                h.reason,
                src,
                ehs,
            )
        if h.candidate_kind == "REPEATED_IMPORT_REFERENCE":
            add(
                h.dependency_name,
                f"{h.dependency_name}_repeated_import_surface",
                "REPEATED_IMPORT_SURFACE_INVARIANT",
                "CANDIDATE_IDENTIFIED",
                "SAFE_FOR_OPTIMIZATION_CONTRACT_REVIEW",
                "OptimizationContract",
                h.reason,
                src,
                ehs,
            )
        # P2 fix: Restrict boundary invariants to matching hotpath kinds
        if h.dependency_name in {"qutip", "qiskit", "qiskit_aer", "stim", "pymatching"} and h.candidate_kind == "QUANTUM_BACKEND_BOUNDARY":
            add(
                h.dependency_name,
                f"{h.dependency_name}_quantum_boundary",
                "QUANTUM_BACKEND_BOUNDARY_INVARIANT",
                "CANDIDATE_REQUIRES_REVIEW",
                "NEEDS_EQUIVALENCE_RECEIPT",
                "CrossBackendEquivalenceReceipt",
                h.reason,
                src,
                ehs,
            )
        if h.dependency_name == "scipy" and h.candidate_kind == "DENSE_SPARSE_BOUNDARY":
            add(
                h.dependency_name,
                f"{h.dependency_name}_sparse_dense_boundary",
                "SPARSE_DENSE_BOUNDARY_INVARIANT",
                "CANDIDATE_REQUIRES_REVIEW",
                "NEEDS_EQUIVALENCE_RECEIPT",
                "CrossBackendEquivalenceReceipt",
                h.reason,
                src,
                ehs,
            )
        if h.dependency_name == "matplotlib" and h.candidate_kind == "PLOTTING_RENDER_BOUNDARY":
            add(
                h.dependency_name,
                "matplotlib_render_boundary",
                "PLOTTING_RENDER_BOUNDARY_INVARIANT",
                "CANDIDATE_IDENTIFIED",
                "SAFE_FOR_OPTIMIZATION_CONTRACT_REVIEW",
                "OptimizationContract",
                h.reason,
                src,
                ehs,
            )
        if h.dependency_name == "pandas" and h.candidate_kind == "DATAFRAME_BOUNDARY":
            add(
                h.dependency_name,
                "pandas_dataframe_boundary",
                "DATAFRAME_SCHEMA_BOUNDARY_INVARIANT",
                "CANDIDATE_REQUIRES_REVIEW",
                "NEEDS_EQUIVALENCE_RECEIPT",
                "CrossBackendEquivalenceReceipt",
                h.reason,
                src,
                ehs,
            )
        if h.dependency_name == "mido" and h.candidate_kind == "AUDIO_MIDI_BOUNDARY":
            add(
                h.dependency_name,
                "mido_audio_midi_boundary",
                "AUDIO_MIDI_BOUNDARY_INVARIANT",
                "CANDIDATE_IDENTIFIED",
                "SAFE_FOR_OPTIMIZATION_CONTRACT_REVIEW",
                "OptimizationContract",
                h.reason,
                src,
                ehs,
            )
        if h.dependency_name == "qldpc_internal" and h.candidate_kind == "INTERNAL_QEC_BOUNDARY":
            add(
                h.dependency_name,
                "qldpc_internal_surface",
                "INTERNAL_QEC_SURFACE_INVARIANT",
                "CANDIDATE_IDENTIFIED",
                "SAFE_FOR_OPTIMIZATION_CONTRACT_REVIEW",
                "OptimizationContract",
                h.reason,
                src,
                ehs,
            )
    cands_sorted = sorted(
        cands,
        key=lambda c: (c.dependency_name, c.invariant_kind, c.review_class, c.required_next_receipt, c.reason, "|".join(c.source_paths)),
    )
    cands_final = [
        build_backend_invariant_candidate(**{**c.to_dict(), "candidate_index": i, "candidate_hash": ""})
        for i, c in enumerate(cands_sorted)
    ]
    ev_final = [
        build_backend_invariant_evidence(**{**e.to_dict(), "evidence_index": i, "evidence_hash": ""})
        for i, e in enumerate(ev)
    ]
    return build_backend_invariant_candidate_receipt(discovery_manifest, hotpath_receipt, tuple(ev_final), tuple(cands_final))


def validate_receipt_matches_inputs(
    receipt: BackendInvariantCandidateReceipt,
    discovery_manifest: HeavyDependencyDiscoveryManifest,
    hotpath_receipt: DependencyImportAndHotPathReceipt,
) -> bool:
    expected = derive_backend_invariant_candidates(discovery_manifest, hotpath_receipt)
    if receipt.to_dict() != expected.to_dict():
        if receipt.discovery_manifest_hash != discovery_manifest.heavy_dependency_discovery_manifest_hash:
            raise ValueError("DISCOVERY_MANIFEST_MISMATCH")
        if receipt.dependency_hotpath_receipt_hash != hotpath_receipt.dependency_hotpath_receipt_hash:
            raise ValueError("HOTPATH_RECEIPT_MISMATCH")
        raise ValueError("BACKEND_INVARIANT_RECEIPT_MISMATCH")
    return True
