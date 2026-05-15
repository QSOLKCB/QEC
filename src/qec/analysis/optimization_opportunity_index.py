"""Optimization Opportunity Index receipt for deterministic discovery-based ranking.

This module provides a deterministic analysis-layer receipt that indexes and ranks
static optimization opportunities derived from v163 discovery artifacts:

- HeavyDependencyDiscoveryManifest: registry of heavy dependencies
- DependencyImportAndHotPathReceipt: hotpath candidates from import analysis
- BackendInvariantCandidateReceipt: invariant candidates for optimization
- CrossBackendEquivalenceReceipt (optional): equivalence proof results

The index produces a canonical, hash-verifiable ranking of optimization opportunities
without performing any backend execution, heavy imports, or runtime benchmarking.

Scoring Model:
    total_priority_score = static_determinism_score + static_value_score
                         + dependency_reduction_score + (5 - implementation_risk_score)

    Each component score ranges from 0-5, yielding a total range of 0-20.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from typing import Any

from .heavy_dependency_discovery import (
    HeavyDependencyDiscoveryManifest,
    validate_heavy_dependency_discovery_manifest,
    get_heavy_dependency_targets,
)
from .dependency_hotpath_receipts import (
    DependencyImportAndHotPathReceipt,
    validate_dependency_import_and_hotpath_receipt,
)
from .backend_invariant_candidate_receipts import (
    BackendInvariantCandidateReceipt,
    BackendInvariantCandidate,
    validate_backend_invariant_candidate_receipt,
)
from .cross_backend_equivalence_receipts import (
    CrossBackendEquivalenceReceipt,
    validate_cross_backend_equivalence_receipt,
)

_SCHEMA_VERSION = "OPTIMIZATION_OPPORTUNITY_INDEX_V1"
_INDEX_MODE = "DETERMINISTIC_DISCOVERY_OPPORTUNITY_RANKING"
_MAX_OPPORTUNITY_EVIDENCE = 4096
_MAX_OPPORTUNITIES = 2048
_MAX_REASON_LENGTH = 256
_MAX_OPPORTUNITY_NAME_LENGTH = 128
_HASH_RE = re.compile(r"^[0-9a-f]{64}$")

_ALLOWED_EVIDENCE_KINDS = frozenset({
    "INVARIANT_CANDIDATE_EVIDENCE",
    "HOTPATH_CANDIDATE_EVIDENCE",
    "EQUIVALENCE_RESULT_EVIDENCE",
    "POLICY_STATUS_EVIDENCE",
    "DEPENDENCY_DISCOVERY_EVIDENCE",
})

_ALLOWED_OPPORTUNITY_KINDS = frozenset({
    "IMPORT_SURFACE_REDUCTION",
    "TOP_LEVEL_IMPORT_DEFERRAL",
    "REPEATED_IMPORT_COLLAPSE",
    "PLOTTING_RENDER_BYPASS",
    "DATAFRAME_SCHEMA_CACHE_REVIEW",
    "SPARSE_DENSE_BOUNDARY_REVIEW",
    "QUANTUM_BACKEND_ADAPTER_REVIEW",
    "AUDIO_MIDI_ADAPTER_REVIEW",
    "INTERNAL_QEC_FASTPATH_REVIEW",
    "POLICY_BLOCKED_DEPENDENCY_REVIEW",
    "UNAVAILABLE_BACKEND_REVIEW",
    "HASH_ONLY_EQUIVALENCE_REVIEW",
    "EXACT_JSON_EQUIVALENCE_REVIEW",
})

_ALLOWED_READINESS = frozenset({
    "READY_FOR_OPTIMIZATION_CONTRACT",
    "NEEDS_EQUIVALENCE_RECEIPT",
    "NEEDS_BENCHMARK_RECEIPT",
    "NEEDS_POLICY_NORMALIZATION",
    "BLOCKED",
    "DISCOVERY_ONLY",
    "NOT_READY",
})

_ALLOWED_NEXT = frozenset({
    "OptimizationContract",
    "FastPathEquivalenceReceipt",
    "OptimizedQECBenchmarkReceipt",
    "UpstreamSourceNormalizationReceipt",
    "LightweightAdapterSpec",
    "CrossBackendEquivalenceReceipt",
    "NONE",
})

_REGISTRY = frozenset(x.dependency_name for x in get_heavy_dependency_targets())

# Readiness priority for sorting (lower = higher priority in ranking)
_READINESS_PRIORITY = {
    "READY_FOR_OPTIMIZATION_CONTRACT": 0,
    "NEEDS_BENCHMARK_RECEIPT": 1,
    "NEEDS_EQUIVALENCE_RECEIPT": 2,
    "DISCOVERY_ONLY": 3,
    "NOT_READY": 4,
    "NEEDS_POLICY_NORMALIZATION": 5,
    "BLOCKED": 6,
}

# Static determinism score by readiness status (0-5)
_STATIC_DETERMINISM_SCORE_BY_READINESS = {
    "READY_FOR_OPTIMIZATION_CONTRACT": 5,
    "NEEDS_EQUIVALENCE_RECEIPT": 2,
    "NEEDS_BENCHMARK_RECEIPT": 3,
    "BLOCKED": 0,
    "DISCOVERY_ONLY": 1,
    "NOT_READY": 1,
    "NEEDS_POLICY_NORMALIZATION": 1,
}

# Static value score by opportunity kind (0-5)
_STATIC_VALUE_SCORE_BY_KIND = {
    "INTERNAL_QEC_FASTPATH_REVIEW": 5,
    "SPARSE_DENSE_BOUNDARY_REVIEW": 5,
    "QUANTUM_BACKEND_ADAPTER_REVIEW": 5,
    "TOP_LEVEL_IMPORT_DEFERRAL": 4,
    "REPEATED_IMPORT_COLLAPSE": 4,
    "PLOTTING_RENDER_BYPASS": 4,
    "DATAFRAME_SCHEMA_CACHE_REVIEW": 4,
    "AUDIO_MIDI_ADAPTER_REVIEW": 3,
    "UNAVAILABLE_BACKEND_REVIEW": 1,
    "POLICY_BLOCKED_DEPENDENCY_REVIEW": 0,
    "HASH_ONLY_EQUIVALENCE_REVIEW": 3,
    "EXACT_JSON_EQUIVALENCE_REVIEW": 3,
    "IMPORT_SURFACE_REDUCTION": 3,
}

# Implementation risk score by opportunity kind (0-5, higher = riskier)
_IMPLEMENTATION_RISK_SCORE_BY_KIND = {
    "PLOTTING_RENDER_BYPASS": 1,
    "AUDIO_MIDI_ADAPTER_REVIEW": 2,
    "REPEATED_IMPORT_COLLAPSE": 2,
    "TOP_LEVEL_IMPORT_DEFERRAL": 3,
    "DATAFRAME_SCHEMA_CACHE_REVIEW": 3,
    "SPARSE_DENSE_BOUNDARY_REVIEW": 4,
    "QUANTUM_BACKEND_ADAPTER_REVIEW": 5,
    "INTERNAL_QEC_FASTPATH_REVIEW": 3,
    "POLICY_BLOCKED_DEPENDENCY_REVIEW": 5,
    "UNAVAILABLE_BACKEND_REVIEW": 5,
    "HASH_ONLY_EQUIVALENCE_REVIEW": 2,
    "EXACT_JSON_EQUIVALENCE_REVIEW": 2,
    "IMPORT_SURFACE_REDUCTION": 2,
}

# Dependency reduction score by opportunity kind (0-5)
_DEPENDENCY_REDUCTION_SCORE_BY_KIND = {
    "TOP_LEVEL_IMPORT_DEFERRAL": 5,
    "REPEATED_IMPORT_COLLAPSE": 5,
    "PLOTTING_RENDER_BYPASS": 4,
    "DATAFRAME_SCHEMA_CACHE_REVIEW": 4,
    "INTERNAL_QEC_FASTPATH_REVIEW": 4,
    "SPARSE_DENSE_BOUNDARY_REVIEW": 3,
    "QUANTUM_BACKEND_ADAPTER_REVIEW": 3,
    "AUDIO_MIDI_ADAPTER_REVIEW": 2,
    "POLICY_BLOCKED_DEPENDENCY_REVIEW": 0,
    "UNAVAILABLE_BACKEND_REVIEW": 0,
    "HASH_ONLY_EQUIVALENCE_REVIEW": 2,
    "EXACT_JSON_EQUIVALENCE_REVIEW": 2,
    "IMPORT_SURFACE_REDUCTION": 3,
}


def _canonical_json(obj: Any) -> str:
    """Convert object to canonical JSON string with sorted keys and compact separators."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _compute_hash(obj: Any) -> str:
    """Compute SHA-256 hash of canonical JSON representation."""
    return hashlib.sha256(_canonical_json(obj).encode()).hexdigest()


def _validate_hash_format(hash_str: str) -> None:
    """Validate that a string is a valid 64-character lowercase hex SHA-256 hash."""
    if not isinstance(hash_str, str) or _HASH_RE.fullmatch(hash_str) is None:
        raise ValueError("INVALID_HASH_FORMAT")


@dataclass(frozen=True)
class OptimizationOpportunityEvidence:
    """Evidence record linking an optimization opportunity to its source artifacts.

    Each evidence record traces back to the discovery artifacts that identified
    the optimization opportunity, enabling full provenance tracking.
    """

    evidence_index: int
    dependency_name: str
    evidence_kind: str
    source_candidate_hash: str | None
    source_hotpath_candidate_hash: str | None
    source_equivalence_result_hash: str | None
    source_probe_hash: str | None
    reason: str
    evidence_hash: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return self.__dict__.copy()

    def to_canonical_json(self) -> str:
        """Convert to canonical JSON string."""
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        """Convert to canonical JSON bytes."""
        return self.to_canonical_json().encode()


@dataclass(frozen=True)
class OptimizationOpportunityEntry:
    """A ranked optimization opportunity with scoring and readiness metadata.

    Each entry represents a single optimization opportunity derived from
    invariant candidates, with deterministic scoring based on readiness,
    value, risk, and dependency reduction potential.
    """

    opportunity_index: int
    dependency_name: str
    opportunity_name: str
    opportunity_kind: str
    readiness_status: str
    required_next_receipt: str
    evidence_hashes: tuple[str, ...]
    source_candidate_hash: str | None
    static_determinism_score: int
    static_value_score: int
    implementation_risk_score: int
    dependency_reduction_score: int
    total_priority_score: int
    rank_reason: str
    opportunity_hash: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        d = self.__dict__.copy()
        d["evidence_hashes"] = list(self.evidence_hashes)
        return d

    def to_canonical_json(self) -> str:
        """Convert to canonical JSON string."""
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        """Convert to canonical JSON bytes."""
        return self.to_canonical_json().encode()


@dataclass(frozen=True)
class OptimizationOpportunityIndex:
    """Deterministic index of ranked optimization opportunities.

    This receipt aggregates evidence and opportunities from discovery artifacts
    into a canonical, hash-verifiable index sorted by priority score and
    readiness status.
    """

    schema_version: str
    index_mode: str
    discovery_manifest_hash: str
    dependency_hotpath_receipt_hash: str
    backend_invariant_candidate_receipt_hash: str
    cross_backend_equivalence_receipt_hash: str | None
    evidence_count: int
    opportunity_count: int
    ready_for_contract_count: int
    needs_equivalence_count: int
    needs_benchmark_count: int
    needs_policy_normalization_count: int
    blocked_count: int
    discovery_only_count: int
    not_ready_count: int
    evidence: tuple[OptimizationOpportunityEvidence, ...]
    opportunities: tuple[OptimizationOpportunityEntry, ...]
    first_evidence_hash: str
    final_evidence_hash: str
    first_opportunity_hash: str
    final_opportunity_hash: str
    optimization_opportunity_index_hash: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            **self.__dict__,
            "evidence": [e.to_dict() for e in self.evidence],
            "opportunities": [o.to_dict() for o in self.opportunities],
        }

    def to_canonical_json(self) -> str:
        """Convert to canonical JSON string."""
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        """Convert to canonical JSON bytes."""
        return self.to_canonical_json().encode()


def build_optimization_opportunity_evidence(**kwargs: Any) -> OptimizationOpportunityEvidence:
    """Build an OptimizationOpportunityEvidence with computed hash.

    Validates all fields and computes the evidence_hash from canonical JSON.
    """
    e = OptimizationOpportunityEvidence(evidence_hash="", **kwargs)
    validate_optimization_opportunity_evidence(e, allow_blank_hash=True)
    d = e.to_dict()
    d.pop("evidence_hash")
    return OptimizationOpportunityEvidence(**{**e.to_dict(), "evidence_hash": _compute_hash(d)})


def build_optimization_opportunity_entry(**kwargs: Any) -> OptimizationOpportunityEntry:
    """Build an OptimizationOpportunityEntry with computed hash.

    Validates all fields and computes the opportunity_hash from canonical JSON.
    """
    k = dict(kwargs)
    k.pop("opportunity_hash", None)
    e = OptimizationOpportunityEntry(opportunity_hash="", **k)
    validate_optimization_opportunity_entry(e, allow_blank_hash=True)
    d = e.to_dict()
    d.pop("opportunity_hash")
    return OptimizationOpportunityEntry(
        **{**e.to_dict(), "opportunity_hash": _compute_hash(d), "evidence_hashes": tuple(e.evidence_hashes)}
    )


def validate_optimization_opportunity_evidence(
    evidence: OptimizationOpportunityEvidence, allow_blank_hash: bool = False
) -> bool:
    """Validate an OptimizationOpportunityEvidence record.

    Checks field types, allowed values, hash formats, and hash integrity.
    """
    if not isinstance(evidence, OptimizationOpportunityEvidence):
        raise ValueError("INVALID_INPUT")
    if (
        not isinstance(evidence.evidence_index, int)
        or isinstance(evidence.evidence_index, bool)
        or evidence.evidence_index < 0
    ):
        raise ValueError("INVALID_INPUT")
    if evidence.dependency_name not in _REGISTRY:
        raise ValueError("INVALID_DEPENDENCY_NAME")
    if evidence.evidence_kind not in _ALLOWED_EVIDENCE_KINDS:
        raise ValueError("INVALID_EVIDENCE_KIND")
    for k in (
        evidence.source_candidate_hash,
        evidence.source_hotpath_candidate_hash,
        evidence.source_equivalence_result_hash,
        evidence.source_probe_hash,
    ):
        if k is not None:
            _validate_hash_format(k)
    if not isinstance(evidence.reason, str) or len(evidence.reason) > _MAX_REASON_LENGTH:
        raise ValueError("INVALID_INPUT")
    d = evidence.to_dict()
    d.pop("evidence_hash")
    exp = _compute_hash(d)
    if evidence.evidence_hash == "" and allow_blank_hash:
        return True
    _validate_hash_format(evidence.evidence_hash)
    if evidence.evidence_hash != exp:
        raise ValueError("HASH_MISMATCH")
    return True

def validate_optimization_opportunity_entry(
    entry: OptimizationOpportunityEntry, allow_blank_hash: bool = False
) -> bool:
    """Validate an OptimizationOpportunityEntry record.

    Checks field types, allowed values, score bounds, hash formats, and hash integrity.
    """
    if not isinstance(entry, OptimizationOpportunityEntry):
        raise ValueError("INVALID_INPUT")
    if entry.dependency_name not in _REGISTRY:
        raise ValueError("INVALID_DEPENDENCY_NAME")
    if entry.opportunity_kind not in _ALLOWED_OPPORTUNITY_KINDS:
        raise ValueError("INVALID_OPPORTUNITY_KIND")
    if entry.readiness_status not in _ALLOWED_READINESS:
        raise ValueError("INVALID_READINESS_STATUS")
    if entry.required_next_receipt not in _ALLOWED_NEXT:
        raise ValueError("INVALID_REQUIRED_NEXT_RECEIPT")
    if (
        not isinstance(entry.opportunity_name, str)
        or not entry.opportunity_name
        or len(entry.opportunity_name) > _MAX_OPPORTUNITY_NAME_LENGTH
    ):
        raise ValueError("INVALID_INPUT")
    if not isinstance(entry.rank_reason, str) or len(entry.rank_reason) > _MAX_REASON_LENGTH:
        raise ValueError("INVALID_INPUT")
    for h in entry.evidence_hashes:
        _validate_hash_format(h)
    if entry.source_candidate_hash is not None:
        _validate_hash_format(entry.source_candidate_hash)
    for s in (
        entry.static_determinism_score,
        entry.static_value_score,
        entry.implementation_risk_score,
        entry.dependency_reduction_score,
    ):
        if not isinstance(s, int) or isinstance(s, bool) or s < 0 or s > 5:
            raise ValueError("INVALID_SCORE")
    exp_total = (
        entry.static_determinism_score
        + entry.static_value_score
        + entry.dependency_reduction_score
        + (5 - entry.implementation_risk_score)
    )
    if entry.total_priority_score != exp_total or entry.total_priority_score < 0 or entry.total_priority_score > 20:
        raise ValueError("INVALID_SCORE")
    d = entry.to_dict()
    d.pop("opportunity_hash")
    exp = _compute_hash(d)
    if entry.opportunity_hash == "" and allow_blank_hash:
        return True
    _validate_hash_format(entry.opportunity_hash)
    if entry.opportunity_hash != exp:
        raise ValueError("HASH_MISMATCH")
    return True


def build_optimization_opportunity_index(
    discovery_manifest: HeavyDependencyDiscoveryManifest,
    hotpath_receipt: DependencyImportAndHotPathReceipt,
    invariant_candidate_receipt: BackendInvariantCandidateReceipt,
    evidence,
    opportunities,
    *,
    equivalence_receipt: CrossBackendEquivalenceReceipt | None = None,
) -> OptimizationOpportunityIndex:
    """Build an OptimizationOpportunityIndex from validated inputs.

    Sorts opportunities by priority, reassigns contiguous indexes, and computes
    the final index hash.
    """
    validate_heavy_dependency_discovery_manifest(discovery_manifest)
    validate_dependency_import_and_hotpath_receipt(hotpath_receipt)
    validate_backend_invariant_candidate_receipt(invariant_candidate_receipt)
    if equivalence_receipt is not None:
        validate_cross_backend_equivalence_receipt(equivalence_receipt)
        if (
            equivalence_receipt.invariant_candidate_receipt_hash
            != invariant_candidate_receipt.backend_invariant_candidate_receipt_hash
        ):
            raise ValueError("EQUIVALENCE_RECEIPT_MISMATCH")
    es = tuple(sorted(tuple(evidence), key=lambda x: x.evidence_index))
    os = tuple(opportunities)
    for e in es:
        validate_optimization_opportunity_evidence(e)
    if tuple(x.evidence_index for x in es) != tuple(range(len(es))):
        raise ValueError("EVIDENCE_ORDER_MISMATCH")
    if len(es) > _MAX_OPPORTUNITY_EVIDENCE or len(os) > _MAX_OPPORTUNITIES:
        raise ValueError("INVALID_INPUT")
    for o in os:
        validate_optimization_opportunity_entry(o)
    os = tuple(
        sorted(
            os,
            key=lambda x: (
                1 if x.readiness_status == "BLOCKED" else 0,
                _READINESS_PRIORITY[x.readiness_status],
                -x.total_priority_score,
                x.implementation_risk_score,
                x.dependency_name,
                x.opportunity_kind,
                x.opportunity_name,
                x.source_candidate_hash or "",
            ),
        )
    )
    os = tuple(
        build_optimization_opportunity_entry(
            **{**o.to_dict(), "opportunity_index": i, "opportunity_hash": o.opportunity_hash}
        )
        for i, o in enumerate(os)
    )
    if tuple(x.opportunity_index for x in os) != tuple(range(len(os))):
        raise ValueError("OPPORTUNITY_ORDER_MISMATCH")
    idx = OptimizationOpportunityIndex(
        _SCHEMA_VERSION,
        _INDEX_MODE,
        discovery_manifest.heavy_dependency_discovery_manifest_hash,
        hotpath_receipt.dependency_hotpath_receipt_hash,
        invariant_candidate_receipt.backend_invariant_candidate_receipt_hash,
        equivalence_receipt.cross_backend_equivalence_receipt_hash if equivalence_receipt else None,
        len(es),
        len(os),
        sum(1 for x in os if x.readiness_status == "READY_FOR_OPTIMIZATION_CONTRACT"),
        sum(1 for x in os if x.readiness_status == "NEEDS_EQUIVALENCE_RECEIPT"),
        sum(1 for x in os if x.readiness_status == "NEEDS_BENCHMARK_RECEIPT"),
        sum(1 for x in os if x.readiness_status == "NEEDS_POLICY_NORMALIZATION"),
        sum(1 for x in os if x.readiness_status == "BLOCKED"),
        sum(1 for x in os if x.readiness_status == "DISCOVERY_ONLY"),
        sum(1 for x in os if x.readiness_status == "NOT_READY"),
        es,
        os,
        es[0].evidence_hash if es else "",
        es[-1].evidence_hash if es else "",
        os[0].opportunity_hash if os else "",
        os[-1].opportunity_hash if os else "",
        "",
    )
    d = idx.to_dict()
    d.pop("optimization_opportunity_index_hash")
    return OptimizationOpportunityIndex(
        **{**idx.__dict__, "optimization_opportunity_index_hash": _compute_hash(d)}
    )


def validate_optimization_opportunity_index(index: OptimizationOpportunityIndex) -> bool:
    """Validate an OptimizationOpportunityIndex receipt.

    Checks schema version, index mode, hash formats, evidence/opportunity ordering,
    counts, and overall hash integrity.
    """
    if not isinstance(index, OptimizationOpportunityIndex):
        raise ValueError("INVALID_INPUT")
    if index.schema_version != _SCHEMA_VERSION:
        raise ValueError("INVALID_SCHEMA_VERSION")
    if index.index_mode != _INDEX_MODE:
        raise ValueError("INVALID_INDEX_MODE")
    for h in (
        index.discovery_manifest_hash,
        index.dependency_hotpath_receipt_hash,
        index.backend_invariant_candidate_receipt_hash,
    ):
        _validate_hash_format(h)
    if index.cross_backend_equivalence_receipt_hash is not None:
        _validate_hash_format(index.cross_backend_equivalence_receipt_hash)
    es = tuple(index.evidence)
    os = tuple(index.opportunities)
    for e in es:
        validate_optimization_opportunity_evidence(e)
    for o in os:
        validate_optimization_opportunity_entry(o)
    if tuple(x.evidence_index for x in es) != tuple(range(len(es))):
        raise ValueError("EVIDENCE_ORDER_MISMATCH")
    if tuple(x.opportunity_index for x in os) != tuple(range(len(os))):
        raise ValueError("OPPORTUNITY_ORDER_MISMATCH")
    if index.evidence_count != len(es) or index.opportunity_count != len(os):
        raise ValueError("OPPORTUNITY_COUNT_MISMATCH")
    if (
        index.first_evidence_hash != (es[0].evidence_hash if es else "")
        or index.final_evidence_hash != (es[-1].evidence_hash if es else "")
    ):
        raise ValueError("EVIDENCE_ORDER_MISMATCH")
    if (
        index.first_opportunity_hash != (os[0].opportunity_hash if os else "")
        or index.final_opportunity_hash != (os[-1].opportunity_hash if os else "")
    ):
        raise ValueError("OPPORTUNITY_ORDER_MISMATCH")
    d = index.to_dict()
    d.pop("optimization_opportunity_index_hash")
    exp = _compute_hash(d)
    _validate_hash_format(index.optimization_opportunity_index_hash)
    if exp != index.optimization_opportunity_index_hash:
        raise ValueError("HASH_MISMATCH")
    return True


def derive_optimization_opportunity_index(
    discovery_manifest: HeavyDependencyDiscoveryManifest,
    hotpath_receipt: DependencyImportAndHotPathReceipt,
    invariant_candidate_receipt: BackendInvariantCandidateReceipt,
    equivalence_receipt: CrossBackendEquivalenceReceipt | None = None,
) -> OptimizationOpportunityIndex:
    """Derive an OptimizationOpportunityIndex from discovery artifacts.

    Consumes the discovery manifest, hotpath receipt, invariant candidate receipt,
    and optional equivalence receipt to produce a deterministic, ranked index of
    optimization opportunities.
    """
    validate_heavy_dependency_discovery_manifest(discovery_manifest)
    validate_dependency_import_and_hotpath_receipt(hotpath_receipt)
    validate_backend_invariant_candidate_receipt(invariant_candidate_receipt)
    if equivalence_receipt is not None:
        validate_cross_backend_equivalence_receipt(equivalence_receipt)
        if (
            equivalence_receipt.invariant_candidate_receipt_hash
            != invariant_candidate_receipt.backend_invariant_candidate_receipt_hash
        ):
            raise ValueError("EQUIVALENCE_RECEIPT_MISMATCH")

    # Build equivalence status lookup by candidate hash
    eq_by_candidate: dict[str, list[str]] = {}
    if equivalence_receipt:
        case_by = {c.case_hash: c for c in equivalence_receipt.comparison_cases}
        for r in equivalence_receipt.comparison_results:
            c = case_by.get(r.case_hash)
            if c and c.source_candidate_hash:
                eq_by_candidate.setdefault(c.source_candidate_hash, []).append(r.result_status)

    ev: list[OptimizationOpportunityEvidence] = []
    op: list[OptimizationOpportunityEntry] = []

    # Build hotpath candidates lookup preserving all candidates per dependency
    hp_by_dep: dict[str, list] = {}
    for h in hotpath_receipt.hotpath_candidates:
        hp_by_dep.setdefault(h.dependency_name, []).append(h)

    # Mapping from invariant kind to opportunity kind
    kind_map = {
        "POLICY_BLOCKED_EXTERNAL_INVARIANT": "POLICY_BLOCKED_DEPENDENCY_REVIEW",
        "UNAVAILABLE_BACKEND_INVARIANT": "UNAVAILABLE_BACKEND_REVIEW",
        "TOP_LEVEL_IMPORT_BOUNDARY_INVARIANT": "TOP_LEVEL_IMPORT_DEFERRAL",
        "REPEATED_IMPORT_SURFACE_INVARIANT": "REPEATED_IMPORT_COLLAPSE",
        "QUANTUM_BACKEND_BOUNDARY_INVARIANT": "QUANTUM_BACKEND_ADAPTER_REVIEW",
        "SPARSE_DENSE_BOUNDARY_INVARIANT": "SPARSE_DENSE_BOUNDARY_REVIEW",
        "PLOTTING_RENDER_BOUNDARY_INVARIANT": "PLOTTING_RENDER_BYPASS",
        "DATAFRAME_SCHEMA_BOUNDARY_INVARIANT": "DATAFRAME_SCHEMA_CACHE_REVIEW",
        "AUDIO_MIDI_BOUNDARY_INVARIANT": "AUDIO_MIDI_ADAPTER_REVIEW",
        "INTERNAL_QEC_SURFACE_INVARIANT": "INTERNAL_QEC_FASTPATH_REVIEW",
        "AVAILABLE_BACKEND_SURFACE_INVARIANT": "EXACT_JSON_EQUIVALENCE_REVIEW",
    }

    for i, c in enumerate(invariant_candidate_receipt.candidates):
        # Add invariant candidate evidence
        ev.append(
            build_optimization_opportunity_evidence(
                evidence_index=len(ev),
                dependency_name=c.dependency_name,
                evidence_kind="INVARIANT_CANDIDATE_EVIDENCE",
                source_candidate_hash=c.candidate_hash,
                source_hotpath_candidate_hash=None,
                source_equivalence_result_hash=None,
                source_probe_hash=None,
                reason=c.invariant_kind,
            )
        )

        # Add hotpath candidate evidence for all matching hotpath candidates
        if c.dependency_name in hp_by_dep:
            for hpc in hp_by_dep[c.dependency_name]:
                ev.append(
                    build_optimization_opportunity_evidence(
                        evidence_index=len(ev),
                        dependency_name=c.dependency_name,
                        evidence_kind="HOTPATH_CANDIDATE_EVIDENCE",
                        source_candidate_hash=c.candidate_hash,
                        source_hotpath_candidate_hash=hpc.candidate_hash,
                        source_equivalence_result_hash=None,
                        source_probe_hash=None,
                        reason=hpc.candidate_kind,
                    )
                )

        if c.invariant_kind not in kind_map:
            continue

        # Determine readiness status and required next receipt
        ok = eq_by_candidate.get(c.candidate_hash, [])
        eq_ready = bool(ok) and all(x == "EQUIVALENT" for x in ok)

        if c.invariant_kind == "POLICY_BLOCKED_EXTERNAL_INVARIANT":
            rs, nx = "BLOCKED", "UpstreamSourceNormalizationReceipt"
        elif c.invariant_kind == "UNAVAILABLE_BACKEND_INVARIANT":
            rs, nx = "DISCOVERY_ONLY", "NONE"
        elif c.invariant_kind == "TOP_LEVEL_IMPORT_BOUNDARY_INVARIANT":
            rs, nx = "NEEDS_BENCHMARK_RECEIPT", "OptimizedQECBenchmarkReceipt"
        elif c.invariant_kind == "REPEATED_IMPORT_SURFACE_INVARIANT":
            rs, nx = "READY_FOR_OPTIMIZATION_CONTRACT", "OptimizationContract"
        elif c.invariant_kind in {
            "QUANTUM_BACKEND_BOUNDARY_INVARIANT",
            "SPARSE_DENSE_BOUNDARY_INVARIANT",
            "DATAFRAME_SCHEMA_BOUNDARY_INVARIANT",
            "AVAILABLE_BACKEND_SURFACE_INVARIANT",
        }:
            rs, nx = (
                ("READY_FOR_OPTIMIZATION_CONTRACT", "OptimizationContract")
                if eq_ready
                else ("NEEDS_EQUIVALENCE_RECEIPT", "CrossBackendEquivalenceReceipt")
            )
        else:
            rs, nx = "READY_FOR_OPTIMIZATION_CONTRACT", "OptimizationContract"

        # Compute scores
        k = kind_map[c.invariant_kind]
        sds = _STATIC_DETERMINISM_SCORE_BY_READINESS[rs]
        sv = _STATIC_VALUE_SCORE_BY_KIND[k]
        ir = _IMPLEMENTATION_RISK_SCORE_BY_KIND[k]
        dr = _DEPENDENCY_REDUCTION_SCORE_BY_KIND[k]
        total = sds + sv + dr + (5 - ir)

        op.append(
            build_optimization_opportunity_entry(
                opportunity_index=i,
                dependency_name=c.dependency_name,
                opportunity_name=c.invariant_name,
                opportunity_kind=k,
                readiness_status=rs,
                required_next_receipt=nx,
                evidence_hashes=tuple(
                    x.evidence_hash for x in ev if x.source_candidate_hash == c.candidate_hash
                ),
                source_candidate_hash=c.candidate_hash,
                static_determinism_score=sds,
                static_value_score=sv,
                implementation_risk_score=ir,
                dependency_reduction_score=dr,
                total_priority_score=total,
                rank_reason="STATIC_DISCOVERY_TRIAGE",
            )
        )

    return build_optimization_opportunity_index(
        discovery_manifest,
        hotpath_receipt,
        invariant_candidate_receipt,
        tuple(ev),
        tuple(op),
        equivalence_receipt=equivalence_receipt,
    )


def validate_index_matches_inputs(
    index: OptimizationOpportunityIndex,
    discovery_manifest: HeavyDependencyDiscoveryManifest,
    hotpath_receipt: DependencyImportAndHotPathReceipt,
    invariant_candidate_receipt: BackendInvariantCandidateReceipt,
    equivalence_receipt: CrossBackendEquivalenceReceipt | None = None,
) -> bool:
    """Validate that an index matches the expected derivation from inputs.

    Re-derives the index from the provided inputs and compares against the
    given index to ensure consistency.
    """
    exp = derive_optimization_opportunity_index(
        discovery_manifest, hotpath_receipt, invariant_candidate_receipt, equivalence_receipt
    )
    if exp.to_dict() != index.to_dict():
        raise ValueError("OPTIMIZATION_INDEX_MISMATCH")
    return True
