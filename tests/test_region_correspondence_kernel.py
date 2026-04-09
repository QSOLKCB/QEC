"""Tests for v137.13.3 — Region Correspondence Kernel."""

from __future__ import annotations

import hashlib
import json

import pytest

from qec.analysis.phase_boundary_topology_kernel import (
    PhaseBoundaryEdge,
    PhaseBoundaryTopologyConfig,
    PhaseRegion,
    PhaseTopologyPath,
    SCHEMA_VERSION as PHASE_SCHEMA_VERSION,
)
from qec.analysis.region_correspondence_kernel import (
    SCHEMA_VERSION,
    RegionCorrespondenceMap,
    RegionCorrespondencePair,
    _sha256_hex,
    build_region_correspondence_map,
    compute_region_correspondence_similarity,
    run_region_correspondence_kernel,
)


def _stable_hash_for(path: PhaseTopologyPath) -> str:
    return hashlib.sha256(
        json.dumps(path.to_hash_payload_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
            "utf-8"
        )
    ).hexdigest()


def _build_path(path_id: str, regions: tuple[tuple[str, float, int, int], ...]) -> PhaseTopologyPath:
    phase_config = PhaseBoundaryTopologyConfig()
    built_regions: list[PhaseRegion] = []
    for idx, (label, morphology_mean, left, right) in enumerate(regions):
        seed = {
            "path_id": path_id,
            "label": label,
            "idx": idx,
            "left": left,
            "right": right,
        }
        built_regions.append(
            PhaseRegion(
                region_id=hashlib.sha256(json.dumps(seed, sort_keys=True).encode("utf-8")).hexdigest(),
                source_start_index=left,
                source_end_index=right,
                region_label=label,
                region_score=0.700000000000,
                continuity_mean=0.600000000000,
                morphology_mean=morphology_mean,
            )
        )

    built_boundaries: list[PhaseBoundaryEdge] = []
    for idx in range(max(0, len(built_regions) - 1)):
        built_boundaries.append(
            PhaseBoundaryEdge(
                source_region_index=idx,
                target_region_index=idx + 1,
                boundary_type="region_transition",
                boundary_magnitude=0.100000000000,
                continuity_delta=0.050000000000,
                morphology_delta=0.050000000000,
            )
        )

    proto = PhaseTopologyPath(
        config=phase_config,
        input_transition_hash=hashlib.sha256(path_id.encode("utf-8")).hexdigest(),
        regions=tuple(built_regions),
        boundaries=tuple(built_boundaries),
        stable_hash="",
        schema_version=PHASE_SCHEMA_VERSION,
    )
    return PhaseTopologyPath(
        config=proto.config,
        input_transition_hash=proto.input_transition_hash,
        regions=proto.regions,
        boundaries=proto.boundaries,
        stable_hash=_stable_hash_for(proto),
        schema_version=proto.schema_version,
    )


def _base_paths() -> tuple[PhaseTopologyPath, PhaseTopologyPath]:
    path_a = _build_path(
        "a",
        (
            ("stable_region", 0.200000000000, 0, 2),
            ("oscillatory_region", 0.600000000000, 3, 5),
            ("resonant_region", 0.800000000000, 6, 8),
        ),
    )
    path_b = _build_path(
        "b",
        (
            ("stable_region", 0.210000000000, 0, 2),
            ("resonant_region", 0.790000000000, 3, 5),
            ("oscillatory_region", 0.620000000000, 6, 8),
        ),
    )
    return path_a, path_b


def test_same_input_same_bytes() -> None:
    path_a, path_b = _base_paths()
    payloads = tuple(run_region_correspondence_kernel((path_a, path_b))[0].to_canonical_bytes() for _ in range(4))
    assert len(set(payloads)) == 1


def test_same_input_same_hash() -> None:
    path_a, path_b = _base_paths()
    result_a, receipt_a = run_region_correspondence_kernel((path_a, path_b))
    result_b, receipt_b = run_region_correspondence_kernel((path_a, path_b))
    assert result_a.stable_hash == result_b.stable_hash
    assert receipt_a.receipt_hash == receipt_b.receipt_hash


def test_repeated_run_byte_identity() -> None:
    path_a, path_b = _base_paths()
    outputs = tuple(run_region_correspondence_kernel((path_a, path_b)) for _ in range(3))
    result_bytes = tuple(o[0].to_canonical_bytes() for o in outputs)
    receipt_bytes = tuple(o[1].to_canonical_bytes() for o in outputs)
    assert len(set(result_bytes)) == 1
    assert len(set(receipt_bytes)) == 1


def test_broken_lineage_rejection() -> None:
    path_a, path_b = _base_paths()
    broken = PhaseTopologyPath(
        config=path_a.config,
        input_transition_hash=path_a.input_transition_hash,
        regions=path_a.regions,
        boundaries=path_a.boundaries,
        stable_hash="0" * 64,
        schema_version=path_a.schema_version,
    )
    with pytest.raises(ValueError, match="broken lineage"):
        run_region_correspondence_kernel((broken, path_b))


def test_duplicate_mapping_rejection() -> None:
    pair_0 = RegionCorrespondencePair(
        pair_id="a" * 64,
        source_path_index=0,
        target_path_index=1,
        source_region_index=0,
        target_region_index=0,
        source_region_id="s0",
        target_region_id="t0",
        source_region_label="stable_region",
        target_region_label="stable_region",
        region_alignment_score=1.0,
        topology_correspondence_score=1.0,
        boundary_coherence_score=1.0,
        global_correspondence_score=1.0,
    )
    pair_1 = RegionCorrespondencePair(
        pair_id="b" * 64,
        source_path_index=0,
        target_path_index=1,
        source_region_index=1,
        target_region_index=0,
        source_region_id="s1",
        target_region_id="t0",
        source_region_label="oscillatory_region",
        target_region_label="stable_region",
        region_alignment_score=0.5,
        topology_correspondence_score=0.5,
        boundary_coherence_score=0.5,
        global_correspondence_score=0.5,
    )
    proto = RegionCorrespondenceMap(
        source_path_index=0,
        target_path_index=1,
        pairs=(pair_0, pair_1),
        region_alignment_score=0.75,
        topology_correspondence_score=0.75,
        boundary_coherence_score=0.75,
        global_correspondence_score=0.75,
        stable_hash="",
    )
    broken_map = RegionCorrespondenceMap(
        source_path_index=proto.source_path_index,
        target_path_index=proto.target_path_index,
        pairs=proto.pairs,
        region_alignment_score=proto.region_alignment_score,
        topology_correspondence_score=proto.topology_correspondence_score,
        boundary_coherence_score=proto.boundary_coherence_score,
        global_correspondence_score=proto.global_correspondence_score,
        stable_hash=_sha256_hex(proto.to_hash_payload_dict()),
    )

    with pytest.raises(ValueError, match="duplicate region mappings"):
        compute_region_correspondence_similarity(broken_map)


def test_canonical_export_stability() -> None:
    path_a, path_b = _base_paths()
    result, receipt = run_region_correspondence_kernel((path_a, path_b))
    assert result.to_canonical_json() == result.to_canonical_json()
    assert result.to_canonical_bytes() == result.to_canonical_bytes()
    assert receipt.to_canonical_json() == receipt.to_canonical_json()
    assert receipt.to_canonical_bytes() == receipt.to_canonical_bytes()


def test_bounded_metric_validation() -> None:
    path_a, path_b = _base_paths()
    correspondence_map = build_region_correspondence_map(path_a, path_b)
    metrics = compute_region_correspondence_similarity(correspondence_map)
    for value in metrics.values():
        assert 0.0 <= value <= 1.0

    result, receipt = run_region_correspondence_kernel((path_a, path_b))
    assert 0.0 <= result.region_alignment_score <= 1.0
    assert 0.0 <= result.topology_correspondence_score <= 1.0
    assert 0.0 <= result.boundary_coherence_score <= 1.0
    assert 0.0 <= result.global_correspondence_score <= 1.0
    assert 0.0 <= receipt.region_alignment_score <= 1.0
    assert 0.0 <= receipt.topology_correspondence_score <= 1.0
    assert 0.0 <= receipt.boundary_coherence_score <= 1.0
    assert 0.0 <= receipt.global_correspondence_score <= 1.0


def test_tie_break_ordering_integrity() -> None:
    source = _build_path(
        "source",
        (
            ("stable_region", 0.200000000000, 0, 2),
            ("stable_region", 0.200000000000, 3, 5),
        ),
    )
    target = _build_path(
        "target",
        (
            ("stable_region", 0.200000000000, 0, 2),
            ("stable_region", 0.200000000000, 3, 5),
        ),
    )
    correspondence_map = build_region_correspondence_map(source, target)
    indices = tuple((p.source_region_index, p.target_region_index) for p in correspondence_map.pairs)
    assert indices == ((0, 0), (1, 1))


def test_wrapper_manual_equivalence() -> None:
    path_a, path_b = _base_paths()
    manual = build_region_correspondence_map(path_a, path_b)
    wrapped, _ = run_region_correspondence_kernel((path_a, path_b))
    assert len(wrapped.correspondence_maps) == 1
    assert wrapped.correspondence_maps[0].to_canonical_bytes() == manual.to_canonical_bytes()


def test_receipt_integrity() -> None:
    path_a, path_b = _base_paths()
    result, receipt = run_region_correspondence_kernel((path_a, path_b))
    assert receipt.schema_version == SCHEMA_VERSION
    assert receipt.kernel_version == SCHEMA_VERSION
    assert receipt.output_stable_hash == result.stable_hash
    assert receipt.validation_passed is True
    assert receipt.receipt_chain[2] == result.stable_hash
    assert receipt.receipt_chain[0] == result.config.stable_sha256()
