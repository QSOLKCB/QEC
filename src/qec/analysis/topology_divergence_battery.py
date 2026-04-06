"""v137.8.4 — Topology Divergence Battery.

Deterministic Layer-4 diagnostics consumer of manifold traversal artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import math
from typing import Any, Mapping

from qec.analysis.e8_symmetry_projection_layer import E8SymmetryResult
from qec.analysis.manifold_traversal_planner import ManifoldTraversalResult
from qec.analysis.polytope_reasoning_engine import PolytopeReasoningResult
from qec.analysis.topological_graph_kernel import TopologicalGraphKernelResult

_JSONScalar = str | int | float | bool | None
_JSONValue = _JSONScalar | tuple["_JSONValue", ...] | dict[str, "_JSONValue"]

_SCHEMA_VERSION = 1

TOPOLOGY_DIVERGENCE_LAW = "TOPOLOGY_DIVERGENCE_LAW"
DETERMINISTIC_SCENARIO_ORDERING_RULE = "DETERMINISTIC_SCENARIO_ORDERING_RULE"
REPLAY_SAFE_DIVERGENCE_IDENTITY_RULE = "REPLAY_SAFE_DIVERGENCE_IDENTITY_RULE"
BOUNDED_DIVERGENCE_SCORE_RULE = "BOUNDED_DIVERGENCE_SCORE_RULE"

# Reserved sentinel used for the baseline (all-paths) scenario.
# This value must never appear as a node_id in a ManifoldTraversalResult.
_BASELINE_SENTINEL = "baseline"


def _canonicalize_json(value: Any) -> _JSONValue:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite float values are not allowed")
        return value
    if isinstance(value, tuple):
        return tuple(_canonicalize_json(v) for v in value)
    if isinstance(value, list):
        return tuple(_canonicalize_json(v) for v in value)
    if isinstance(value, Mapping):
        keys = tuple(value.keys())
        if any(not isinstance(k, str) for k in keys):
            raise ValueError("payload keys must be strings")
        out: dict[str, _JSONValue] = {}
        for key in sorted(keys):
            out[key] = _canonicalize_json(value[key])
        return out
    raise ValueError(f"unsupported canonical payload type: {type(value)!r}")


def _canonical_json(value: Any) -> str:
    return json.dumps(
        _canonicalize_json(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _canonical_bytes(value: Any) -> bytes:
    return _canonical_json(value).encode("utf-8")


def _sha256_hex(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _clamp01(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, float):
        raise ValueError("score must be float")
    if not math.isfinite(value):
        raise ValueError("score must be finite")
    return min(1.0, max(0.0, value))


def _mean(values: tuple[float, ...]) -> float:
    if not values:
        return 0.0
    return _clamp01(float(sum(values) / len(values)))


def _score_from_count(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return _clamp01(float(numerator / denominator))


def _validate_unit_interval(value: float, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be finite number in [0.0, 1.0]")
    score = float(value)
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        raise ValueError(f"{name} must be finite number in [0.0, 1.0]")


def _validate_optional_lineage(
    *,
    traversal_artifact: ManifoldTraversalResult,
    symmetry_artifact: E8SymmetryResult | None,
    polytope_artifact: PolytopeReasoningResult | None,
    graph_artifact: TopologicalGraphKernelResult | None,
) -> None:
    if symmetry_artifact is not None:
        if not isinstance(symmetry_artifact, E8SymmetryResult):
            raise ValueError("symmetry_artifact must be an E8SymmetryResult")
        if symmetry_artifact.stable_hash() != symmetry_artifact.symmetry_hash:
            raise ValueError("symmetry_artifact symmetry_hash must match stable_hash")
        if symmetry_artifact.symmetry_hash != traversal_artifact.source_symmetry_hash:
            raise ValueError("symmetry_artifact symmetry_hash must match traversal_artifact.source_symmetry_hash")
        if symmetry_artifact.source_polytope_hash != traversal_artifact.source_polytope_hash:
            raise ValueError(
                "symmetry_artifact source_polytope_hash must match traversal_artifact.source_polytope_hash"
            )
        if symmetry_artifact.source_graph_hash != traversal_artifact.source_graph_hash:
            raise ValueError("symmetry_artifact source_graph_hash must match traversal_artifact.source_graph_hash")

    if polytope_artifact is not None:
        if not isinstance(polytope_artifact, PolytopeReasoningResult):
            raise ValueError("polytope_artifact must be a PolytopeReasoningResult")
        if polytope_artifact.stable_hash() != polytope_artifact.polytope_hash:
            raise ValueError("polytope_artifact polytope_hash must match stable_hash")
        if polytope_artifact.polytope_hash != traversal_artifact.source_polytope_hash:
            raise ValueError("polytope_artifact polytope_hash must match traversal_artifact.source_polytope_hash")
        if polytope_artifact.source_graph_hash != traversal_artifact.source_graph_hash:
            raise ValueError("polytope_artifact source_graph_hash must match traversal_artifact.source_graph_hash")
        if symmetry_artifact is not None and polytope_artifact.polytope_hash != symmetry_artifact.source_polytope_hash:
            raise ValueError("polytope_artifact polytope_hash must match symmetry_artifact.source_polytope_hash")

    if graph_artifact is not None:
        if not isinstance(graph_artifact, TopologicalGraphKernelResult):
            raise ValueError("graph_artifact must be a TopologicalGraphKernelResult")
        if graph_artifact.stable_hash() != graph_artifact.graph_hash:
            raise ValueError("graph_artifact graph_hash must match stable_hash")
        if graph_artifact.graph_hash != traversal_artifact.source_graph_hash:
            raise ValueError("graph_artifact graph_hash must match traversal_artifact.source_graph_hash")
        if polytope_artifact is not None and graph_artifact.graph_hash != polytope_artifact.source_graph_hash:
            raise ValueError("graph_artifact graph_hash must match polytope_artifact.source_graph_hash")
        if symmetry_artifact is not None and graph_artifact.graph_hash != symmetry_artifact.source_graph_hash:
            raise ValueError("graph_artifact graph_hash must match symmetry_artifact.source_graph_hash")


def _validate_traversal_artifact(traversal_artifact: ManifoldTraversalResult) -> None:
    if not isinstance(traversal_artifact, ManifoldTraversalResult):
        raise ValueError("traversal_artifact must be a ManifoldTraversalResult")

    if traversal_artifact.stable_hash() != traversal_artifact.traversal_hash:
        raise ValueError("traversal_artifact traversal_hash must match stable_hash")

    if traversal_artifact.node_count != len(traversal_artifact.nodes):
        raise ValueError("traversal_artifact node_count must match nodes length")
    if traversal_artifact.path_count != len(traversal_artifact.paths):
        raise ValueError("traversal_artifact path_count must match paths length")

    node_ids = tuple(node.node_id for node in traversal_artifact.nodes)
    if len(set(node_ids)) != len(node_ids):
        raise ValueError("traversal_artifact nodes must have unique node_id values")
    if any(node.node_id == _BASELINE_SENTINEL for node in traversal_artifact.nodes):
        raise ValueError(f"node_id '{_BASELINE_SENTINEL}' is reserved and cannot be used in traversal artifacts")

    node_keys = tuple((node.node_index, node.source_basis_index, node.node_id) for node in traversal_artifact.nodes)
    if node_keys != tuple(sorted(node_keys)):
        raise ValueError("traversal_artifact nodes must be sorted deterministically")

    path_keys = tuple((path.path_index, path.path_id) for path in traversal_artifact.paths)
    if path_keys != tuple(sorted(path_keys)):
        raise ValueError("traversal_artifact paths must be sorted deterministically")

    node_id_set = set(node_ids)
    path_indices: list[int] = []
    for path in traversal_artifact.paths:
        if len(path.node_ids) != path.path_length:
            raise ValueError("traversal_artifact path_length must match len(node_ids)")
        if path.path_length <= 0:
            raise ValueError("traversal_artifact path_length must be positive")
        if any(node_id not in node_id_set for node_id in path.node_ids):
            raise ValueError("traversal_artifact path node_ids must reference known nodes")
        path_indices.append(path.path_index)
        for name, value in (
            ("path_continuity_score", path.path_continuity_score),
            ("path_alignment_score", path.path_alignment_score),
            ("route_integrity_score", path.route_integrity_score),
            ("traversal_efficiency_score", path.traversal_efficiency_score),
            ("path_score", path.path_score),
        ):
            _validate_unit_interval(float(value), f"traversal_artifact {name}")

    if path_indices != list(range(len(path_indices))):
        raise ValueError("traversal_artifact path_index values must be contiguous and zero-based")

    for name, value in (
        ("path_continuity_score", traversal_artifact.path_continuity_score),
        ("manifold_alignment_score", traversal_artifact.manifold_alignment_score),
        ("symmetry_route_integrity_score", traversal_artifact.symmetry_route_integrity_score),
        ("traversal_efficiency_score", traversal_artifact.traversal_efficiency_score),
        ("overall_traversal_score", traversal_artifact.overall_traversal_score),
    ):
        _validate_unit_interval(float(value), f"traversal_artifact {name}")


@dataclass(frozen=True)
class DivergenceSegment:
    segment_id: str
    scenario_id: str
    segment_index: int
    path_id: str
    start_node_id: str
    end_node_id: str
    split_pressure_score: float
    fragmentation_impact_score: float
    segment_divergence_score: float

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "segment_id": self.segment_id,
            "scenario_id": self.scenario_id,
            "segment_index": self.segment_index,
            "path_id": self.path_id,
            "start_node_id": self.start_node_id,
            "end_node_id": self.end_node_id,
            "split_pressure_score": self.split_pressure_score,
            "fragmentation_impact_score": self.fragmentation_impact_score,
            "segment_divergence_score": self.segment_divergence_score,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")

    def stable_hash(self) -> str:
        return _sha256_hex(self.to_dict())


@dataclass(frozen=True)
class DivergenceScenario:
    scenario_id: str
    scenario_index: int
    anchor_node_id: str
    path_ids: tuple[str, ...]
    split_count: int
    segment_count: int
    segments: tuple[DivergenceSegment, ...]
    branch_divergence_score: float
    split_entropy_score: float
    path_fragmentation_score: float
    traversal_resilience_score: float
    overall_divergence_score: float

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "scenario_id": self.scenario_id,
            "scenario_index": self.scenario_index,
            "anchor_node_id": self.anchor_node_id,
            "path_ids": self.path_ids,
            "split_count": self.split_count,
            "segment_count": self.segment_count,
            "segments": tuple(segment.to_dict() for segment in self.segments),
            "branch_divergence_score": self.branch_divergence_score,
            "split_entropy_score": self.split_entropy_score,
            "path_fragmentation_score": self.path_fragmentation_score,
            "traversal_resilience_score": self.traversal_resilience_score,
            "overall_divergence_score": self.overall_divergence_score,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")

    def stable_hash(self) -> str:
        return _sha256_hex(self.to_dict())


@dataclass(frozen=True)
class TopologyDivergenceResult:
    schema_version: int
    source_graph_hash: str
    source_polytope_hash: str
    source_symmetry_hash: str
    source_traversal_hash: str
    source_replay_identity_hash: str
    scenario_count: int
    segment_count: int
    scenarios: tuple[DivergenceScenario, ...]
    branch_divergence_score: float
    split_entropy_score: float
    path_fragmentation_score: float
    traversal_resilience_score: float
    overall_divergence_score: float
    law_invariants: tuple[str, ...]
    divergence_hash: str

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "schema_version": self.schema_version,
            "source_graph_hash": self.source_graph_hash,
            "source_polytope_hash": self.source_polytope_hash,
            "source_symmetry_hash": self.source_symmetry_hash,
            "source_traversal_hash": self.source_traversal_hash,
            "source_replay_identity_hash": self.source_replay_identity_hash,
            "scenario_count": self.scenario_count,
            "segment_count": self.segment_count,
            "scenarios": tuple(scenario.to_dict() for scenario in self.scenarios),
            "branch_divergence_score": self.branch_divergence_score,
            "split_entropy_score": self.split_entropy_score,
            "path_fragmentation_score": self.path_fragmentation_score,
            "traversal_resilience_score": self.traversal_resilience_score,
            "overall_divergence_score": self.overall_divergence_score,
            "law_invariants": self.law_invariants,
            "divergence_hash": self.divergence_hash,
        }

    def to_hash_payload_dict(self) -> dict[str, _JSONValue]:
        payload = self.to_dict()
        payload.pop("divergence_hash")
        return payload

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")

    def stable_hash(self) -> str:
        return _sha256_hex(self.to_hash_payload_dict())


@dataclass(frozen=True)
class TopologyDivergenceReceipt:
    schema_version: int
    source_graph_hash: str
    source_polytope_hash: str
    source_symmetry_hash: str
    source_traversal_hash: str
    divergence_hash: str
    traversal_resilience_score: float
    overall_divergence_score: float
    receipt_hash: str

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "schema_version": self.schema_version,
            "source_graph_hash": self.source_graph_hash,
            "source_polytope_hash": self.source_polytope_hash,
            "source_symmetry_hash": self.source_symmetry_hash,
            "source_traversal_hash": self.source_traversal_hash,
            "divergence_hash": self.divergence_hash,
            "traversal_resilience_score": self.traversal_resilience_score,
            "overall_divergence_score": self.overall_divergence_score,
            "receipt_hash": self.receipt_hash,
        }

    def to_hash_payload_dict(self) -> dict[str, _JSONValue]:
        payload = self.to_dict()
        payload.pop("receipt_hash")
        return payload

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")

    def stable_hash(self) -> str:
        return _sha256_hex(self.to_hash_payload_dict())


def _build_neighbors(traversal_artifact: ManifoldTraversalResult) -> dict[str, tuple[str, ...]]:
    neighbors: dict[str, set[str]] = {node.node_id: set() for node in traversal_artifact.nodes}
    for path in traversal_artifact.paths:
        for left_node_id, right_node_id in zip(path.node_ids[:-1], path.node_ids[1:]):
            neighbors[left_node_id].add(right_node_id)
            neighbors[right_node_id].add(left_node_id)
    return {node_id: tuple(sorted(node_neighbors)) for node_id, node_neighbors in sorted(neighbors.items())}


def _split_nodes(neighbors: dict[str, tuple[str, ...]]) -> tuple[str, ...]:
    return tuple(node_id for node_id, node_neighbors in neighbors.items() if len(node_neighbors) > 2)


def _scenario_paths(
    traversal_artifact: ManifoldTraversalResult,
    anchor_node_id: str,
) -> tuple[tuple[int, str], ...]:
    items: list[tuple[int, str]] = []
    for path in traversal_artifact.paths:
        if anchor_node_id == _BASELINE_SENTINEL or anchor_node_id in path.node_ids:
            items.append((path.path_index, path.path_id))
    return tuple(sorted(items, key=lambda item: (item[0], item[1])))


def _normalized_entropy(values: tuple[float, ...]) -> float:
    if not values:
        return 0.0
    total = float(sum(values))
    if total <= 0.0:
        return 0.0
    probs = tuple(float(v / total) for v in values if v > 0.0)
    if len(probs) <= 1:
        return 0.0
    entropy = float(-sum(p * math.log(p) for p in probs))
    return _clamp01(float(entropy / math.log(len(probs))))


def _anchor_branch_split_values(
    anchor_node_id: str, segments: tuple["DivergenceSegment", ...]
) -> tuple[float, ...]:
    """Return per-branch segment counts for segments adjacent to the anchor node.

    Counts how many segments touch each neighbor of the anchor, giving a
    distribution whose entropy measures true route-split diversity.
    """
    if anchor_node_id == _BASELINE_SENTINEL:
        return ()
    branch_counts: dict[str, float] = {}
    for segment in segments:
        if segment.start_node_id == anchor_node_id:
            branch_node_id = segment.end_node_id
        elif segment.end_node_id == anchor_node_id:
            branch_node_id = segment.start_node_id
        else:
            continue
        branch_counts[branch_node_id] = branch_counts.get(branch_node_id, 0.0) + 1.0
    return tuple(sorted(branch_counts.values()))


def _build_scenario(
    *,
    traversal_artifact: ManifoldTraversalResult,
    neighbors: dict[str, tuple[str, ...]],
    anchor_node_id: str,
    scenario_index: int,
) -> DivergenceScenario:
    path_pairs = _scenario_paths(traversal_artifact, anchor_node_id)
    path_ids = tuple(path_id for _, path_id in path_pairs)
    paths_by_id = {path.path_id: path for path in traversal_artifact.paths}

    segments: list[DivergenceSegment] = []
    for _, path_id in path_pairs:
        path = paths_by_id[path_id]
        for edge_index, (start_node_id, end_node_id) in enumerate(zip(path.node_ids[:-1], path.node_ids[1:])):
            split_num = max(0, len(neighbors[start_node_id]) - 1) + max(0, len(neighbors[end_node_id]) - 1)
            split_pressure_score = _score_from_count(split_num, 6)
            fragmentation_impact_score = _clamp01(
                float(
                    1.0
                    - (
                        0.4 * path.path_continuity_score
                        + 0.35 * path.route_integrity_score
                        + 0.25 * path.traversal_efficiency_score
                    )
                )
            )
            segment_divergence_score = _clamp01(float(0.5 * split_pressure_score + 0.5 * fragmentation_impact_score))
            segment_payload = {
                "source_traversal_hash": traversal_artifact.traversal_hash,
                "scenario_index": scenario_index,
                "path_id": path_id,
                "edge_index": edge_index,
                "start_node_id": start_node_id,
                "end_node_id": end_node_id,
            }
            segments.append(
                DivergenceSegment(
                    segment_id=_sha256_hex(segment_payload),
                    scenario_id="",
                    segment_index=edge_index,
                    path_id=path_id,
                    start_node_id=start_node_id,
                    end_node_id=end_node_id,
                    split_pressure_score=split_pressure_score,
                    fragmentation_impact_score=fragmentation_impact_score,
                    segment_divergence_score=segment_divergence_score,
                )
            )

    segments = sorted(segments, key=lambda s: (s.segment_index, s.path_id, s.start_node_id, s.end_node_id, s.segment_id))
    scenario_id = _sha256_hex(
        {
            "source_traversal_hash": traversal_artifact.traversal_hash,
            "scenario_index": scenario_index,
            "anchor_node_id": anchor_node_id,
            "path_ids": path_ids,
            "segment_ids": tuple(segment.segment_id for segment in segments),
        }
    )
    stable_segments = tuple(
        replace(segment, scenario_id=scenario_id, segment_index=segment_index)
        for segment_index, segment in enumerate(segments)
    )

    branch_divergence_score = _mean(tuple(segment.split_pressure_score for segment in stable_segments))
    split_values = _anchor_branch_split_values(anchor_node_id, stable_segments)
    split_entropy_score = _normalized_entropy(split_values)
    path_fragmentation_score = _mean(tuple(segment.fragmentation_impact_score for segment in stable_segments))

    traversal_resilience_score = _clamp01(
        float(1.0 - (0.45 * path_fragmentation_score + 0.35 * branch_divergence_score + 0.2 * split_entropy_score))
    )
    overall_divergence_score = _clamp01(
        float(0.4 * branch_divergence_score + 0.2 * split_entropy_score + 0.4 * path_fragmentation_score)
    )

    return DivergenceScenario(
        scenario_id=scenario_id,
        scenario_index=scenario_index,
        anchor_node_id=anchor_node_id,
        path_ids=path_ids,
        split_count=1 if anchor_node_id != _BASELINE_SENTINEL else 0,
        segment_count=len(stable_segments),
        segments=stable_segments,
        branch_divergence_score=branch_divergence_score,
        split_entropy_score=split_entropy_score,
        path_fragmentation_score=path_fragmentation_score,
        traversal_resilience_score=traversal_resilience_score,
        overall_divergence_score=overall_divergence_score,
    )


def run_topology_divergence_battery(
    traversal_artifact: ManifoldTraversalResult,
    *,
    symmetry_artifact: E8SymmetryResult | None = None,
    polytope_artifact: PolytopeReasoningResult | None = None,
    graph_artifact: TopologicalGraphKernelResult | None = None,
) -> TopologyDivergenceResult:
    _validate_traversal_artifact(traversal_artifact)
    _validate_optional_lineage(
        traversal_artifact=traversal_artifact,
        symmetry_artifact=symmetry_artifact,
        polytope_artifact=polytope_artifact,
        graph_artifact=graph_artifact,
    )

    neighbors = _build_neighbors(traversal_artifact)
    anchors = (_BASELINE_SENTINEL,) + _split_nodes(neighbors)
    scenarios = tuple(
        _build_scenario(
            traversal_artifact=traversal_artifact,
            neighbors=neighbors,
            anchor_node_id=anchor_node_id,
            scenario_index=scenario_index,
        )
        for scenario_index, anchor_node_id in enumerate(anchors)
    )

    scenario_count = len(scenarios)
    segment_count = sum(scenario.segment_count for scenario in scenarios)
    branch_divergence_score = _mean(tuple(scenario.branch_divergence_score for scenario in scenarios))
    split_entropy_score = _mean(tuple(scenario.split_entropy_score for scenario in scenarios))
    path_fragmentation_score = _mean(tuple(scenario.path_fragmentation_score for scenario in scenarios))
    traversal_resilience_score = _mean(tuple(scenario.traversal_resilience_score for scenario in scenarios))
    overall_divergence_score = _mean(tuple(scenario.overall_divergence_score for scenario in scenarios))

    result = TopologyDivergenceResult(
        schema_version=_SCHEMA_VERSION,
        source_graph_hash=traversal_artifact.source_graph_hash,
        source_polytope_hash=traversal_artifact.source_polytope_hash,
        source_symmetry_hash=traversal_artifact.source_symmetry_hash,
        source_traversal_hash=traversal_artifact.traversal_hash,
        source_replay_identity_hash=traversal_artifact.source_replay_identity_hash,
        scenario_count=scenario_count,
        segment_count=segment_count,
        scenarios=scenarios,
        branch_divergence_score=branch_divergence_score,
        split_entropy_score=split_entropy_score,
        path_fragmentation_score=path_fragmentation_score,
        traversal_resilience_score=traversal_resilience_score,
        overall_divergence_score=overall_divergence_score,
        law_invariants=(
            TOPOLOGY_DIVERGENCE_LAW,
            DETERMINISTIC_SCENARIO_ORDERING_RULE,
            REPLAY_SAFE_DIVERGENCE_IDENTITY_RULE,
            BOUNDED_DIVERGENCE_SCORE_RULE,
        ),
        divergence_hash="",
    )
    return replace(result, divergence_hash=result.stable_hash())


def export_topology_divergence_bytes(artifact: TopologyDivergenceResult) -> bytes:
    if not isinstance(artifact, TopologyDivergenceResult):
        raise ValueError("artifact must be a TopologyDivergenceResult")
    return artifact.to_canonical_bytes()


def generate_topology_divergence_receipt(artifact: TopologyDivergenceResult) -> TopologyDivergenceReceipt:
    if not isinstance(artifact, TopologyDivergenceResult):
        raise ValueError("artifact must be a TopologyDivergenceResult")
    receipt = TopologyDivergenceReceipt(
        schema_version=artifact.schema_version,
        source_graph_hash=artifact.source_graph_hash,
        source_polytope_hash=artifact.source_polytope_hash,
        source_symmetry_hash=artifact.source_symmetry_hash,
        source_traversal_hash=artifact.source_traversal_hash,
        divergence_hash=artifact.divergence_hash,
        traversal_resilience_score=artifact.traversal_resilience_score,
        overall_divergence_score=artifact.overall_divergence_score,
        receipt_hash="",
    )
    return replace(receipt, receipt_hash=receipt.stable_hash())


__all__ = [
    "BOUNDED_DIVERGENCE_SCORE_RULE",
    "DETERMINISTIC_SCENARIO_ORDERING_RULE",
    "REPLAY_SAFE_DIVERGENCE_IDENTITY_RULE",
    "TOPOLOGY_DIVERGENCE_LAW",
    "DivergenceScenario",
    "DivergenceSegment",
    "TopologyDivergenceReceipt",
    "TopologyDivergenceResult",
    "export_topology_divergence_bytes",
    "generate_topology_divergence_receipt",
    "run_topology_divergence_battery",
]
