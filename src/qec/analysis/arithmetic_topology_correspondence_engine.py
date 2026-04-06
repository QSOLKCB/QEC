"""v137.8.5 — Arithmetic Topology Correspondence Engine.

Deterministic Layer-4 consumer of topology divergence artifacts.
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
from qec.analysis.topology_divergence_battery import TopologyDivergenceResult

_JSONScalar = str | int | float | bool | None
_JSONValue = _JSONScalar | tuple["_JSONValue", ...] | dict[str, "_JSONValue"]

_SCHEMA_VERSION = 1

ARITHMETIC_TOPOLOGY_CORRESPONDENCE_LAW = "ARITHMETIC_TOPOLOGY_CORRESPONDENCE_LAW"
DETERMINISTIC_WITNESS_ORDERING_RULE = "DETERMINISTIC_WITNESS_ORDERING_RULE"
REPLAY_SAFE_CORRESPONDENCE_IDENTITY_RULE = "REPLAY_SAFE_CORRESPONDENCE_IDENTITY_RULE"
BOUNDED_CORRESPONDENCE_SCORE_RULE = "BOUNDED_CORRESPONDENCE_SCORE_RULE"


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


def _mean(values: tuple[float, ...], default: float = 1.0) -> float:
    if not values:
        return _clamp01(float(default))
    return _clamp01(float(sum(values) / len(values)))


def _validate_unit_interval(value: float, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be finite number in [0.0, 1.0]")
    score = float(value)
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        raise ValueError(f"{name} must be finite number in [0.0, 1.0]")


def _validate_divergence_artifact(divergence_artifact: TopologyDivergenceResult) -> None:
    if not isinstance(divergence_artifact, TopologyDivergenceResult):
        raise ValueError("divergence_artifact must be a TopologyDivergenceResult")
    if divergence_artifact.stable_hash() != divergence_artifact.divergence_hash:
        raise ValueError("divergence_artifact divergence_hash must match stable_hash")
    if divergence_artifact.scenario_count != len(divergence_artifact.scenarios):
        raise ValueError("divergence_artifact scenario_count must match scenarios length")

    scenario_keys = tuple((scenario.scenario_index, scenario.anchor_node_id, scenario.scenario_id) for scenario in divergence_artifact.scenarios)
    if scenario_keys != tuple(sorted(scenario_keys)):
        raise ValueError("divergence_artifact scenarios must be sorted deterministically")
    scenario_indexes = tuple(scenario.scenario_index for scenario in divergence_artifact.scenarios)
    expected_scenario_indexes = tuple(range(len(divergence_artifact.scenarios)))
    if scenario_indexes != expected_scenario_indexes:
        raise ValueError("divergence_artifact scenario_index values must be contiguous and zero-based")
    scenario_ids = tuple(scenario.scenario_id for scenario in divergence_artifact.scenarios)
    if len(set(scenario_ids)) != len(scenario_ids):
        raise ValueError("divergence_artifact scenario_id values must be unique")

    segment_total = 0
    for scenario in divergence_artifact.scenarios:
        if scenario.segment_count != len(scenario.segments):
            raise ValueError("divergence_artifact scenario segment_count must match segments length")

        segment_keys = tuple(
            (segment.segment_index, segment.path_id, segment.start_node_id, segment.end_node_id, segment.segment_id)
            for segment in scenario.segments
        )
        if segment_keys != tuple(sorted(segment_keys)):
            raise ValueError("divergence_artifact scenario segments must be sorted deterministically")

        segment_indexes = tuple(segment.segment_index for segment in scenario.segments)
        expected_segment_indexes = tuple(range(len(scenario.segments)))
        if segment_indexes != expected_segment_indexes:
            raise ValueError("divergence_artifact scenario segment_index values must be contiguous and zero-based")

        segment_ids = tuple(segment.segment_id for segment in scenario.segments)
        if len(set(segment_ids)) != len(segment_ids):
            raise ValueError("divergence_artifact scenario segment_id values must be unique")

        for segment in scenario.segments:
            if segment.scenario_id != scenario.scenario_id:
                raise ValueError("divergence_artifact segment scenario_id must match parent scenario")
            _validate_unit_interval(segment.split_pressure_score, "divergence_artifact split_pressure_score")
            _validate_unit_interval(
                segment.fragmentation_impact_score,
                "divergence_artifact fragmentation_impact_score",
            )
            _validate_unit_interval(segment.segment_divergence_score, "divergence_artifact segment_divergence_score")
        segment_total += scenario.segment_count

        for name, value in (
            ("branch_divergence_score", scenario.branch_divergence_score),
            ("split_entropy_score", scenario.split_entropy_score),
            ("path_fragmentation_score", scenario.path_fragmentation_score),
            ("traversal_resilience_score", scenario.traversal_resilience_score),
            ("overall_divergence_score", scenario.overall_divergence_score),
        ):
            _validate_unit_interval(float(value), f"divergence_artifact scenario {name}")

    if divergence_artifact.segment_count != segment_total:
        raise ValueError("divergence_artifact segment_count must equal sum of scenario segment_count")

    for name, value in (
        ("branch_divergence_score", divergence_artifact.branch_divergence_score),
        ("split_entropy_score", divergence_artifact.split_entropy_score),
        ("path_fragmentation_score", divergence_artifact.path_fragmentation_score),
        ("traversal_resilience_score", divergence_artifact.traversal_resilience_score),
        ("overall_divergence_score", divergence_artifact.overall_divergence_score),
    ):
        _validate_unit_interval(float(value), f"divergence_artifact {name}")


def _validate_optional_lineage(
    *,
    divergence_artifact: TopologyDivergenceResult,
    traversal_artifact: ManifoldTraversalResult | None,
    symmetry_artifact: E8SymmetryResult | None,
    polytope_artifact: PolytopeReasoningResult | None,
    graph_artifact: TopologicalGraphKernelResult | None,
) -> None:
    if traversal_artifact is not None:
        if not isinstance(traversal_artifact, ManifoldTraversalResult):
            raise ValueError("traversal_artifact must be a ManifoldTraversalResult")
        if traversal_artifact.stable_hash() != traversal_artifact.traversal_hash:
            raise ValueError("traversal_artifact traversal_hash must match stable_hash")
        if traversal_artifact.traversal_hash != divergence_artifact.source_traversal_hash:
            raise ValueError("traversal_artifact traversal_hash must match divergence_artifact.source_traversal_hash")
        if traversal_artifact.source_symmetry_hash != divergence_artifact.source_symmetry_hash:
            raise ValueError("traversal_artifact source_symmetry_hash must match divergence_artifact.source_symmetry_hash")
        if traversal_artifact.source_polytope_hash != divergence_artifact.source_polytope_hash:
            raise ValueError("traversal_artifact source_polytope_hash must match divergence_artifact.source_polytope_hash")
        if traversal_artifact.source_graph_hash != divergence_artifact.source_graph_hash:
            raise ValueError("traversal_artifact source_graph_hash must match divergence_artifact.source_graph_hash")

    if symmetry_artifact is not None:
        if not isinstance(symmetry_artifact, E8SymmetryResult):
            raise ValueError("symmetry_artifact must be an E8SymmetryResult")
        if symmetry_artifact.stable_hash() != symmetry_artifact.symmetry_hash:
            raise ValueError("symmetry_artifact symmetry_hash must match stable_hash")
        if symmetry_artifact.symmetry_hash != divergence_artifact.source_symmetry_hash:
            raise ValueError("symmetry_artifact symmetry_hash must match divergence_artifact.source_symmetry_hash")
        if symmetry_artifact.source_polytope_hash != divergence_artifact.source_polytope_hash:
            raise ValueError("symmetry_artifact source_polytope_hash must match divergence_artifact.source_polytope_hash")
        if symmetry_artifact.source_graph_hash != divergence_artifact.source_graph_hash:
            raise ValueError("symmetry_artifact source_graph_hash must match divergence_artifact.source_graph_hash")

    if polytope_artifact is not None:
        if not isinstance(polytope_artifact, PolytopeReasoningResult):
            raise ValueError("polytope_artifact must be a PolytopeReasoningResult")
        if polytope_artifact.stable_hash() != polytope_artifact.polytope_hash:
            raise ValueError("polytope_artifact polytope_hash must match stable_hash")
        if polytope_artifact.polytope_hash != divergence_artifact.source_polytope_hash:
            raise ValueError("polytope_artifact polytope_hash must match divergence_artifact.source_polytope_hash")
        if polytope_artifact.source_graph_hash != divergence_artifact.source_graph_hash:
            raise ValueError("polytope_artifact source_graph_hash must match divergence_artifact.source_graph_hash")

    if graph_artifact is not None:
        if not isinstance(graph_artifact, TopologicalGraphKernelResult):
            raise ValueError("graph_artifact must be a TopologicalGraphKernelResult")
        if graph_artifact.stable_hash() != graph_artifact.graph_hash:
            raise ValueError("graph_artifact graph_hash must match stable_hash")
        if graph_artifact.graph_hash != divergence_artifact.source_graph_hash:
            raise ValueError("graph_artifact graph_hash must match divergence_artifact.source_graph_hash")


@dataclass(frozen=True)
class ArithmeticWitness:
    witness_id: str
    witness_index: int
    scenario_id: str
    anchor_node_id: str
    path_count: int
    split_count: int
    segment_count: int
    arithmetic_mass: float
    divergence_complement_score: float
    witness_consistency_score: float

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "witness_id": self.witness_id,
            "witness_index": self.witness_index,
            "scenario_id": self.scenario_id,
            "anchor_node_id": self.anchor_node_id,
            "path_count": self.path_count,
            "split_count": self.split_count,
            "segment_count": self.segment_count,
            "arithmetic_mass": self.arithmetic_mass,
            "divergence_complement_score": self.divergence_complement_score,
            "witness_consistency_score": self.witness_consistency_score,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")

    def stable_hash(self) -> str:
        return _sha256_hex(self.to_dict())


@dataclass(frozen=True)
class CorrespondencePrimitive:
    primitive_id: str
    primitive_index: int
    witness_id: str
    scenario_id: str
    segment_id: str
    path_id: str
    arithmetic_step: int
    split_pressure_score: float
    segment_divergence_score: float
    arithmetic_alignment_score: float

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "primitive_id": self.primitive_id,
            "primitive_index": self.primitive_index,
            "witness_id": self.witness_id,
            "scenario_id": self.scenario_id,
            "segment_id": self.segment_id,
            "path_id": self.path_id,
            "arithmetic_step": self.arithmetic_step,
            "split_pressure_score": self.split_pressure_score,
            "segment_divergence_score": self.segment_divergence_score,
            "arithmetic_alignment_score": self.arithmetic_alignment_score,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")

    def stable_hash(self) -> str:
        return _sha256_hex(self.to_dict())


@dataclass(frozen=True)
class ArithmeticTopologyCorrespondenceResult:
    schema_version: int
    source_graph_hash: str
    source_polytope_hash: str
    source_symmetry_hash: str
    source_traversal_hash: str
    source_divergence_hash: str
    source_replay_identity_hash: str
    witness_count: int
    primitive_count: int
    witnesses: tuple[ArithmeticWitness, ...]
    primitives: tuple[CorrespondencePrimitive, ...]
    witness_consistency_score: float
    split_arithmetic_alignment_score: float
    divergence_mapping_integrity_score: float
    topology_arithmetic_coherence_score: float
    overall_correspondence_score: float
    law_invariants: tuple[str, ...]
    correspondence_hash: str

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "schema_version": self.schema_version,
            "source_graph_hash": self.source_graph_hash,
            "source_polytope_hash": self.source_polytope_hash,
            "source_symmetry_hash": self.source_symmetry_hash,
            "source_traversal_hash": self.source_traversal_hash,
            "source_divergence_hash": self.source_divergence_hash,
            "source_replay_identity_hash": self.source_replay_identity_hash,
            "witness_count": self.witness_count,
            "primitive_count": self.primitive_count,
            "witnesses": tuple(witness.to_dict() for witness in self.witnesses),
            "primitives": tuple(primitive.to_dict() for primitive in self.primitives),
            "witness_consistency_score": self.witness_consistency_score,
            "split_arithmetic_alignment_score": self.split_arithmetic_alignment_score,
            "divergence_mapping_integrity_score": self.divergence_mapping_integrity_score,
            "topology_arithmetic_coherence_score": self.topology_arithmetic_coherence_score,
            "overall_correspondence_score": self.overall_correspondence_score,
            "law_invariants": self.law_invariants,
            "correspondence_hash": self.correspondence_hash,
        }

    def to_hash_payload_dict(self) -> dict[str, _JSONValue]:
        payload = self.to_dict()
        payload.pop("correspondence_hash")
        return payload

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")

    def stable_hash(self) -> str:
        return _sha256_hex(self.to_hash_payload_dict())


@dataclass(frozen=True)
class ArithmeticTopologyCorrespondenceReceipt:
    schema_version: int
    source_graph_hash: str
    source_polytope_hash: str
    source_symmetry_hash: str
    source_traversal_hash: str
    source_divergence_hash: str
    correspondence_hash: str
    witness_count: int
    primitive_count: int
    overall_correspondence_score: float
    receipt_hash: str

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "schema_version": self.schema_version,
            "source_graph_hash": self.source_graph_hash,
            "source_polytope_hash": self.source_polytope_hash,
            "source_symmetry_hash": self.source_symmetry_hash,
            "source_traversal_hash": self.source_traversal_hash,
            "source_divergence_hash": self.source_divergence_hash,
            "correspondence_hash": self.correspondence_hash,
            "witness_count": self.witness_count,
            "primitive_count": self.primitive_count,
            "overall_correspondence_score": self.overall_correspondence_score,
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


def _build_witnesses(divergence_artifact: TopologyDivergenceResult) -> tuple[ArithmeticWitness, ...]:
    witnesses: list[ArithmeticWitness] = []
    for scenario in divergence_artifact.scenarios:
        path_count = len(scenario.path_ids)
        segment_count = scenario.segment_count
        arithmetic_mass = _clamp01(float((path_count + scenario.split_count + 1) / (segment_count + 1)))
        divergence_complement = _clamp01(float(1.0 - scenario.overall_divergence_score))
        consistency = _clamp01(
            float(
                0.5 * divergence_complement
                + 0.3 * (1.0 - scenario.path_fragmentation_score)
                + 0.2 * scenario.traversal_resilience_score
            )
        )

        witness_payload = {
            "source_divergence_hash": divergence_artifact.divergence_hash,
            "scenario_id": scenario.scenario_id,
            "anchor_node_id": scenario.anchor_node_id,
            "path_count": path_count,
            "split_count": scenario.split_count,
            "segment_count": segment_count,
        }
        witnesses.append(
            ArithmeticWitness(
                witness_id=_sha256_hex(witness_payload),
                witness_index=scenario.scenario_index,
                scenario_id=scenario.scenario_id,
                anchor_node_id=scenario.anchor_node_id,
                path_count=path_count,
                split_count=scenario.split_count,
                segment_count=segment_count,
                arithmetic_mass=arithmetic_mass,
                divergence_complement_score=divergence_complement,
                witness_consistency_score=consistency,
            )
        )

    ordered = tuple(sorted(witnesses, key=lambda w: (w.witness_index, w.anchor_node_id, w.scenario_id, w.witness_id)))
    return tuple(replace(witness, witness_index=index) for index, witness in enumerate(ordered))


def _build_primitives(
    divergence_artifact: TopologyDivergenceResult,
    witness_by_scenario_id: dict[str, ArithmeticWitness],
) -> tuple[CorrespondencePrimitive, ...]:
    primitives: list[CorrespondencePrimitive] = []
    for scenario in divergence_artifact.scenarios:
        witness = witness_by_scenario_id[scenario.scenario_id]
        for segment in scenario.segments:
            arithmetic_step = int(segment.segment_index + 1 + scenario.split_count)
            arithmetic_alignment = _clamp01(
                float(1.0 - (0.6 * segment.segment_divergence_score + 0.4 * segment.split_pressure_score))
            )
            primitive_payload = {
                "source_divergence_hash": divergence_artifact.divergence_hash,
                "scenario_id": scenario.scenario_id,
                "segment_id": segment.segment_id,
                "witness_id": witness.witness_id,
                "arithmetic_step": arithmetic_step,
            }
            primitives.append(
                CorrespondencePrimitive(
                    primitive_id=_sha256_hex(primitive_payload),
                    primitive_index=0,
                    witness_id=witness.witness_id,
                    scenario_id=scenario.scenario_id,
                    segment_id=segment.segment_id,
                    path_id=segment.path_id,
                    arithmetic_step=arithmetic_step,
                    split_pressure_score=segment.split_pressure_score,
                    segment_divergence_score=segment.segment_divergence_score,
                    arithmetic_alignment_score=arithmetic_alignment,
                )
            )

    ordered = tuple(
        sorted(
            primitives,
            key=lambda p: (
                p.scenario_id,
                p.path_id,
                p.arithmetic_step,
                p.segment_id,
                p.primitive_id,
            ),
        )
    )
    return tuple(replace(primitive, primitive_index=index) for index, primitive in enumerate(ordered))


def build_arithmetic_topology_correspondence(
    divergence_artifact: TopologyDivergenceResult,
    *,
    traversal_artifact: ManifoldTraversalResult | None = None,
    symmetry_artifact: E8SymmetryResult | None = None,
    polytope_artifact: PolytopeReasoningResult | None = None,
    graph_artifact: TopologicalGraphKernelResult | None = None,
) -> ArithmeticTopologyCorrespondenceResult:
    _validate_divergence_artifact(divergence_artifact)
    _validate_optional_lineage(
        divergence_artifact=divergence_artifact,
        traversal_artifact=traversal_artifact,
        symmetry_artifact=symmetry_artifact,
        polytope_artifact=polytope_artifact,
        graph_artifact=graph_artifact,
    )

    witnesses = _build_witnesses(divergence_artifact)
    witness_scenario_ids = tuple(w.scenario_id for w in witnesses)
    if len(set(witness_scenario_ids)) != len(witness_scenario_ids):
        raise ValueError("witnesses contain duplicate scenario_id values; divergence_artifact validation must enforce uniqueness")
    witness_by_scenario_id = {witness.scenario_id: witness for witness in witnesses}
    primitives = _build_primitives(divergence_artifact, witness_by_scenario_id)

    witness_consistency_score = _mean(tuple(w.witness_consistency_score for w in witnesses), default=1.0)
    split_arithmetic_alignment_score = _mean(tuple(p.arithmetic_alignment_score for p in primitives), default=1.0)

    expected_segment_count = sum(scenario.segment_count for scenario in divergence_artifact.scenarios)
    mapping_integrity = _clamp01(float(len(primitives) / expected_segment_count)) if expected_segment_count > 0 else 1.0

    topology_arithmetic_coherence_score = _clamp01(
        float(
            0.85 * (1.0 - divergence_artifact.overall_divergence_score)
            + 0.15 * split_arithmetic_alignment_score
        )
    )
    overall_correspondence_score = _mean(
        (
            witness_consistency_score,
            split_arithmetic_alignment_score,
            mapping_integrity,
            topology_arithmetic_coherence_score,
        ),
        default=1.0,
    )

    result = ArithmeticTopologyCorrespondenceResult(
        schema_version=_SCHEMA_VERSION,
        source_graph_hash=divergence_artifact.source_graph_hash,
        source_polytope_hash=divergence_artifact.source_polytope_hash,
        source_symmetry_hash=divergence_artifact.source_symmetry_hash,
        source_traversal_hash=divergence_artifact.source_traversal_hash,
        source_divergence_hash=divergence_artifact.divergence_hash,
        source_replay_identity_hash=divergence_artifact.source_replay_identity_hash,
        witness_count=len(witnesses),
        primitive_count=len(primitives),
        witnesses=witnesses,
        primitives=primitives,
        witness_consistency_score=witness_consistency_score,
        split_arithmetic_alignment_score=split_arithmetic_alignment_score,
        divergence_mapping_integrity_score=mapping_integrity,
        topology_arithmetic_coherence_score=topology_arithmetic_coherence_score,
        overall_correspondence_score=overall_correspondence_score,
        law_invariants=(
            ARITHMETIC_TOPOLOGY_CORRESPONDENCE_LAW,
            DETERMINISTIC_WITNESS_ORDERING_RULE,
            REPLAY_SAFE_CORRESPONDENCE_IDENTITY_RULE,
            BOUNDED_CORRESPONDENCE_SCORE_RULE,
        ),
        correspondence_hash="",
    )
    return replace(result, correspondence_hash=result.stable_hash())


def export_arithmetic_correspondence_bytes(artifact: ArithmeticTopologyCorrespondenceResult) -> bytes:
    if not isinstance(artifact, ArithmeticTopologyCorrespondenceResult):
        raise ValueError("artifact must be an ArithmeticTopologyCorrespondenceResult")
    return artifact.to_canonical_bytes()


def generate_arithmetic_correspondence_receipt(
    artifact: ArithmeticTopologyCorrespondenceResult,
) -> ArithmeticTopologyCorrespondenceReceipt:
    if not isinstance(artifact, ArithmeticTopologyCorrespondenceResult):
        raise ValueError("artifact must be an ArithmeticTopologyCorrespondenceResult")
    receipt = ArithmeticTopologyCorrespondenceReceipt(
        schema_version=artifact.schema_version,
        source_graph_hash=artifact.source_graph_hash,
        source_polytope_hash=artifact.source_polytope_hash,
        source_symmetry_hash=artifact.source_symmetry_hash,
        source_traversal_hash=artifact.source_traversal_hash,
        source_divergence_hash=artifact.source_divergence_hash,
        correspondence_hash=artifact.correspondence_hash,
        witness_count=artifact.witness_count,
        primitive_count=artifact.primitive_count,
        overall_correspondence_score=artifact.overall_correspondence_score,
        receipt_hash="",
    )
    return replace(receipt, receipt_hash=receipt.stable_hash())


__all__ = [
    "ARITHMETIC_TOPOLOGY_CORRESPONDENCE_LAW",
    "BOUNDED_CORRESPONDENCE_SCORE_RULE",
    "DETERMINISTIC_WITNESS_ORDERING_RULE",
    "REPLAY_SAFE_CORRESPONDENCE_IDENTITY_RULE",
    "ArithmeticTopologyCorrespondenceReceipt",
    "ArithmeticTopologyCorrespondenceResult",
    "ArithmeticWitness",
    "CorrespondencePrimitive",
    "build_arithmetic_topology_correspondence",
    "export_arithmetic_correspondence_bytes",
    "generate_arithmetic_correspondence_receipt",
]
