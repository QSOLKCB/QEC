"""Deterministic Layer-4 geometry/topology reasoning layer.

Theory invariants preserved by this module:

- POLYTOPE_STATE_MAP_LAW:
  Same normalized node/relation structure yields identical canonical map bytes.
- TOPOLOGY_AWARE_ROUTE_SCORING_INVARIANT:
  Route scores are derived only from explicit structural features and bounded in [0, 1].
- ATTRACTOR_MANIFOLD_DETECTION_RULE:
  Attractor manifold detection uses only explicit connectivity, stability, and boundary rules.
- REPLAY_SAFE_GEOMETRY_CHAIN:
  Geometry ledger entries are parent-linked by stable SHA-256 hashes.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

_FLOAT_PRECISION: int = 12
_HASH_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class GeometryStateNode:
    node_id: str
    state_label: str
    dimension: int
    boundary_score: float
    curvature_hint: float
    bounded: bool
    node_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "state_label": self.state_label,
            "dimension": self.dimension,
            "boundary_score": self.boundary_score,
            "curvature_hint": self.curvature_hint,
            "bounded": self.bounded,
            "node_hash": self.node_hash,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())


@dataclass(frozen=True)
class TransitionRelation:
    source_node: str
    target_node: str
    transition_cost: float
    adjacency_kind: str
    stable: bool
    relation_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_node": self.source_node,
            "target_node": self.target_node,
            "transition_cost": self.transition_cost,
            "adjacency_kind": self.adjacency_kind,
            "stable": self.stable,
            "relation_hash": self.relation_hash,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())


@dataclass(frozen=True)
class PolytopeStateMap:
    nodes: tuple[GeometryStateNode, ...]
    relations: tuple[TransitionRelation, ...]
    adjacency: tuple[tuple[str, tuple[str, ...]], ...]
    manifold_candidates: tuple[str, ...]
    map_hash: str
    map_valid: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": [n.to_dict() for n in self.nodes],
            "relations": [r.to_dict() for r in self.relations],
            "adjacency": [[k, list(v)] for k, v in self.adjacency],
            "manifold_candidates": list(self.manifold_candidates),
            "map_hash": self.map_hash,
            "map_valid": self.map_valid,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())


@dataclass(frozen=True)
class TopologyAwareRouteScore:
    base_route_score: float
    topology_penalty: float
    topology_affinity: float
    final_route_score: float
    score_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "base_route_score": self.base_route_score,
            "topology_penalty": self.topology_penalty,
            "topology_affinity": self.topology_affinity,
            "final_route_score": self.final_route_score,
            "score_hash": self.score_hash,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())


@dataclass(frozen=True)
class AttractorManifoldReport:
    attractor_detected: bool
    manifold_label: str
    cluster_nodes: tuple[str, ...]
    manifold_score: float
    report_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "attractor_detected": self.attractor_detected,
            "manifold_label": self.manifold_label,
            "cluster_nodes": list(self.cluster_nodes),
            "manifold_score": self.manifold_score,
            "report_hash": self.report_hash,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())


@dataclass(frozen=True)
class GeometryLedgerEntry:
    sequence_id: int
    map_hash: str
    parent_hash: str
    route_score: float
    manifold_score: float
    entry_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "sequence_id": self.sequence_id,
            "map_hash": self.map_hash,
            "parent_hash": self.parent_hash,
            "route_score": self.route_score,
            "manifold_score": self.manifold_score,
            "entry_hash": self.entry_hash,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())


@dataclass(frozen=True)
class GeometryLedger:
    entries: tuple[GeometryLedgerEntry, ...]
    head_hash: str
    chain_valid: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "entries": [e.to_dict() for e in self.entries],
            "head_hash": self.head_hash,
            "chain_valid": self.chain_valid,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())


@dataclass(frozen=True)
class GeometryTransitionReport:
    structural_stability: float
    boundary_risk: float
    topology_class: str
    deterministic: bool
    report_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "structural_stability": self.structural_stability,
            "boundary_risk": self.boundary_risk,
            "topology_class": self.topology_class,
            "deterministic": self.deterministic,
            "report_hash": self.report_hash,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())


def _canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _hash_sha256(payload: Any) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _canon_str(value: Any, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be str")
    normalized = " ".join(value.strip().split())
    if not normalized:
        raise ValueError(f"{field} must be non-empty")
    return normalized


def _round12(value: float) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError("numeric value must be int/float")
    f = float(value)
    if not math.isfinite(f):
        raise ValueError("numeric value must be finite")
    return round(f, _FLOAT_PRECISION)


def _clamp01(value: float) -> float:
    return _round12(min(1.0, max(0.0, value)))


def _require_mapping(value: Any, expected_len: int, field_names: tuple[str, ...]) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    if isinstance(value, (tuple, list)) and len(value) == expected_len:
        return dict(zip(field_names, value))
    raise ValueError("entry must be mapping or fixed-length tuple/list")


def _node_payload(node: GeometryStateNode) -> dict[str, Any]:
    return {
        "node_id": node.node_id,
        "state_label": node.state_label,
        "dimension": node.dimension,
        "boundary_score": node.boundary_score,
        "curvature_hint": node.curvature_hint,
        "bounded": node.bounded,
    }


def _relation_payload(relation: TransitionRelation) -> dict[str, Any]:
    return {
        "source_node": relation.source_node,
        "target_node": relation.target_node,
        "transition_cost": relation.transition_cost,
        "adjacency_kind": relation.adjacency_kind,
        "stable": relation.stable,
    }


def _map_payload(polytope_map: PolytopeStateMap) -> dict[str, Any]:
    return {
        "nodes": [_node_payload(n) | {"node_hash": n.node_hash} for n in polytope_map.nodes],
        "relations": [_relation_payload(r) | {"relation_hash": r.relation_hash} for r in polytope_map.relations],
        "adjacency": [[k, list(v)] for k, v in polytope_map.adjacency],
        "manifold_candidates": list(polytope_map.manifold_candidates),
    }


def _route_payload(score: TopologyAwareRouteScore) -> dict[str, Any]:
    return {
        "base_route_score": score.base_route_score,
        "topology_penalty": score.topology_penalty,
        "topology_affinity": score.topology_affinity,
        "final_route_score": score.final_route_score,
    }


def _manifold_payload(report: AttractorManifoldReport) -> dict[str, Any]:
    return {
        "attractor_detected": report.attractor_detected,
        "manifold_label": report.manifold_label,
        "cluster_nodes": list(report.cluster_nodes),
        "manifold_score": report.manifold_score,
    }


def _ledger_entry_payload(entry: GeometryLedgerEntry) -> dict[str, Any]:
    return {
        "sequence_id": entry.sequence_id,
        "map_hash": entry.map_hash,
        "parent_hash": entry.parent_hash,
        "route_score": entry.route_score,
        "manifold_score": entry.manifold_score,
    }


def _is_hash(value: str) -> bool:
    return isinstance(value, str) and bool(_HASH_RE.fullmatch(value))


def normalize_geometry_inputs(
    nodes: Iterable[Any], relations: Iterable[Any]
) -> tuple[tuple[GeometryStateNode, ...], tuple[TransitionRelation, ...]]:
    """Normalize and validate geometry nodes and transition relations.

    Duplicate relation handling rule (deterministic): relations are keyed by
    (source_node, target_node, adjacency_kind), and exactly one relation per key
    is retained: the lexicographically-minimal tuple by
    (transition_cost, int(stable), relation_hash).
    """

    normalized_nodes: list[GeometryStateNode] = []
    node_ids: set[str] = set()
    node_fields = (
        "node_id",
        "state_label",
        "dimension",
        "boundary_score",
        "curvature_hint",
        "bounded",
    )
    for raw in nodes:
        data = _require_mapping(raw, 6, node_fields)
        node_id = _canon_str(data.get("node_id"), "node_id")
        if node_id in node_ids:
            raise ValueError(f"duplicate node_id: {node_id}")
        state_label = _canon_str(data.get("state_label"), "state_label")
        dimension = data.get("dimension")
        if not isinstance(dimension, int) or isinstance(dimension, bool) or dimension < 0:
            raise ValueError("dimension must be non-negative int (bool rejected)")
        boundary_score = _clamp01(_round12(data.get("boundary_score")))
        curvature_hint = _round12(data.get("curvature_hint"))
        bounded = data.get("bounded")
        if not isinstance(bounded, bool):
            raise ValueError("bounded must be bool")

        tmp = GeometryStateNode(
            node_id=node_id,
            state_label=state_label,
            dimension=dimension,
            boundary_score=boundary_score,
            curvature_hint=curvature_hint,
            bounded=bounded,
            node_hash="",
        )
        node_hash = _hash_sha256(_node_payload(tmp))
        normalized_nodes.append(GeometryStateNode(**(tmp.__dict__ | {"node_hash": node_hash})))
        node_ids.add(node_id)

    sorted_nodes = tuple(sorted(normalized_nodes, key=lambda n: (n.dimension, n.state_label, n.node_id)))

    relation_fields = (
        "source_node",
        "target_node",
        "transition_cost",
        "adjacency_kind",
        "stable",
    )
    by_relation_key: dict[tuple[str, str, str], TransitionRelation] = {}
    for raw in relations:
        data = _require_mapping(raw, 5, relation_fields)
        source_node = _canon_str(data.get("source_node"), "source_node")
        target_node = _canon_str(data.get("target_node"), "target_node")
        if source_node not in node_ids or target_node not in node_ids:
            raise ValueError("relation endpoint not present in node set")
        transition_cost = _round12(data.get("transition_cost"))
        if transition_cost < 0:
            raise ValueError("transition_cost must be non-negative")
        adjacency_kind = _canon_str(data.get("adjacency_kind"), "adjacency_kind")
        stable = data.get("stable")
        if not isinstance(stable, bool):
            raise ValueError("stable must be bool")

        tmp = TransitionRelation(
            source_node=source_node,
            target_node=target_node,
            transition_cost=transition_cost,
            adjacency_kind=adjacency_kind,
            stable=stable,
            relation_hash="",
        )
        relation_hash = _hash_sha256(_relation_payload(tmp))
        relation = TransitionRelation(**(tmp.__dict__ | {"relation_hash": relation_hash}))
        rel_key = (source_node, target_node, adjacency_kind)
        current = by_relation_key.get(rel_key)
        if current is None:
            by_relation_key[rel_key] = relation
        else:
            choose_key = (relation.transition_cost, int(relation.stable), relation.relation_hash)
            current_key = (current.transition_cost, int(current.stable), current.relation_hash)
            if choose_key < current_key:
                by_relation_key[rel_key] = relation

    sorted_relations = tuple(
        sorted(
            by_relation_key.values(),
            key=lambda r: (r.source_node, r.target_node, r.adjacency_kind, r.transition_cost),
        )
    )
    return sorted_nodes, sorted_relations


def build_polytope_state_map(
    nodes: Iterable[Any], relations: Iterable[Any]
) -> PolytopeStateMap:
    normalized_nodes, normalized_relations = normalize_geometry_inputs(nodes, relations)
    outgoing: dict[str, list[str]] = {n.node_id: [] for n in normalized_nodes}
    stable_out_degree: dict[str, int] = {n.node_id: 0 for n in normalized_nodes}
    stable_in_degree: dict[str, int] = {n.node_id: 0 for n in normalized_nodes}

    for relation in normalized_relations:
        outgoing[relation.source_node].append(relation.target_node)
        if relation.stable:
            stable_out_degree[relation.source_node] += 1
            stable_in_degree[relation.target_node] += 1

    adjacency = tuple(
        (node_id, tuple(sorted(set(targets))))
        for node_id, targets in sorted(outgoing.items(), key=lambda kv: kv[0])
    )

    node_lookup = {n.node_id: n for n in normalized_nodes}
    candidates = sorted(
        node_id
        for node_id in node_lookup
        if stable_out_degree[node_id] > 0
        and stable_in_degree[node_id] > 0
        and node_lookup[node_id].boundary_score <= 0.4
    )
    polytope = PolytopeStateMap(
        nodes=normalized_nodes,
        relations=normalized_relations,
        adjacency=adjacency,
        manifold_candidates=tuple(candidates),
        map_hash="",
        map_valid=True,
    )
    map_hash = _hash_sha256(_map_payload(polytope))
    with_hash = PolytopeStateMap(**(polytope.__dict__ | {"map_hash": map_hash}))
    map_valid = validate_polytope_state_map(with_hash)
    return PolytopeStateMap(**(with_hash.__dict__ | {"map_valid": map_valid}))


def validate_polytope_state_map(polytope_map: PolytopeStateMap) -> bool:
    recomputed_valid = _validate_polytope_state_map_structure(polytope_map)
    if polytope_map.map_valid != recomputed_valid:
        return False
    return recomputed_valid


def _validate_polytope_state_map_structure(polytope_map: PolytopeStateMap) -> bool:
    try:
        node_ids = tuple(n.node_id for n in polytope_map.nodes)
        if len(node_ids) != len(set(node_ids)):
            return False

        sorted_nodes = tuple(
            sorted(polytope_map.nodes, key=lambda n: (n.dimension, n.state_label, n.node_id))
        )
        if sorted_nodes != polytope_map.nodes:
            return False

        for node in polytope_map.nodes:
            if node.node_hash != _hash_sha256(_node_payload(node)):
                return False

        ids = set(node_ids)
        sorted_relations = tuple(
            sorted(
                polytope_map.relations,
                key=lambda r: (r.source_node, r.target_node, r.adjacency_kind, r.transition_cost),
            )
        )
        if sorted_relations != polytope_map.relations:
            return False
        seen_relation_keys: set[tuple[str, str, str]] = set()
        for rel in polytope_map.relations:
            if rel.source_node not in ids or rel.target_node not in ids:
                return False
            rel_key = (rel.source_node, rel.target_node, rel.adjacency_kind)
            if rel_key in seen_relation_keys:
                return False
            seen_relation_keys.add(rel_key)
            if rel.relation_hash != _hash_sha256(_relation_payload(rel)):
                return False

        expected_outgoing: dict[str, set[str]] = {node_id: set() for node_id in node_ids}
        for rel in polytope_map.relations:
            expected_outgoing[rel.source_node].add(rel.target_node)
        expected_adjacency = tuple(
            (node_id, tuple(sorted(expected_outgoing[node_id])))
            for node_id in sorted(expected_outgoing)
        )
        if expected_adjacency != polytope_map.adjacency:
            return False

        stable_out: dict[str, int] = {node_id: 0 for node_id in node_ids}
        stable_in: dict[str, int] = {node_id: 0 for node_id in node_ids}
        boundaries = {n.node_id: n.boundary_score for n in polytope_map.nodes}
        for rel in polytope_map.relations:
            if rel.stable:
                stable_out[rel.source_node] += 1
                stable_in[rel.target_node] += 1
        expected_candidates = tuple(
            sorted(
                node_id
                for node_id in node_ids
                if stable_out[node_id] > 0 and stable_in[node_id] > 0 and boundaries[node_id] <= 0.4
            )
        )
        if expected_candidates != polytope_map.manifold_candidates:
            return False

        if polytope_map.map_hash != _hash_sha256(_map_payload(polytope_map)):
            return False
        if not _is_hash(polytope_map.map_hash):
            return False
        return True
    except Exception:
        return False


def compute_topology_aware_route_score(
    polytope_map: PolytopeStateMap,
    route: Iterable[str],
    base_route_score: float = 0.5,
) -> TopologyAwareRouteScore:
    if not validate_polytope_state_map(polytope_map):
        raise ValueError("invalid polytope map")

    base = _clamp01(_round12(base_route_score))
    route_nodes = tuple(_canon_str(v, "route_node") for v in route)
    if not route_nodes:
        raise ValueError("route must be non-empty")

    node_lookup = {n.node_id: n for n in polytope_map.nodes}
    for node_id in route_nodes:
        if node_id not in node_lookup:
            raise ValueError(f"unknown route node: {node_id}")

    relation_lookup: dict[tuple[str, str], TransitionRelation] = {
        (r.source_node, r.target_node): r for r in polytope_map.relations
    }

    boundary_risk = _round12(
        sum(node_lookup[node_id].boundary_score for node_id in route_nodes) / len(route_nodes)
    )

    transitions = tuple(zip(route_nodes[:-1], route_nodes[1:]))
    if transitions:
        unstable_count = sum(
            1 for pair in transitions if pair not in relation_lookup or not relation_lookup[pair].stable
        )
        jump_count = sum(
            1
            for s, t in transitions
            if node_lookup[s].dimension != node_lookup[t].dimension
        )
        stable_count = len(transitions) - unstable_count
        unstable_transition_penalty = _round12(unstable_count / len(transitions))
        dimension_jump_penalty = _round12(jump_count / len(transitions))
        stable_adjacency_ratio = _round12(stable_count / len(transitions))
    else:
        unstable_transition_penalty = 0.0
        dimension_jump_penalty = 0.0
        stable_adjacency_ratio = 1.0

    manifold_candidates = set(polytope_map.manifold_candidates)
    overlap = sum(1 for node_id in route_nodes if node_id in manifold_candidates)
    manifold_overlap_ratio = _round12(overlap / len(route_nodes))

    topology_penalty = _clamp01(
        0.4 * boundary_risk + 0.3 * dimension_jump_penalty + 0.3 * unstable_transition_penalty
    )
    topology_affinity = _clamp01(0.5 * stable_adjacency_ratio + 0.5 * manifold_overlap_ratio)
    final_route_score = _clamp01(base - 0.4 * topology_penalty + 0.3 * topology_affinity)

    score = TopologyAwareRouteScore(
        base_route_score=base,
        topology_penalty=topology_penalty,
        topology_affinity=topology_affinity,
        final_route_score=final_route_score,
        score_hash="",
    )
    return TopologyAwareRouteScore(
        **(score.__dict__ | {"score_hash": _hash_sha256(_route_payload(score))})
    )


def detect_attractor_manifold(polytope_map: PolytopeStateMap) -> AttractorManifoldReport:
    if not validate_polytope_state_map(polytope_map):
        raise ValueError("invalid polytope map")

    node_lookup = {n.node_id: n for n in polytope_map.nodes}
    ids = tuple(sorted(node_lookup))
    stable_edges = {(r.source_node, r.target_node) for r in polytope_map.relations if r.stable}

    candidates = [
        node_id
        for node_id in ids
        if node_id in polytope_map.manifold_candidates and node_lookup[node_id].boundary_score <= 0.4
    ]
    cluster_nodes: tuple[str, ...] = tuple(candidates)

    if len(cluster_nodes) < 2:
        report = AttractorManifoldReport(
            attractor_detected=False,
            manifold_label="none",
            cluster_nodes=tuple(),
            manifold_score=0.0,
            report_hash="",
        )
        return AttractorManifoldReport(
            **(report.__dict__ | {"report_hash": _hash_sha256(_manifold_payload(report))})
        )

    possible_pairs = len(cluster_nodes) * (len(cluster_nodes) - 1)
    stable_internal = sum(
        1 for s in cluster_nodes for t in cluster_nodes if s != t and (s, t) in stable_edges
    )
    cohesion = _round12(stable_internal / possible_pairs) if possible_pairs else 0.0
    avg_boundary = _round12(sum(node_lookup[n].boundary_score for n in cluster_nodes) / len(cluster_nodes))

    manifold_score = _clamp01(0.6 * cohesion + 0.4 * (1.0 - avg_boundary))

    if cohesion >= 0.8 and len(cluster_nodes) >= 3:
        label = "compact_cluster"
    elif cohesion >= 0.7:
        label = "stable_loop"
    else:
        label = "local_sheet"

    report = AttractorManifoldReport(
        attractor_detected=manifold_score >= 0.55,
        manifold_label=label if manifold_score >= 0.55 else "none",
        cluster_nodes=cluster_nodes if manifold_score >= 0.55 else tuple(),
        manifold_score=manifold_score if manifold_score >= 0.55 else 0.0,
        report_hash="",
    )
    return AttractorManifoldReport(
        **(report.__dict__ | {"report_hash": _hash_sha256(_manifold_payload(report))})
    )


def empty_geometry_ledger() -> GeometryLedger:
    return GeometryLedger(entries=tuple(), head_hash="0" * 64, chain_valid=True)


def append_geometry_ledger_entry(
    ledger: GeometryLedger,
    map_hash: str,
    route_score: float,
    manifold_score: float,
) -> GeometryLedger:
    if not validate_geometry_ledger(ledger):
        raise ValueError("malformed geometry ledger")
    if not _is_hash(map_hash):
        raise ValueError("map_hash must be lowercase hex SHA-256")

    score_route = _clamp01(_round12(route_score))
    score_manifold = _clamp01(_round12(manifold_score))

    parent_hash = ledger.head_hash
    sequence_id = len(ledger.entries)
    if not _is_hash(parent_hash):
        raise ValueError("ledger head_hash must be lowercase hex SHA-256")

    entry = GeometryLedgerEntry(
        sequence_id=sequence_id,
        map_hash=map_hash,
        parent_hash=parent_hash,
        route_score=score_route,
        manifold_score=score_manifold,
        entry_hash="",
    )
    entry_hash = _hash_sha256(_ledger_entry_payload(entry))
    full = GeometryLedgerEntry(**(entry.__dict__ | {"entry_hash": entry_hash}))
    updated_entries = ledger.entries + (full,)
    updated = GeometryLedger(entries=updated_entries, head_hash=full.entry_hash, chain_valid=True)
    return GeometryLedger(**(updated.__dict__ | {"chain_valid": validate_geometry_ledger(updated)}))


def validate_geometry_ledger(ledger: GeometryLedger) -> bool:
    recomputed_valid = _validate_geometry_ledger_structure(ledger)
    if ledger.chain_valid != recomputed_valid:
        return False
    return recomputed_valid


def _validate_geometry_ledger_structure(ledger: GeometryLedger) -> bool:
    try:
        if not _is_hash(ledger.head_hash):
            return False
        parent = "0" * 64
        for idx, entry in enumerate(ledger.entries):
            if entry.sequence_id != idx:
                return False
            if entry.parent_hash != parent:
                return False
            if not _is_hash(entry.map_hash) or not _is_hash(entry.parent_hash) or not _is_hash(entry.entry_hash):
                return False
            for score in (entry.route_score, entry.manifold_score):
                if not math.isfinite(score) or score < 0.0 or score > 1.0:
                    return False
            if entry.entry_hash != _hash_sha256(_ledger_entry_payload(entry)):
                return False
            parent = entry.entry_hash
        expected_head = ledger.entries[-1].entry_hash if ledger.entries else "0" * 64
        if ledger.head_hash != expected_head:
            return False
        return True
    except Exception:
        return False


def run_geometry_topology_reasoning_layer(
    nodes: Iterable[Any],
    relations: Iterable[Any],
    route: Iterable[str] | None = None,
    base_route_score: float = 0.5,
    prior_geometry_ledger: GeometryLedger | None = None,
) -> tuple[
    PolytopeStateMap,
    TopologyAwareRouteScore,
    AttractorManifoldReport,
    GeometryTransitionReport,
    GeometryLedger,
]:
    polytope = build_polytope_state_map(nodes, relations)
    score = compute_topology_aware_route_score(
        polytope,
        route if route is not None else tuple(n.node_id for n in polytope.nodes),
        base_route_score=base_route_score,
    )
    attractor = detect_attractor_manifold(polytope)

    route_nodes = tuple(_canon_str(v, "route_node") for v in (route if route is not None else tuple(n.node_id for n in polytope.nodes)))
    node_lookup = {n.node_id: n for n in polytope.nodes}
    boundary_risk = _round12(sum(node_lookup[n].boundary_score for n in route_nodes) / len(route_nodes))
    structural_stability = _clamp01(1.0 - boundary_risk)

    if attractor.attractor_detected:
        topology_class = attractor.manifold_label
    elif score.topology_penalty >= 0.6:
        topology_class = "boundary_fragile"
    else:
        topology_class = "open_graph"

    report = GeometryTransitionReport(
        structural_stability=structural_stability,
        boundary_risk=boundary_risk,
        topology_class=topology_class,
        deterministic=True,
        report_hash="",
    )
    report = GeometryTransitionReport(
        **(
            report.__dict__
            | {
                "report_hash": _hash_sha256(
                    {
                        "structural_stability": report.structural_stability,
                        "boundary_risk": report.boundary_risk,
                        "topology_class": report.topology_class,
                        "deterministic": report.deterministic,
                    }
                )
            }
        )
    )

    ledger = prior_geometry_ledger if prior_geometry_ledger is not None else empty_geometry_ledger()
    updated_ledger = append_geometry_ledger_entry(
        ledger,
        map_hash=polytope.map_hash,
        route_score=score.final_route_score,
        manifold_score=attractor.manifold_score,
    )

    return polytope, score, attractor, report, updated_ledger
