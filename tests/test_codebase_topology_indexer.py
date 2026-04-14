import hashlib

import pytest

from qec.memory.codebase_topology_indexer import (
    build_codebase_topology_index,
    normalize_codebase_topology_input,
    traverse_codebase_topology_index,
    validate_codebase_topology_index,
)


def _h(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sample_topology_payload():
    nodes = [
        {"node_id": "pkg.core", "node_kind": "package", "module_path": "qec/core", "lineage_hash": _h("pkg.core"), "index_epoch": 0},
        {"node_id": "mod.alpha", "node_kind": "module", "module_path": "qec/core/alpha.py", "lineage_hash": _h("mod.alpha"), "index_epoch": 0},
        {"node_id": "mod.beta", "node_kind": "module", "module_path": "qec/core/beta.py", "lineage_hash": _h("mod.beta"), "index_epoch": 0},
        {"node_id": "rel.137.16.2", "node_kind": "release", "module_path": "releases/v137.16.2", "lineage_hash": _h("rel.137.16.2"), "index_epoch": 0},
        {"node_id": "rel.137.16.1", "node_kind": "release", "module_path": "releases/v137.16.1", "lineage_hash": _h("rel.137.16.1"), "index_epoch": 0},
        {"node_id": "test.alpha", "node_kind": "test", "module_path": "tests/test_alpha.py", "lineage_hash": _h("test.alpha"), "index_epoch": 0},
    ]
    edges = [
        {"edge_id": "e1", "source_node_id": "mod.alpha", "target_node_id": "pkg.core", "relationship_kind": "belongs_to", "edge_weight": 1.0, "index_epoch": 0},
        {"edge_id": "e2", "source_node_id": "mod.beta", "target_node_id": "pkg.core", "relationship_kind": "belongs_to", "edge_weight": 1.0, "index_epoch": 0},
        {"edge_id": "e3", "source_node_id": "mod.alpha", "target_node_id": "mod.beta", "relationship_kind": "depends_on", "edge_weight": 2.0, "index_epoch": 0},
        {"edge_id": "e4", "source_node_id": "rel.137.16.2", "target_node_id": "rel.137.16.1", "relationship_kind": "supersedes", "edge_weight": 1.0, "index_epoch": 0},
        {"edge_id": "e5", "source_node_id": "mod.alpha", "target_node_id": "test.alpha", "relationship_kind": "tested_by", "edge_weight": 1.0, "index_epoch": 0},
    ]
    return {"index_id": "idx.v137.16.2", "nodes": nodes, "edges": edges}


def test_repeated_run_byte_identity_and_hash_identity():
    payload = _sample_topology_payload()
    a = build_codebase_topology_index(payload)
    b = build_codebase_topology_index(payload)
    assert a.to_canonical_bytes() == b.to_canonical_bytes()
    assert a.index_hash == b.index_hash


def test_duplicate_node_rejection():
    payload = _sample_topology_payload()
    payload["nodes"].append(dict(payload["nodes"][0]))
    with pytest.raises(ValueError, match="duplicate topology node id"):
        normalize_codebase_topology_input(payload)


def test_duplicate_edge_rejection():
    payload = _sample_topology_payload()
    payload["edges"].append(dict(payload["edges"][0]))
    with pytest.raises(ValueError, match="duplicate topology edge id"):
        normalize_codebase_topology_input(payload)


def test_invalid_hierarchy_references_rejection():
    payload = _sample_topology_payload()
    payload["edges"].append(
        {
            "edge_id": "e999",
            "source_node_id": "rel.137.16.2",
            "target_node_id": "mod.alpha",
            "relationship_kind": "belongs_to",
            "edge_weight": 1.0,
            "index_epoch": 0,
        }
    )
    with pytest.raises(ValueError, match="invalid hierarchy relationship"):
        build_codebase_topology_index(payload)


def test_deterministic_hierarchy_traversal():
    index_obj = build_codebase_topology_index(_sample_topology_payload())
    a = traverse_codebase_topology_index(index_obj, "hierarchy")
    b = traverse_codebase_topology_index(index_obj, "hierarchy")
    assert a.visited_nodes == b.visited_nodes
    assert a.visited_edges == b.visited_edges
    assert a.traversal_hash == b.traversal_hash


def test_deterministic_dependency_traversal():
    index_obj = build_codebase_topology_index(_sample_topology_payload())
    a = traverse_codebase_topology_index(index_obj, "dependency")
    b = traverse_codebase_topology_index(index_obj, "dependency")
    assert a.visited_nodes == b.visited_nodes
    assert a.visited_edges == b.visited_edges
    assert a.traversal_hash == b.traversal_hash


def test_deterministic_lineage_traversal():
    index_obj = build_codebase_topology_index(_sample_topology_payload())
    a = traverse_codebase_topology_index(index_obj, "lineage")
    b = traverse_codebase_topology_index(index_obj, "lineage")
    assert a.visited_nodes == b.visited_nodes
    assert a.visited_edges == b.visited_edges
    assert a.traversal_hash == b.traversal_hash


def test_canonical_export_stability():
    index_obj = build_codebase_topology_index(_sample_topology_payload())
    first = index_obj.to_canonical_json()
    second = index_obj.to_canonical_json()
    assert first == second


def test_deterministic_coverage_traversal():
    index_obj = build_codebase_topology_index(_sample_topology_payload())
    a = traverse_codebase_topology_index(index_obj, "coverage")
    b = traverse_codebase_topology_index(index_obj, "coverage")
    assert a.visited_nodes == b.visited_nodes
    assert a.visited_edges == b.visited_edges
    assert a.traversal_hash == b.traversal_hash
    assert "test.alpha" in a.visited_nodes
    assert "e5" in a.visited_edges


def test_invalid_traversal_mode_rejection():
    index_obj = build_codebase_topology_index(_sample_topology_payload())
    with pytest.raises(ValueError, match="unsupported traversal mode"):
        traverse_codebase_topology_index(index_obj, "nonexistent_mode")


def test_validation_report_valid():
    index_obj = build_codebase_topology_index(_sample_topology_payload())
    report = validate_codebase_topology_index(index_obj)
    assert report.is_valid is True
    assert report.violations == ()
