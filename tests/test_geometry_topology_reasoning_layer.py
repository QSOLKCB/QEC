import pytest

from qec.analysis.geometry_topology_reasoning_layer import (
    GeometryLedger,
    PolytopeStateMap,
    append_geometry_ledger_entry,
    build_polytope_state_map,
    compute_topology_aware_route_score,
    detect_attractor_manifold,
    empty_geometry_ledger,
    normalize_geometry_inputs,
    run_geometry_topology_reasoning_layer,
    validate_geometry_ledger,
    validate_polytope_state_map,
)


def _sample_nodes():
    return [
        {
            "node_id": "n2",
            "state_label": "sheet_b",
            "dimension": 1,
            "boundary_score": 0.2,
            "curvature_hint": 0.1,
            "bounded": True,
        },
        {
            "node_id": "n1",
            "state_label": "sheet_a",
            "dimension": 1,
            "boundary_score": 0.1,
            "curvature_hint": 0.3,
            "bounded": True,
        },
        {
            "node_id": "n3",
            "state_label": "sheet_c",
            "dimension": 2,
            "boundary_score": 0.3,
            "curvature_hint": 0.2,
            "bounded": True,
        },
    ]


def _sample_relations():
    return [
        {
            "source_node": "n1",
            "target_node": "n2",
            "transition_cost": 0.2,
            "adjacency_kind": "stable",
            "stable": True,
        },
        {
            "source_node": "n2",
            "target_node": "n1",
            "transition_cost": 0.2,
            "adjacency_kind": "stable",
            "stable": True,
        },
        {
            "source_node": "n2",
            "target_node": "n3",
            "transition_cost": 0.4,
            "adjacency_kind": "bridge",
            "stable": False,
        },
        {
            "source_node": "n3",
            "target_node": "n2",
            "transition_cost": 0.3,
            "adjacency_kind": "stable",
            "stable": True,
        },
    ]


def test_normalize_geometry_inputs_is_deterministic():
    a = normalize_geometry_inputs(_sample_nodes(), _sample_relations())
    b = normalize_geometry_inputs(reversed(_sample_nodes()), reversed(_sample_relations()))
    assert a == b


def test_normalize_rejects_duplicate_node_id():
    nodes = _sample_nodes() + [_sample_nodes()[0].copy()]
    with pytest.raises(ValueError, match="duplicate node_id"):
        normalize_geometry_inputs(nodes, _sample_relations())


def test_normalize_rejects_unknown_relation_endpoint():
    relations = _sample_relations()
    relations[0] = {**relations[0], "target_node": "n404"}
    with pytest.raises(ValueError, match="endpoint"):
        normalize_geometry_inputs(_sample_nodes(), relations)


def test_polytope_state_map_is_deterministic():
    a = build_polytope_state_map(_sample_nodes(), _sample_relations())
    b = build_polytope_state_map(reversed(_sample_nodes()), reversed(_sample_relations()))
    assert a == b
    assert a.map_valid is True


def test_validate_polytope_state_map_detects_corruption():
    good = build_polytope_state_map(_sample_nodes(), _sample_relations())
    bad = PolytopeStateMap(
        nodes=good.nodes,
        relations=good.relations,
        adjacency=tuple(),
        manifold_candidates=good.manifold_candidates,
        map_hash=good.map_hash,
        map_valid=True,
    )
    assert validate_polytope_state_map(bad) is False


def test_route_score_is_bounded():
    poly = build_polytope_state_map(_sample_nodes(), _sample_relations())
    score = compute_topology_aware_route_score(poly, route=("n1", "n2", "n3"), base_route_score=0.7)
    assert 0.0 <= score.final_route_score <= 1.0
    assert 0.0 <= score.topology_penalty <= 1.0
    assert 0.0 <= score.topology_affinity <= 1.0


def test_route_score_rejects_unknown_route_node():
    poly = build_polytope_state_map(_sample_nodes(), _sample_relations())
    with pytest.raises(ValueError, match="unknown route node"):
        compute_topology_aware_route_score(poly, route=("n1", "n404"))


def test_route_score_is_deterministic():
    poly = build_polytope_state_map(_sample_nodes(), _sample_relations())
    a = compute_topology_aware_route_score(poly, route=("n1", "n2", "n3"), base_route_score=0.71)
    b = compute_topology_aware_route_score(poly, route=("n1", "n2", "n3"), base_route_score=0.71)
    assert a == b


def test_attractor_manifold_detection_is_stable():
    poly = build_polytope_state_map(_sample_nodes(), _sample_relations())
    a = detect_attractor_manifold(poly)
    b = detect_attractor_manifold(poly)
    assert a == b
    assert a.attractor_detected is True


def test_no_attractor_path_returns_none_label():
    nodes = _sample_nodes()
    for node in nodes:
        node["boundary_score"] = 0.95
    poly = build_polytope_state_map(nodes, _sample_relations())
    report = detect_attractor_manifold(poly)
    assert report.attractor_detected is False
    assert report.manifold_label == "none"


def test_geometry_ledger_chain_is_stable():
    poly = build_polytope_state_map(_sample_nodes(), _sample_relations())
    ledger = empty_geometry_ledger()
    ledger = append_geometry_ledger_entry(ledger, poly.map_hash, route_score=0.8, manifold_score=0.6)
    ledger = append_geometry_ledger_entry(ledger, poly.map_hash, route_score=0.7, manifold_score=0.5)
    assert validate_geometry_ledger(ledger) is True
    assert ledger.chain_valid is True


def test_geometry_ledger_detects_corruption():
    poly = build_polytope_state_map(_sample_nodes(), _sample_relations())
    ledger = empty_geometry_ledger()
    ledger = append_geometry_ledger_entry(ledger, poly.map_hash, route_score=0.8, manifold_score=0.6)
    bad = GeometryLedger(entries=ledger.entries, head_hash="f" * 64, chain_valid=True)
    assert validate_geometry_ledger(bad) is False


def test_append_rejects_malformed_geometry_ledger():
    ledger = GeometryLedger(entries=tuple(), head_hash="bad", chain_valid=True)
    with pytest.raises(ValueError, match="malformed geometry ledger"):
        append_geometry_ledger_entry(ledger, "0" * 64, route_score=0.5, manifold_score=0.5)


def test_same_input_same_bytes():
    out1 = run_geometry_topology_reasoning_layer(
        _sample_nodes(),
        _sample_relations(),
        route=("n1", "n2", "n3"),
        base_route_score=0.5,
    )
    out2 = run_geometry_topology_reasoning_layer(
        _sample_nodes(),
        _sample_relations(),
        route=("n1", "n2", "n3"),
        base_route_score=0.5,
    )
    assert out1 == out2
    assert out1[0].to_canonical_json() == out2[0].to_canonical_json()


def test_no_decoder_imports():
    import qec.analysis.geometry_topology_reasoning_layer as mod

    names = set(mod.__dict__.keys())
    assert not any("decoder" in name.lower() for name in names)


def test_empty_geometry_ledger_valid_baseline():
    ledger = empty_geometry_ledger()
    assert ledger.head_hash == "0" * 64
    assert ledger.chain_valid is True
    assert validate_geometry_ledger(ledger) is True


def test_route_score_multi_adjacency_kind_same_endpoints():
    """Multiple relations sharing (source, target) but different adjacency_kind.

    When one of the relations is stable, the transition must be counted as stable
    regardless of the order in which relations are processed. A dict keyed only by
    (source_node, target_node) would overwrite the stable relation with the unstable
    one (or vice versa) non-deterministically; the frozenset-based implementation must
    reflect the correct stability for the pair.
    """
    nodes = _sample_nodes()

    # Build two polytopes: one where n1→n2 has ONLY an unstable relation,
    # and one where it has BOTH a stable and an unstable relation.
    # The mixed-relation polytope must score at least as well as the all-unstable one
    # on the n1→n2 route (stable wins → lower unstable_transition_penalty).
    unstable_only_relations = [
        {"source_node": "n1", "target_node": "n2", "transition_cost": 0.5, "adjacency_kind": "bridge", "stable": False},
        {"source_node": "n2", "target_node": "n3", "transition_cost": 0.4, "adjacency_kind": "bridge", "stable": False},
        {"source_node": "n3", "target_node": "n2", "transition_cost": 0.3, "adjacency_kind": "stable", "stable": True},
    ]
    mixed_relations = list(unstable_only_relations) + [
        # Add a stable variant of n1→n2 with a different adjacency_kind.
        {"source_node": "n1", "target_node": "n2", "transition_cost": 0.2, "adjacency_kind": "stable", "stable": True},
    ]

    poly_unstable = build_polytope_state_map(nodes, unstable_only_relations)
    poly_mixed = build_polytope_state_map(nodes, mixed_relations)

    score_unstable = compute_topology_aware_route_score(poly_unstable, route=("n1", "n2"), base_route_score=0.8)
    score_mixed = compute_topology_aware_route_score(poly_mixed, route=("n1", "n2"), base_route_score=0.8)

    # Mixed polytope has a stable relation for n1→n2 so its topology_penalty must be
    # strictly lower than the all-unstable polytope (all else equal for a single-transition route).
    assert score_mixed.topology_penalty < score_unstable.topology_penalty

    # Both scores must remain bounded and deterministic.
    for score in (score_unstable, score_mixed):
        assert 0.0 <= score.final_route_score <= 1.0
        assert 0.0 <= score.topology_penalty <= 1.0
        assert 0.0 <= score.topology_affinity <= 1.0

    score_mixed2 = compute_topology_aware_route_score(poly_mixed, route=("n1", "n2"), base_route_score=0.8)
    assert score_mixed == score_mixed2
