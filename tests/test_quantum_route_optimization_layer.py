import pytest

from qec.analysis.quantum_route_optimization_layer import (
    GENESIS_HASH,
    RouteLedger,
    RouteLedgerEntry,
    append_route_ledger_entry,
    build_weighted_route_lattice,
    compute_deterministic_shortest_path,
    detect_path_divergence,
    normalize_route_inputs,
    run_quantum_route_optimization_layer,
    schedule_shortest_path,
    validate_route_ledger,
)


def _sample_nodes():
    return (
        {"node_id": "A", "node_kind": "entry", "state_label": "sA", "bounded": True},
        {"node_id": "B", "node_kind": "mid", "state_label": "sB", "bounded": True},
        {"node_id": "C", "node_kind": "mid", "state_label": "sC", "bounded": True},
        {"node_id": "D", "node_kind": "terminal", "state_label": "sD", "bounded": True},
    )


def _sample_edges():
    return (
        {"source_node": "A", "target_node": "B", "transition_weight": 1.0, "allowed": True},
        {"source_node": "B", "target_node": "D", "transition_weight": 1.0, "allowed": True},
        {"source_node": "A", "target_node": "C", "transition_weight": 1.0, "allowed": True},
        {"source_node": "C", "target_node": "D", "transition_weight": 1.0, "allowed": True},
    )


def test_normalize_route_inputs_is_deterministic():
    n1, e1 = normalize_route_inputs(_sample_nodes(), _sample_edges())
    n2, e2 = normalize_route_inputs(reversed(_sample_nodes()), reversed(_sample_edges()))
    assert tuple(n.node_id for n in n1) == tuple(n.node_id for n in n2)
    assert tuple((e.source_node, e.target_node, e.transition_weight) for e in e1) == tuple(
        (e.source_node, e.target_node, e.transition_weight) for e in e2
    )


def test_build_weighted_route_lattice_is_stable():
    nodes, edges = normalize_route_inputs(_sample_nodes(), _sample_edges())
    a = build_weighted_route_lattice(nodes, edges)
    b = build_weighted_route_lattice(nodes, edges)
    assert a.to_canonical_json() == b.to_canonical_json()


def test_build_lattice_rejects_unknown_endpoint():
    with pytest.raises(ValueError):
        normalize_route_inputs(_sample_nodes(), ({"source_node": "A", "target_node": "Z", "transition_weight": 1.0},))


def test_build_lattice_rejects_negative_weight():
    with pytest.raises(ValueError):
        normalize_route_inputs(_sample_nodes(), ({"source_node": "A", "target_node": "B", "transition_weight": -0.1},))


def test_shortest_path_is_deterministic():
    nodes, edges = normalize_route_inputs(_sample_nodes(), _sample_edges())
    lattice = build_weighted_route_lattice(nodes, edges)
    r1 = compute_deterministic_shortest_path(lattice, "A", "D")
    r2 = compute_deterministic_shortest_path(lattice, "A", "D")
    assert r1.to_canonical_json() == r2.to_canonical_json()


def test_shortest_path_tie_break_is_stable():
    nodes, edges = normalize_route_inputs(_sample_nodes(), _sample_edges())
    lattice = build_weighted_route_lattice(nodes, edges)
    result = compute_deterministic_shortest_path(lattice, "A", "D")
    assert result.path_nodes == ("A", "B", "D")


def test_unreachable_target_is_reported_cleanly():
    nodes, edges = normalize_route_inputs(_sample_nodes(), ({"source_node": "A", "target_node": "B", "transition_weight": 1.0},))
    lattice = build_weighted_route_lattice(nodes, edges)
    result = compute_deterministic_shortest_path(lattice, "A", "D")
    assert result.reachable is False
    assert result.path_nodes == ()
    assert result.total_weight == 0.0


def test_schedule_shortest_path_is_stable():
    nodes, edges = normalize_route_inputs(_sample_nodes(), _sample_edges())
    lattice = build_weighted_route_lattice(nodes, edges)
    result = compute_deterministic_shortest_path(lattice, "A", "D")
    s1 = schedule_shortest_path(lattice, result)
    s2 = schedule_shortest_path(lattice, result)
    assert tuple(step.to_canonical_json() for step in s1) == tuple(step.to_canonical_json() for step in s2)


def test_divergence_score_is_bounded():
    report = detect_path_divergence(("A", "B", "C"), ("A", "X", "C", "D"))
    assert 0.0 <= report.divergence_score <= 1.0


def test_identical_paths_have_zero_divergence():
    report = detect_path_divergence(("A", "B"), ("A", "B"))
    assert report.divergence_detected is False
    assert report.divergence_score == 0.0


def test_route_ledger_chain_is_stable():
    ledger = RouteLedger(entries=(), head_hash=GENESIS_HASH, chain_valid=True)
    ledger = append_route_ledger_entry(ledger, "route-1", 0.1, 2.0)
    ledger = append_route_ledger_entry(ledger, "route-2", 0.2, 3.0)
    assert validate_route_ledger(ledger)


def test_route_ledger_detects_corruption():
    entry = RouteLedgerEntry(
        sequence_id=0,
        route_hash="route-1",
        parent_hash=GENESIS_HASH,
        divergence_score=0.1,
        total_weight=1.0,
        entry_hash="bad",
    )
    ledger = RouteLedger(entries=(entry,), head_hash="bad-head", chain_valid=True)
    assert not validate_route_ledger(ledger)


def test_append_rejects_malformed_route_ledger():
    entry = RouteLedgerEntry(
        sequence_id=0,
        route_hash="route-1",
        parent_hash=GENESIS_HASH,
        divergence_score=0.1,
        total_weight=1.0,
        entry_hash="bad",
    )
    malformed = RouteLedger(entries=(entry,), head_hash="still-bad", chain_valid=True)
    with pytest.raises(ValueError):
        append_route_ledger_entry(malformed, "route-2", 0.2, 2.0)


def test_same_input_same_bytes():
    args = dict(nodes=_sample_nodes(), edges=_sample_edges(), source_node="A", target_node="D")
    out1 = run_quantum_route_optimization_layer(**args)
    out2 = run_quantum_route_optimization_layer(**args)
    assert tuple(x.to_canonical_json() if hasattr(x, "to_canonical_json") else tuple(s.to_canonical_json() for s in x) for x in out1) == tuple(
        x.to_canonical_json() if hasattr(x, "to_canonical_json") else tuple(s.to_canonical_json() for s in x)
        for x in out2
    )


def test_no_decoder_imports():
    import qec.analysis.quantum_route_optimization_layer as layer

    names = set(layer.__dict__.keys())
    assert "decoder" not in " ".join(sorted(names)).lower()


def test_insertion_order_independence():
    out1 = run_quantum_route_optimization_layer(
        nodes=_sample_nodes(),
        edges=_sample_edges(),
        source_node="A",
        target_node="D",
    )
    out2 = run_quantum_route_optimization_layer(
        nodes=tuple(reversed(_sample_nodes())),
        edges=tuple(reversed(_sample_edges())),
        source_node="A",
        target_node="D",
    )
    assert out1[0].to_canonical_json() == out2[0].to_canonical_json()
    assert out1[1].to_canonical_json() == out2[1].to_canonical_json()
    assert tuple(step.to_canonical_json() for step in out1[2]) == tuple(step.to_canonical_json() for step in out2[2])
    assert out1[3].to_canonical_json() == out2[3].to_canonical_json()
    assert out1[4].to_canonical_json() == out2[4].to_canonical_json()


def test_duplicate_edge_resolution_is_deterministic():
    nodes = _sample_nodes()
    edges = _sample_edges() + (
        {"source_node": "A", "target_node": "B", "transition_weight": 2.0, "allowed": True},
    )
    normalized_nodes, normalized_edges = normalize_route_inputs(nodes, edges)
    lattice = build_weighted_route_lattice(normalized_nodes, normalized_edges)
    result = compute_deterministic_shortest_path(lattice, "A", "D")
    assert result.path_nodes == ("A", "B", "D")
    assert result.total_weight == 2.0


def test_self_loop_rejection():
    with pytest.raises(ValueError):
        normalize_route_inputs(
            _sample_nodes(),
            ({"source_node": "A", "target_node": "A", "transition_weight": 0.0, "allowed": True},),
        )


def test_contradictory_chain_valid_flag_rejected():
    ledger = RouteLedger(entries=(), head_hash=GENESIS_HASH, chain_valid=False)
    with pytest.raises(ValueError):
        validate_route_ledger(ledger)


def test_nan_inf_rejected_in_weights_or_scores():
    with pytest.raises(ValueError):
        normalize_route_inputs(_sample_nodes(), ({"source_node": "A", "target_node": "B", "transition_weight": float("nan")},))
    with pytest.raises(ValueError):
        append_route_ledger_entry(RouteLedger(entries=(), head_hash=GENESIS_HASH, chain_valid=True), "x", float("inf"), 1.0)
