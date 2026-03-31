from qec.analysis.graph_controllability import (
    build_condensation_dag,
    build_state_graph,
    classify_scc_risk,
    find_escape_path,
    run_graph_controllability,
    tarjan_scc,
)


def test_simple_scc_cycle() -> None:
    graph = build_state_graph([("A", "B"), ("B", "A")])
    sccs = tarjan_scc(graph)
    assert sccs == (("A", "B"),)


def test_multiple_sccs() -> None:
    graph = build_state_graph([("A", "B"), ("B", "A"), ("B", "C")])
    sccs = tarjan_scc(graph)
    assert sccs == (("A", "B"), ("C",))


def test_deterministic_scc_ordering() -> None:
    edges = [("D", "E"), ("E", "D"), ("A", "B"), ("B", "A"), ("B", "C")]
    graph = build_state_graph(edges)
    sccs_first = tarjan_scc(graph)
    sccs_second = tarjan_scc(graph)
    assert sccs_first == sccs_second
    assert sccs_first == (("A", "B"), ("C",), ("D", "E"))


def test_condensation_dag_correctness() -> None:
    graph = build_state_graph([("A", "B"), ("B", "A"), ("B", "C"), ("C", "D"), ("D", "C")])
    sccs = tarjan_scc(graph)
    dag = build_condensation_dag(graph, sccs)
    assert sccs == (("A", "B"), ("C", "D"))
    assert dag == {0: [1], 1: []}


def test_shortest_escape_path() -> None:
    graph = build_state_graph(
        [
            ("S", "A"),
            ("S", "B"),
            ("A", "C"),
            ("B", "D"),
            ("C", "SAFE"),
            ("D", "SAFE"),
        ]
    )
    path = find_escape_path(graph, start="S", safe_nodes={"SAFE"})
    assert path == ("S", "A", "C", "SAFE")


def test_deterministic_path_tie_break_unchanged() -> None:
    graph = build_state_graph(
        [
            ("S", "A"),
            ("S", "B"),
            ("A", "SAFE"),
            ("B", "SAFE"),
        ]
    )
    path = find_escape_path(graph, start="S", safe_nodes={"SAFE"})
    assert path == ("S", "A", "SAFE")


def test_no_escape_path() -> None:
    graph = build_state_graph([("A", "B"), ("B", "C")])
    path = find_escape_path(graph, start="A", safe_nodes={"SAFE"})
    assert path == ()


def test_sink_scc_classified_critical() -> None:
    graph = build_state_graph([("A", "B"), ("B", "A"), ("B", "C")])
    sccs = tarjan_scc(graph)
    risk = classify_scc_risk(graph, sccs, safe_nodes=set())
    assert risk == {0: "warning", 1: "critical"}


def test_safe_scc_classification() -> None:
    graph = build_state_graph([("A", "B"), ("B", "A"), ("B", "C")])
    sccs = tarjan_scc(graph)
    risk = classify_scc_risk(graph, sccs, safe_nodes={"C"})
    assert risk == {0: "warning", 1: "safe"}


def test_deterministic_repeatability() -> None:
    edges = [("B", "C"), ("A", "B"), ("C", "A"), ("C", "D")]
    result_first = run_graph_controllability(edges, start="A", safe_nodes={"D"})
    result_second = run_graph_controllability(list(reversed(edges)), start="A", safe_nodes={"D"})

    assert result_first == result_second
    assert result_first["escape_possible"] is True
    assert result_first["escape_path"] == ("A", "B", "C", "D")


def test_deep_graph_iterative_scc_regression() -> None:
    n_nodes = 1500
    edges = [(f"N{i}", f"N{i+1}") for i in range(n_nodes - 1)]
    graph = build_state_graph(edges)

    sccs_first = tarjan_scc(graph)
    sccs_second = tarjan_scc(graph)
    expected = tuple((node,) for node in sorted(graph.keys()))

    assert sccs_first == sccs_second
    assert len(sccs_first) == n_nodes
    assert sccs_first == expected
