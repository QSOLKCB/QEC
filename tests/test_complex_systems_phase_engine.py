from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from qec.analysis.complex_systems_phase_engine import (
    AttractorTransitionEdge,
    GENESIS_HASH,
    PhaseLedger,
    PhaseLedgerEntry,
    append_phase_ledger_entry,
    build_attractor_transition_graph,
    compute_bifurcation_warning_score,
    detect_phase_state,
    normalize_phase_inputs,
    run_complex_systems_phase_engine,
    validate_phase_ledger,
)


def _stability() -> dict[str, float]:
    return {"alpha": 0.91, "beta": 0.87, "gamma": 0.89}


def _perturbation() -> dict[str, float]:
    return {"p2": 0.08, "p1": 0.12}


def test_deterministic_normalization() -> None:
    a = normalize_phase_inputs({"b": 0.2, "a": 0.1}, {"y": 0.4, "x": 0.3})
    b = normalize_phase_inputs({"a": 0.1, "b": 0.2}, {"x": 0.3, "y": 0.4})
    assert a == b


def test_stable_phase_classification() -> None:
    snap = detect_phase_state(_stability(), _perturbation())
    assert snap.phase_label == "stable"


def test_bounded_warning_score() -> None:
    score = compute_bifurcation_warning_score(0.95, 0.1, 0.8, 0.7)
    assert 0.0 <= score <= 1.0
    assert score == 1.0


def test_graph_determinism() -> None:
    edges = (
        ("s1", "s2", 0.3, True),
        ("s2", "s3", 0.5, False),
    )
    g1 = build_attractor_transition_graph(["s3", "s1", "s2"], edges)
    g2 = build_attractor_transition_graph(["s2", "s1", "s3"], reversed(edges))
    assert g1.to_canonical_json() == g2.to_canonical_json()


def test_duplicate_edge_rejection() -> None:
    with pytest.raises(ValueError, match="duplicate edge"):
        build_attractor_transition_graph(
            ["a", "b"],
            [("a", "b", 0.2, True), ("a", "b", 0.4, False)],
        )


def test_same_input_same_bytes() -> None:
    out1 = run_complex_systems_phase_engine(_stability(), _perturbation(), noise_level=0.1)
    out2 = run_complex_systems_phase_engine(_stability(), _perturbation(), noise_level=0.1)
    blob1 = "|".join(part.to_canonical_json() for part in out1)
    blob2 = "|".join(part.to_canonical_json() for part in out2)
    assert blob1 == blob2


def test_ledger_chain_stability() -> None:
    phase, warning, _, ledger1 = run_complex_systems_phase_engine(
        _stability(),
        _perturbation(),
        attractor_edges=[("a", "b", 0.2, True)],
        noise_level=0.1,
    )
    ledger2 = append_phase_ledger_entry(phase, warning, ledger1)
    assert validate_phase_ledger(ledger2)
    assert ledger2.entries[0].parent_hash == GENESIS_HASH


def test_corruption_detection() -> None:
    _, _, _, ledger = run_complex_systems_phase_engine(_stability(), _perturbation(), noise_level=0.1)
    bad_entry = PhaseLedgerEntry(
        sequence_id=0,
        phase_hash=ledger.entries[0].phase_hash,
        parent_hash="bad-parent",
        warning_score=ledger.entries[0].warning_score,
        stability_score=ledger.entries[0].stability_score,
    )
    corrupted = PhaseLedger(entries=(bad_entry,), head_hash=ledger.head_hash, chain_valid=True)
    assert not validate_phase_ledger(corrupted)


def test_no_decoder_imports() -> None:
    source = Path("src/qec/analysis/complex_systems_phase_engine.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
            continue
        if isinstance(node, ast.ImportFrom) and node.module is not None:
            imports.append(node.module)
            imports.extend(f"{node.module}.{alias.name}" for alias in node.names)
    assert not any(name.startswith("qec.decoder") for name in imports)


def test_insertion_order_independence() -> None:
    out_a = run_complex_systems_phase_engine(
        {"a": 0.9, "b": 0.8, "c": 0.85},
        {"x": 0.1, "y": 0.2},
        attractor_edges=[AttractorTransitionEdge("sA", "sB", 0.4, True)],
        noise_level=0.15,
    )
    out_b = run_complex_systems_phase_engine(
        {"c": 0.85, "a": 0.9, "b": 0.8},
        {"y": 0.2, "x": 0.1},
        attractor_edges=[AttractorTransitionEdge("sA", "sB", 0.4, True)],
        noise_level=0.15,
    )
    assert json.dumps([x.to_dict() for x in out_a], sort_keys=True) == json.dumps(
        [x.to_dict() for x in out_b],
        sort_keys=True,
    )


def test_stable_to_transitional_detection() -> None:
    stable = detect_phase_state(_stability(), _perturbation())
    transitional = detect_phase_state(
        {"alpha": 0.45, "beta": 0.4},
        {"p1": 0.8, "p2": 0.9},
        prior_phase_state=stable,
    )
    assert stable.phase_label == "stable"
    assert transitional.phase_label in {"transitional", "bifurcating"}


def test_bifurcation_saturation_case() -> None:
    score = compute_bifurcation_warning_score(
        current_stability=0.0,
        prior_stability=1.0,
        transition_pressure=1.0,
        attractor_divergence=1.0,
    )
    assert score == 1.0
