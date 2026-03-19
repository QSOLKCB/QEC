from __future__ import annotations

from qec.analysis.spectral_conjecture_validation import validate_conjectures
from qec.analysis.spectral_counterexamples import extract_counterexamples
from qec.analysis.spectral_theory_memory import (
    initialize_theory_memory,
    summarize_theory_memory,
    update_theory_memory,
)
from qec.discovery.discovery_engine import run_structure_discovery


def _dataset() -> dict:
    return {
        "X": [[1.0, 2.0], [2.0, 1.0], [3.0, 0.0]],
        "y": [1.0, 2.0, 3.0],
        "rows": [{"candidate_id": "a"}, {"candidate_id": "b"}, {"candidate_id": "c"}],
        "feature_names": ["f0", "f1"],
    }


def test_validation_is_deterministic_and_sorted() -> None:
    conjectures = [
        {"conjecture_id": "c2", "target_value": 0.0, "equation_string": "y=0"},
        {"conjecture_id": "c1", "coefficients": [1.0, 0.0], "equation_string": "y=f0"},
    ]
    v1 = validate_conjectures(conjectures, _dataset(), tolerance=0.2)
    v2 = validate_conjectures(conjectures, _dataset(), tolerance=0.2)
    assert v1 == v2
    assert [r["conjecture_id"] for r in v1] == ["c1", "c2"]


def test_counterexample_ordering_is_deterministic() -> None:
    conjecture = {"conjecture_id": "c1", "target_value": 0.0}
    c1 = extract_counterexamples(conjecture, _dataset(), error_threshold=0.5)
    c2 = extract_counterexamples(conjecture, _dataset(), error_threshold=0.5)
    assert c1 == c2
    assert [x["row_index"] for x in c1] == [2, 1, 0]


def test_theory_memory_update_is_stable() -> None:
    conjectures = [
        {"conjecture_id": "c1", "equation_string": "y=f0"},
        {"conjecture_id": "c2", "equation_string": "y=0"},
    ]
    validations = [
        {"conjecture_id": "c1", "support_score": 0.9, "passes_tolerance": True},
        {"conjecture_id": "c2", "support_score": 0.2, "passes_tolerance": False},
    ]
    counterexamples = [{"conjecture_id": "c2"}]

    memory = initialize_theory_memory()
    memory = update_theory_memory(memory, conjectures, validations, counterexamples)
    memory = update_theory_memory(memory, conjectures, validations, counterexamples)
    summary = summarize_theory_memory(memory)
    assert [r["conjecture_id"] for r in summary] == ["c1", "c2"]
    assert summary[0]["status"] == "supported"
    assert summary[1]["status"] == "rejected"


def test_engine_theory_refinement_opt_in() -> None:
    spec = {"num_variables": 6, "num_checks": 3, "variable_degree": 2, "check_degree": 4}
    result = run_structure_discovery(
        spec,
        num_generations=2,
        population_size=4,
        base_seed=7,
        enable_theory_extraction=True,
        enable_theory_refinement=True,
        theory_refinement_interval=1,
    )
    assert "theory_memory" in result
    assert "theory_memory_summary" in result
    assert "conjecture_validations" in result
    assert "conjecture_counterexamples" in result
    summary = result["generation_summaries"][-1]
    assert "num_validated_conjectures" in summary
    assert "num_supported_conjectures" in summary
    assert "num_fragile_conjectures" in summary
    assert "num_rejected_conjectures" in summary
    assert "mean_theory_support_score" in summary
