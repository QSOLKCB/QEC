from __future__ import annotations

from qec.discovery.discovery_engine import run_structure_discovery


def _spec() -> dict[str, int]:
    return {
        "num_variables": 6,
        "num_checks": 3,
        "variable_degree": 2,
        "check_degree": 4,
    }


def test_curriculum_and_clustering_flags_are_opt_in_deterministic() -> None:
    kwargs = dict(num_generations=1, population_size=4, base_seed=9)
    a = run_structure_discovery(
        _spec(),
        enable_curriculum_learning=True,
        enable_motif_learning=True,
        enable_motif_clustering=True,
        enable_operator_stability_guard=True,
        **kwargs,
    )
    b = run_structure_discovery(
        _spec(),
        enable_curriculum_learning=True,
        enable_motif_learning=True,
        enable_motif_clustering=True,
        enable_operator_stability_guard=True,
        **kwargs,
    )
    assert a["generation_summaries"] == b["generation_summaries"]
    assert a["operator_stats"] == b["operator_stats"]
