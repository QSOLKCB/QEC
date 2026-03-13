from __future__ import annotations

from experiments.nb_perturbation_scoring_experiment import run


def test_experiment_runs_and_is_deterministic() -> None:
    r1 = run()
    r2 = run()
    assert r1 == r2
    for key in [
        "top1_agreement",
        "topk_overlap",
        "spearman_proxy",
        "avg_candidate_count",
        "avg_exact_rechecks",
    ]:
        assert isinstance(r1[key], float)
