"""Deterministic pipeline orchestration for benchmark stress results.

Combines the four aggregation stages into a single deterministic unit:
    table → comparisons → pareto → scores

Version: v71.0.1
"""

from src.qec.modules.aggregation.table import build_experiment_table
from src.qec.modules.comparisons.pairwise import build_pairwise_comparison
from src.qec.modules.pareto.frontier import build_pareto_frontier
from src.qec.modules.scoring.scores import build_scores


def build_full_pipeline(
    suites: list,
    mode: str,
) -> dict:
    """Build the complete aggregation pipeline from computed suite(s).

    Assembles the result dict and runs the four-stage pipeline:
        1. build_experiment_table
        2. build_pairwise_comparison
        3. build_pareto_frontier
        4. build_scores

    Parameters
    ----------
    suites : list[dict]
        List of suite dicts from ``_run_single_genome_suite``.
        For single mode: exactly one suite.
        For sweep mode: one or more suites.
    mode : str
        Either ``"single"`` or ``"sweep"``.

    Returns
    -------
    dict
        Complete result dict with mode, suite data, table, comparisons,
        pareto, and scores.  Structure is identical to the output of
        ``run_benchmark_stress``.
    """
    if mode == "single":
        result = suites[0]
        result["mode"] = "single"
    else:
        result = {
            "mode": "sweep",
            "results": suites,
        }

    result["table"] = build_experiment_table(result)
    result["comparisons"] = build_pairwise_comparison(result)
    result["pareto"] = build_pareto_frontier(result)
    result["scores"] = build_scores(result)
    return result
