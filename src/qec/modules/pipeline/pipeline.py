"""Deterministic pipeline orchestration for benchmark stress results.

Combines the four aggregation stages into a single deterministic,
non-mutating unit:
    validate → table → comparisons → pareto → scores

The pipeline never mutates its input dictionaries.  It constructs and
returns a new result dict.

Version: v71.0.3
"""

import os

from src.qec.modules.aggregation.table import build_experiment_table
from src.qec.modules.comparisons.pairwise import build_pairwise_comparison
from src.qec.modules.pareto.frontier import build_pareto_frontier
from src.qec.modules.scoring.scores import build_scores

from src.qec.modules.pipeline.validation import (
    validate_mode,
    validate_suites,
    validate_sweep_result,
)

_DEBUG_IMMUTABILITY = os.environ.get("QEC_DEBUG_IMMUTABILITY") == "1"


def build_full_pipeline(
    suites: list,
    mode: str,
) -> dict:
    """Build the complete aggregation pipeline from computed suite(s).

    Assembles a **new** result dict and runs the four-stage pipeline:
        1. build_experiment_table
        2. build_pairwise_comparison
        3. build_pareto_frontier
        4. build_scores

    This function is non-mutating: the input *suites* dicts are never
    modified.

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

    Raises
    ------
    ValueError
        - If *mode* is not ``"single"`` or ``"sweep"``.
        - If *suites* is empty.
        - If *mode* is ``"single"`` and *suites* has != 1 element.
        - If the assembled result is structurally invalid.
    """
    # --- validation ---
    validate_mode(mode)
    validate_suites(mode, suites)

    # --- dev-only immutability snapshot ---
    if _DEBUG_IMMUTABILITY:
        import copy
        _before = copy.deepcopy(suites)

    # --- assemble result (non-mutating) ---
    # Shallow copy is sufficient: pipeline stages read nested data
    # but never mutate it.  See stage docstrings for contracts.
    if mode == "single":
        result = {**suites[0], "mode": "single"}
    else:
        result = {
            "mode": "sweep",
            "results": suites,
        }

    validate_sweep_result(result)

    # --- four-stage pipeline (each stage reads prior outputs) ---
    table = build_experiment_table(result)
    result["table"] = table

    comparisons = build_pairwise_comparison(result)
    result["comparisons"] = comparisons

    pareto = build_pareto_frontier(result)
    result["pareto"] = pareto

    scores = build_scores(result)
    result["scores"] = scores

    # --- dev-only immutability check ---
    if _DEBUG_IMMUTABILITY:
        assert suites == _before, "Pipeline mutated input suites!"

    return result
