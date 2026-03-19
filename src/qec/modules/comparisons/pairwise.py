"""Pairwise genome comparison builder for benchmark stress results.

Version: v71.0.0
"""

from collections import defaultdict

from qec.modules.common import _EXCLUDED_KEYS


def build_pairwise_comparison(result: dict) -> list:
    """Compute pairwise metric deltas between genomes for each scenario.

    For each scenario, iterates all ordered pairs (i, j) where i < j
    and computes (row_j[metric] - row_i[metric]) for all numeric fields.

    Parameters
    ----------
    result : dict
        Output of ``run_benchmark_stress``.  Must contain ``"table"`` key.

    Returns
    -------
    list[dict]
        Each row: genome_a, genome_b, scenario, and ``<metric>_delta`` fields.
        Deterministic ordering: scenario order preserved, genome order preserved.

    Raises
    ------
    ValueError
        If ``result`` has no ``"table"`` key.
    """
    if "table" not in result:
        raise ValueError("Result dict missing 'table' key")

    table = result["table"]

    # Group rows by scenario, preserving insertion order
    scenario_groups: dict = defaultdict(list)
    for row in table:
        scenario_groups[row["scenario"]].append(row)

    comparisons: list = []
    for scenario, rows in scenario_groups.items():
        if len(rows) < 2:
            continue
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                row_i = rows[i]
                row_j = rows[j]
                comp: dict = {
                    "genome_a": row_i["genome_id"],
                    "genome_b": row_j["genome_id"],
                    "scenario": scenario,
                }
                # Compute deltas for all numeric fields
                for key in row_i:
                    if key in _EXCLUDED_KEYS:
                        continue
                    val_i = row_i[key]
                    val_j = row_j.get(key)
                    if isinstance(val_i, (int, float)) and isinstance(val_j, (int, float)):
                        comp[f"{key}_delta"] = val_j - val_i
                comparisons.append(comp)

    return comparisons
