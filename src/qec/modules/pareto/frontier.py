"""Pareto frontier identification for benchmark stress results.

Version: v71.0.0
"""

from src.qec.modules.common import _EXCLUDED_KEYS, _is_finite_numeric


def build_pareto_frontier(result: dict) -> list:
    """Identify Pareto-optimal genomes per scenario based on numeric metrics.

    For each scenario, a genome is Pareto-optimal (non-dominated) if no other
    genome is at least as good on all numeric metrics and strictly better on
    at least one.

    Parameters
    ----------
    result : dict
        Output of ``run_benchmark_stress``.  Must contain ``"table"`` key.

    Returns
    -------
    list[dict]
        Each entry: ``{"scenario": str, "pareto_genomes": list[str]}``.
        Ordered by scenario appearance in the table.

    Raises
    ------
    ValueError
        If ``result`` has no ``"table"`` key, if a comparison references an
        unknown scenario, or if a scenario has no valid numeric metrics.
    """
    if "table" not in result:
        raise ValueError("Result dict missing 'table' key")

    table = result["table"]

    # Build scenario → rows mapping (preserving insertion order)
    scenario_genomes: dict = {}
    for row in table:
        sc = row["scenario"]
        if sc not in scenario_genomes:
            scenario_genomes[sc] = []
        scenario_genomes[sc].append(row)

    # Validate that comparisons only reference known scenarios
    for comp in result.get("comparisons", []):
        if comp["scenario"] not in scenario_genomes:
            raise ValueError("unknown scenario")

    frontier: list = []
    for scenario, rows in scenario_genomes.items():
        # Identify numeric metric keys present and finite in all rows
        numeric_keys = sorted(
            k for k in rows[0]
            if k not in _EXCLUDED_KEYS
            and all(_is_finite_numeric(r.get(k)) for r in rows)
        )

        if not numeric_keys:
            raise ValueError("no valid numeric deltas")

        # Find non-dominated genomes
        non_dominated: list = []
        for i, row_i in enumerate(rows):
            dominated = False
            for j, row_j in enumerate(rows):
                if i == j:
                    continue
                # row_j dominates row_i iff row_j >= row_i on all and > on at least one
                all_geq = all(row_j[k] >= row_i[k] for k in numeric_keys)
                any_gt = any(row_j[k] > row_i[k] for k in numeric_keys)
                if all_geq and any_gt:
                    dominated = True
                    break
            if not dominated:
                non_dominated.append(row_i["genome_id"])

        frontier.append({
            "scenario": scenario,
            "pareto_genomes": non_dominated,
        })

    return frontier
