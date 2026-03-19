"""Experiment table builder for benchmark stress results.

Version: v71.0.0
"""

from src.qec.modules.common import _EXCLUDED_KEYS


def build_experiment_table(result: dict) -> list:
    """Convert benchmark result (single or sweep) into a flat list of row dicts.

    Each row corresponds to one (genome, scenario) pair with flattened metrics.
    Input is not mutated.  Order follows genome order then scenario order.

    Parameters
    ----------
    result : dict
        Output of ``run_benchmark_stress`` (mode="single" or mode="sweep").

    Returns
    -------
    list[dict]
        Flat rows with genome_id, scenario, version, base_seed_label,
        n_vars, n_iters_base, and all flattened metric values.

    Raises
    ------
    ValueError
        If ``mode`` is not "single" or "sweep", if a suite is missing
        ``scenarios``, or if a metric key collides with a reserved row key.
    """
    mode = result.get("mode")
    if mode not in ("single", "sweep"):
        raise ValueError(f"Invalid result mode: {mode!r}")

    if mode == "sweep":
        suites = result["results"]
    else:
        suites = [result]

    rows: list = []
    for suite in suites:
        if "scenarios" not in suite or not isinstance(suite["scenarios"], list):
            raise ValueError("Malformed suite: missing or invalid 'scenarios'")

        version = suite.get("version", "")
        base_seed_label = suite.get("base_seed_label", "")
        n_vars = suite.get("n_vars")
        n_iters_base = suite.get("n_iters_base")

        for scenario in suite["scenarios"]:
            row: dict = {
                "genome_id": scenario["genome_id"],
                "scenario": scenario["scenario"],
                "version": version,
                "base_seed_label": base_seed_label,
                "n_vars": n_vars,
                "n_iters_base": n_iters_base,
            }
            # Flatten metrics dict with collision guard
            metrics = scenario.get("metrics", {})
            if metrics:
                overlap = _EXCLUDED_KEYS & metrics.keys()
                if overlap:
                    raise ValueError(
                        f"Metric key collision with reserved keys: {sorted(overlap)}"
                    )
                for k, v in metrics.items():
                    row[k] = v
            rows.append(row)

    return rows
