"""
Sonification Regression Detection — v73.1.0

Compares two multirun outputs (from run_sonification_multirun v73.0.0)
and detects improvements, regressions, or equivalence.
"""

import copy
import json
import os


def compare_sonification_runs(
    run_a: dict,
    run_b: dict,
    output_dir: str = None,
) -> dict:
    """
    Compares two multirun outputs (v73.0).

    Parameters
    ----------
    run_a : dict
        Output from run_sonification_multirun (baseline).
    run_b : dict
        Output from run_sonification_multirun (candidate).
    output_dir : str, optional
        If provided, writes regression_summary.json to this directory.

    Returns
    -------
    dict
        Contains metric deltas, verdict shift, and classification.
    """
    a = copy.deepcopy(run_a)
    b = copy.deepcopy(run_b)

    # Step 1 — Compute deltas
    mean_delta = b["global_mean_score"] - a["global_mean_score"]
    variance_delta = b["global_variance"] - a["global_variance"]
    stability_delta = b["stability_score"] - a["stability_score"]

    # Step 2 — Verdict shift
    a_totals = a.get("verdict_totals", {})
    b_totals = b.get("verdict_totals", {})

    a_improve = a_totals.get("multidim_improves_structure", 0)
    b_improve = b_totals.get("multidim_improves_structure", 0)

    a_total = sum(a_totals.get(k, 0) for k in a_totals)
    b_total = sum(b_totals.get(k, 0) for k in b_totals)

    run_a_improve_ratio = a_improve / a_total if a_total > 0 else 0.0
    run_b_improve_ratio = b_improve / b_total if b_total > 0 else 0.0

    # Step 3 — Classification
    # Check equivalence first: tiny mean delta dominates.
    if abs(mean_delta) < 1e-6:
        classification = "equivalent"
    elif mean_delta > 0 and stability_delta > 0:
        classification = "improved"
    elif mean_delta < 0 and stability_delta < 0:
        classification = "regressed"
    else:
        classification = "mixed"

    result = {
        "deltas": {
            "mean": mean_delta,
            "variance": variance_delta,
            "stability": stability_delta,
        },
        "verdict_shift": {
            "run_a_improve_ratio": run_a_improve_ratio,
            "run_b_improve_ratio": run_b_improve_ratio,
        },
        "classification": classification,
    }

    # Optional output
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, "regression_summary.json")
        with open(path, "w") as f:
            json.dump(result, f, indent=2, sort_keys=True)

    return result
