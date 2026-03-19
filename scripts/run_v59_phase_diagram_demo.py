#!/usr/bin/env python
"""
v5.9 Demo — Decoder Phase Diagram Generator.

Runs a small deterministic 2D parameter sweep on a minimal code instance
and generates an empirical decoder phase diagram.  Prints a dominant-phase
table, summary counts, and writes a JSON artifact.

This is a diagnostic helper only — not part of the core benchmark harness.

Usage:
    python scripts/run_v59_phase_diagram_demo.py

All seeds are fixed.  Output is fully deterministic and reproducible.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any

# Ensure imports resolve from repository root.
_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import numpy as np

from qec_qldpc_codes import bp_decode, syndrome, channel_llr, create_code
from qec.diagnostics.bp_phase_space import (
    compute_bp_phase_space,
    compute_metastability_score,
)
from qec.diagnostics.ternary_decoder_topology import (
    compute_ternary_decoder_topology,
)
from qec.diagnostics.phase_diagram import (
    build_decoder_phase_diagram,
    make_phase_grid,
)
from qec.diagnostics.phase_boundary_analysis import (
    analyze_phase_boundaries,
)
from qec.diagnostics.non_backtracking_spectrum import (
    compute_non_backtracking_spectrum,
)
from qec.diagnostics.bethe_hessian import (
    compute_bethe_hessian,
)
from qec.diagnostics.bp_stability_proxy import (
    estimate_bp_stability,
)
from qec.diagnostics.bp_jacobian_estimator import (
    estimate_bp_jacobian_spectral_radius,
)
from qec.diagnostics.nb_localization import (
    compute_nb_localization_metrics,
)
from qec.diagnostics.nb_trapping_candidates import (
    compute_nb_trapping_candidates,
)
from qec.diagnostics.phase_heatmap import (
    print_phase_heatmap,
)

# ── Configuration ─────────────────────────────────────────────────────

CODE_NAME = "rate_0.50"
LIFTING_SIZE = 8
CODE_SEED = 42
P_VALUES = [0.01, 0.03, 0.05]
DISTANCES = [3, 5, 7]
TRIALS_PER_POINT = 5
MAX_ITERS = 30
BP_MODE = "min_sum"
SCHEDULE = "flooding"
RNG_BASE_SEED = 12345
OUTPUT_FILE = "phase_diagram_v59_demo.json"


def run_phase_diagram_demo() -> dict[str, Any]:
    """Run the phase diagram demo experiment.

    Returns the full phase diagram result with boundary analysis.
    """
    grid = make_phase_grid(
        x_name="physical_error_rate",
        x_values=P_VALUES,
        y_name="code_distance",
        y_values=DISTANCES,
    )

    # Pre-create codes for each distance.
    codes: dict[int, np.ndarray] = {}
    for d in DISTANCES:
        code = create_code(name=CODE_NAME, lifting_size=d, seed=CODE_SEED)
        codes[d] = code.H_X

    def trial_runner(p: float, distance: float) -> list[dict[str, Any]]:
        """Run decoding trials at a single (p, distance) grid point."""
        d = int(distance)
        H = codes[d]
        n = H.shape[1]

        # v6.0: Compute spectral diagnostics per grid point (code-level).
        nb_result = compute_non_backtracking_spectrum(H)
        bethe_result = compute_bethe_hessian(H)
        stability_result = estimate_bp_stability(nb_result, bethe_result)

        # v6.1: Compute localization diagnostics per grid point (code-level).
        localization_result = compute_nb_localization_metrics(H)

        # v6.2: Compute trapping-set candidate diagnostics per grid point.
        trapping_result = compute_nb_trapping_candidates(H, localization_result)

        # Deterministic RNG derived from base seed + grid point.
        seed = RNG_BASE_SEED + int(p * 100000) + d * 1000
        rng = np.random.default_rng(seed)

        trial_results: list[dict[str, Any]] = []
        for _ in range(TRIALS_PER_POINT):
            # Generate error instance.
            e = (rng.random(n) < p).astype(np.uint8)
            s = syndrome(H, e)
            llr = channel_llr(e, p)

            # Decode with LLR history.
            result = bp_decode(
                H, llr,
                max_iters=MAX_ITERS,
                mode=BP_MODE,
                schedule=SCHEDULE,
                syndrome_vec=s,
                llr_history=MAX_ITERS,
            )
            correction, iters, llr_hist = result[0], result[1], result[2]

            # Phase-space diagnostic.
            trajectory_states = [
                llr_hist[i].copy() for i in range(llr_hist.shape[0])
            ]
            ps_result = compute_bp_phase_space(trajectory_states)

            # Syndrome residual for ternary classification.
            s_correction = syndrome(H, correction)
            final_sw = int(np.sum(s_correction != s))
            syndrome_residuals = [final_sw] * len(trajectory_states)

            # Ternary topology classification.
            tt_result = compute_ternary_decoder_topology(
                phase_space_result=ps_result,
                syndrome_residuals=syndrome_residuals,
            )

            # Metastability score.
            ms = compute_metastability_score(ps_result["residual_norms"])
            tt_result["metastability_score"] = ms

            # v6.0: Attach spectral diagnostics to trial result.
            tt_result["spectral_radius"] = stability_result["spectral_radius"]
            tt_result["bethe_min_eigenvalue"] = stability_result["bethe_min_eigenvalue"]
            tt_result["bp_stability_score"] = stability_result["bp_stability_score"]

            # v6.0: Jacobian spectral radius from LLR history.
            jacobian_result = estimate_bp_jacobian_spectral_radius(llr_hist)
            tt_result["jacobian_spectral_radius_est"] = jacobian_result["jacobian_spectral_radius_est"]

            # v6.1: Attach localization diagnostics to trial result.
            tt_result["nb_max_ipr"] = localization_result["max_ipr"]
            tt_result["nb_num_localized_modes"] = len(localization_result["localized_modes"])
            tt_result["nb_top_localization_score"] = localization_result["top_localization_score"]

            # v6.2: Attach trapping-set candidate diagnostics to trial result.
            tt_result["nb_num_candidate_nodes"] = trapping_result["num_candidate_nodes"]
            tt_result["nb_max_node_participation"] = trapping_result["max_node_participation"]
            tt_result["nb_num_candidate_clusters"] = trapping_result["num_candidate_clusters"]

            trial_results.append(tt_result)

        return trial_results

    phase_diagram = build_decoder_phase_diagram(grid, trial_runner)
    boundary_analysis = analyze_phase_boundaries(phase_diagram)

    return {
        "phase_diagram": phase_diagram,
        "boundary_analysis": boundary_analysis,
    }


def print_summary(result: dict[str, Any]) -> None:
    """Print a compact phase-diagram summary."""
    pd = result["phase_diagram"]
    ba = result["boundary_analysis"]
    axes = pd["grid_axes"]
    cells = pd["cells"]

    x_name = axes["x_name"]
    y_name = axes["y_name"]
    x_values = axes["x_values"]
    y_values = axes["y_values"]

    print("=" * 60)
    print("v6.0.0 — Decoder Phase Diagram Demo")
    print("=" * 60)
    print(f"Code: {CODE_NAME}, lifting_size per distance")
    print(f"BP: mode={BP_MODE}, schedule={SCHEDULE}, max_iters={MAX_ITERS}")
    print(f"Trials per grid point: {TRIALS_PER_POINT}")
    print(f"Grid: {x_name} x {y_name}")
    print(f"  {x_name}: {x_values}")
    print(f"  {y_name}: {y_values}")
    print("=" * 60)
    print()

    # Build lookup.
    cell_lookup: dict[tuple, dict] = {}
    for c in cells:
        cell_lookup[(c["x"], c["y"])] = c

    # Phase label map.
    label_map = {1: "+1", 0: " 0", -1: "-1"}

    # Dominant phase table.
    print(f"Dominant Phase Map ({x_name} → columns, {y_name} → rows):")
    header = f"{'':>12s}"
    for x in x_values:
        header += f" {x:>10}"
    print(header)

    for y in y_values:
        row = f"{y:>12}"
        for x in x_values:
            cell = cell_lookup.get((x, y))
            if cell is not None:
                row += f" {label_map.get(cell['dominant_phase'], ' ?'):>10s}"
            else:
                row += f" {'--':>10s}"
        print(row)

    print()

    # Success fraction table.
    print(f"Success Fraction ({x_name} → columns, {y_name} → rows):")
    header = f"{'':>12s}"
    for x in x_values:
        header += f" {x:>10}"
    print(header)

    for y in y_values:
        row = f"{y:>12}"
        for x in x_values:
            cell = cell_lookup.get((x, y))
            if cell is not None:
                row += f" {cell['success_fraction']:>10.2f}"
            else:
                row += f" {'--':>10s}"
        print(row)

    print()

    # Phase entropy table.
    print(f"Phase Entropy ({x_name} → columns, {y_name} → rows):")
    header = f"{'':>12s}"
    for x in x_values:
        header += f" {x:>10}"
    print(header)

    for y in y_values:
        row = f"{y:>12}"
        for x in x_values:
            cell = cell_lookup.get((x, y))
            if cell is not None:
                row += f" {cell['phase_entropy']:>10.4f}"
            else:
                row += f" {'--':>10s}"
        print(row)

    print()

    # Summary counts.
    success_cells = sum(1 for c in cells if c["dominant_phase"] == 1)
    boundary_cells = sum(1 for c in cells if c["dominant_phase"] == 0)
    failure_cells = sum(1 for c in cells if c["dominant_phase"] == -1)

    print(f"Phase counts: {success_cells} success (+1), "
          f"{boundary_cells} boundary (0), {failure_cells} failure (-1) "
          f"out of {len(cells)} cells")

    bs = ba["boundary_summary"]
    print(f"Boundary analysis: {bs['num_boundary_cells']} boundary, "
          f"{bs['num_mixed_cells']} mixed, {bs['num_critical_cells']} critical")

    # v6.0: ASCII phase heatmap.
    print_phase_heatmap(pd)

    # JSON check.
    json_str = json.dumps(result, indent=2)
    print()
    print("-" * 60)
    print("JSON serialization: OK")
    print(f"JSON size: {len(json_str)} bytes")
    print("-" * 60)


def main() -> int:
    """Entry point."""
    result = run_phase_diagram_demo()
    print_summary(result)

    # Write JSON output.
    output_path = os.path.join(_repo_root, OUTPUT_FILE)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nPhase diagram written to: {output_path}")

    # Determinism verification: run again and compare.
    result2 = run_phase_diagram_demo()
    j1 = json.dumps(result, sort_keys=True)
    j2 = json.dumps(result2, sort_keys=True)
    if j1 == j2:
        print("Determinism check: PASSED (identical outputs)")
    else:
        print("Determinism check: FAILED (outputs differ)")
        return 1

    # Clean up output file.
    os.remove(output_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
