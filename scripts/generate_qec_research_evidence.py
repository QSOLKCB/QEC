# SPDX-License-Identifier: MPL-2.0
"""Generate/check synthetic research evidence; no paper dataset is reproduced."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np

from qec.decoder.bp_decoder_reference import bp_decode
from qec.experiments.css_equivalence import build_css_equivalence_witness
from qec.experiments.syndrome_delivery import (
    DeliveryPlan, SyndromeTrace, run_syndrome_delivery,
)
from qec.sonify.canonical import canonical_bytes, canonical_sha256

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/research/css-delivery-v1.json"
BOUND_SOURCES = (
    "scripts/generate_qec_research_evidence.py",
    "src/qec/experiments/css_equivalence.py",
    "src/qec/experiments/syndrome_delivery.py",
    "src/qec/decoder/bp_decoder_reference.py",
    "src/qec_qldpc_codes.py",
    "src/qec/sonify/canonical.py",
)


def reference_repetition_decode(syndrome):
    """Call unchanged BP with fixed parameters on a three-bit repetition check."""
    check = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.uint8)
    correction, _ = bp_decode(
        check, np.array([2.0, 2.0, 2.0]), max_iters=20,
        syndrome_vec=np.array(syndrome, dtype=np.uint8), postprocess="osd0",
    )
    return tuple(int(bit) for bit in correction)


def build_evidence():
    """Return canonical source-bound synthetic cases, independent of wall clock."""
    # Steane CSS presentation; all single-site X/Z errors plus the zero error.
    checks = [[1, 1, 1, 1, 0, 0, 0], [1, 1, 0, 0, 1, 1, 0],
              [1, 0, 1, 0, 1, 0, 1]]
    zero = [0] * 7
    errors = [(zero, zero)]
    for site in range(7):
        error = [int(index == site) for index in range(7)]
        errors.extend(((error, zero), (zero, error)))
    css = build_css_equivalence_witness(
        checks, checks,
        x_row_transform=[[1, 1, 0], [0, 1, 1], [0, 0, 1]],
        z_row_transform=[[0, 1, 0], [1, 0, 0], [0, 0, 1]],
        qubit_permutation=[6, 2, 4, 0, 5, 1, 3], errors=errors,
    )
    trace = SyndromeTrace(2, ((0, 0), (1, 0), (1, 1), (0, 1)))
    cases = []
    for label, plan in (
        ("zero_delay", DeliveryPlan((1, 1), (0, 0))),
        ("out_of_order_delayed", DeliveryPlan(
            (1, 1), (2, 1), ((3, 0), (-2, -1), (0, 1), (-1, 0)),
            frame_period_ticks=1, deadline_ticks=3, decode_ticks=1)),
    ):
        cases.append({"case": label, "receipt": run_syndrome_delivery(
            trace, plan, reference_repetition_decode, correction_width=3,
            decoder_label="unchanged_reference_bp_repetition_3_osd0")})
    body = {
        "schema": "qec.css-delivery-research-evidence.v1",
        "data_origin": "locally_constructed_synthetic_cases_not_paper_data",
        "research_sources": [
            {"url": "https://arxiv.org/abs/2610.18218v1",
             "use": "motivation_for_supplied_binary_CSS_presentation_witness"},
            {"url": "https://arxiv.org/abs/2610.22112v1",
             "use": "motivation_for_explicit_comparison_inputs_only_not_Patchwork_reproduction"},
        ],
        "source_binding_scope": "listed_source_bytes_not_complete_dependency_or_runtime_attestation",
        "source_sha256": {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
                          for path in BOUND_SOURCES},
        "native_adapter_parameters": {
            "check_matrix": [[1, 1, 0], [0, 1, 1]], "llr": ["2.0", "2.0", "2.0"],
            "max_iters": 20, "postprocess": "osd0",
            "comparison": "correction_bits_only_not_iteration_count_or_logical_failure_rate",
        },
        "css_witness": css, "delivery_cases": cases,
        "claims": {"synthetic_only": True, "hardware_demonstrated": False,
                   "patchwork_implemented": False, "decoder_promoted": False,
                   "paper_benchmark_reproduced": False},
    }
    return {**body, "sha256": canonical_sha256(body)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="check existing fixture without writing")
    args = parser.parse_args(argv)
    expected = canonical_bytes(build_evidence()) + b"\n"
    if args.check:
        if not FIXTURE.exists() or FIXTURE.read_bytes() != expected:
            parser.exit(1, "research fixture differs; regenerate and review the source changes\n")
        print("research fixture matches source-bound replay")
    else:
        FIXTURE.parent.mkdir(parents=True, exist_ok=True)
        FIXTURE.write_bytes(expected)
        print(FIXTURE.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
