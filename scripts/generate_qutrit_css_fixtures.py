# SPDX-License-Identifier: MPL-2.0
"""Export bounded GF(3) conformance cases from a pinned css-decoupling checkout.

The donor is optional and imported only by this explicit regeneration command.
Normal QEC tests replay the saved data without it. See the companion research
note for conventions, attribution and the limits of this finite comparison.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/research/qutrit-css-decoupling-v1.json"
DONOR_COMMIT = "a48048c1097f4d1b37a55eca9a1796a823f0c124"
DONOR_URL = "https://github.com/hmx21/css-decoupling"


def canonical_bytes(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def check_checkout(path):
    """Refuse a different commit or changed/untracked donor source."""
    def git(*args):
        return subprocess.check_output(
            ["git", "-C", str(path), *args], text=True
        ).strip()

    if git("rev-parse", "HEAD") != DONOR_COMMIT:
        raise ValueError(f"donor must be checked out at {DONOR_COMMIT}")
    if git("status", "--porcelain", "--untracked-files=all"):
        raise ValueError("donor checkout must be clean")


def matrix_data(matrix):
    return {
        "rows": matrix.nrows,
        "cols": matrix.ncols,
        "entries": [
            [i, j, [[list(exponent), int(coefficient)]
                    for exponent, coefficient in sorted(poly.terms.items())]]
            for (i, j), poly in sorted(matrix.entries.items())
        ],
    }


def build_fixture(donor):
    check_checkout(donor)
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(donor / "src"))
    import css_decoupling
    import mpmath
    import sympy
    from css_decoupling import Matrix, Ring, decouple, verify_result
    from css_decoupling.circuits.gates import Circuit, SUM, SWAP
    from css_decoupling.topology.periodic import periodic_code

    if Path(css_decoupling.__file__).resolve() != donor / "src/css_decoupling/__init__.py":
        raise ValueError("imported donor is not the pinned checkout")
    if (sympy.__version__, mpmath.__version__) != ("1.14.0", "1.3.0"):
        raise ValueError("fixture regeneration requires sympy==1.14.0 and mpmath==1.3.0")

    ring = Ring(3, 2)
    x, y = ring.variables
    # The donor README's prime-three toric input, not experimental data.
    hx = Matrix.from_rows(ring, [[x - 1, y - 1]])
    hz = Matrix.from_rows(ring, [[1 - y**-1, x**-1 - 1]])
    result = decouple(hx, hz, dimension=2, algorithm="general", output="circuit",
                      options={"linear_backend": "sympy", "contraction_strategy": "normalized"})
    report = verify_result(result, expected_HX=hx, expected_HZ=hz)
    if result.status != "circuit_ready" or not report.valid:
        raise ValueError(f"donor toric verification failed: {report.errors}")

    def gate_data(gate):
        if isinstance(gate, SUM):
            return {"kind": "SUM", "control": gate.control, "target": gate.target,
                    "weight": gate.weight % 3, "displacement": list(gate.displacement)}
        return {"kind": "SWAP", "left": gate.left, "right": gate.right}

    def circuit_data(name, circuit, gates, inverse):
        return {"name": name, "qudits": circuit.qudits, "gates": gates,
                "actions": {action: matrix_data(circuit.matrix(action))
                            for action in ("X", "Z")},
                "inverse_actions": {action: matrix_data(inverse.matrix(action))
                                    for action in ("X", "Z")}}

    cases = []
    for name, gates in (
        ("identity", ()),
        ("sum_weight_one", (SUM(0, 1, 1, (1, -1)),)),
        ("sum_weight_two", (SUM(0, 1, 2, (-1, 1)),)),
        ("swap", (SWAP(0, 1),)),
        ("mixed", (SUM(0, 1, 2, (1, -1)), SWAP(0, 1),
                   SUM(1, 0, 1, (-1, 0)), SUM(0, 1, 1, (0, 1)))),
    ):
        circuit = Circuit(ring, 2, gates)
        cases.append(circuit_data(name, circuit, [gate_data(g) for g in gates],
                                  circuit.inverse()))
    cases.append(circuit_data("toric_general", result.circuit,
                              list(result.circuit.gates), result.circuit.action.inverse()))

    periodic = []
    for lx, ly in ((2, 2), (2, 3), (3, 3)):
        count = periodic_code(hx, hz, [[lx, 0], [0, ly]])
        periodic.append({"periods": [lx, ly], "physical_qudits": count.physical_qudits,
                         "rank_x": count.rank_x, "rank_z": count.rank_z,
                         "logical_qudits": count.logical_qudits})
    return {
        "schema": "qec.qutrit-css-decoupling-conformance.v1",
        "prime": 3, "dimension": 2,
        "provenance": {
            "paper": "https://arxiv.org/abs/2610.40592v1",
            "donor_repository": DONOR_URL, "donor_commit": DONOR_COMMIT,
            "donor_license": "MIT", "sympy": sympy.__version__, "mpmath": mpmath.__version__,
            "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "origin": "QEC-selected synthetic gates and donor README toric input",
        },
        "conventions": {
            "vectors": "columns; X then Z; global Pauli phase omitted",
            "gate_order": "first listed first",
            "sum": "control at r+displacement; target at r; target += weight*control",
            "laurent_entries": "[row,column,[[exponent,coefficient],...]]",
            "finite_order": "cell major; lexicographic cells; species within cell",
        },
        "circuits": cases,
        "toric": {"HX": matrix_data(hx), "HZ": matrix_data(hz), "periodic": periodic,
                  "donor_verification_valid": report.valid, "status": result.status,
                  "metadata": result.metadata,
                  "certificate_sha256": hashlib.sha256(canonical_bytes(result.certificate())).hexdigest()},
        "claims": {"finite_software_conformance_only": True,
                   "python_verified_by_lean": False, "polynomial_runtime_demonstrated": False,
                   "hardware_demonstrated": False, "decoder_promoted": False},
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--donor-checkout", required=True, type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true", help="compare all bytes with the saved fixture")
    mode.add_argument("--output", type=Path, help="write to a new path; never overwrite evidence")
    args = parser.parse_args(argv)
    expected = canonical_bytes(build_fixture(args.donor_checkout.resolve()))
    if args.check:
        if FIXTURE.read_bytes() != expected:
            parser.exit(1, "qutrit fixture differs from pinned donor replay\n")
        print("qutrit fixture matches pinned donor replay")
    else:
        with args.output.open("xb") as stream:
            stream.write(expected)
        print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
