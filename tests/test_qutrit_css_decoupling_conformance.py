# SPDX-License-Identifier: MPL-2.0
"""Independent finite-torus replay of pinned donor data; no donor imports.

This tests saved SUM/SWAP actions and QEC's existing GF(3) consumers. It is not
a verifier of the donor's general decoupling theorem or its full certificate.
"""
from copy import deepcopy
import hashlib
from itertools import product
import json
from pathlib import Path

import pytest

from qec.decoder.qutrit import ExactDecoder, Pauli, QutritStabilizerCode, paulis_of_weight
from qec.decoder.qutrit.gf3 import rank
from qec.decoder.qutrit.stabilizer import symplectic

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/research/qutrit-css-decoupling-v1.json"
DATA = json.loads(FIXTURE.read_text())
CASES = DATA["circuits"]


def finite_matrix(encoded, periods):
    """Expand monomials by translating ket coordinates; cell-major ordering."""
    cells = list(product(*(range(length) for length in periods)))
    positions = {cell: i for i, cell in enumerate(cells)}
    rows, cols = encoded["rows"], encoded["cols"]
    matrix = [[0] * (cols * len(cells)) for _ in range(rows * len(cells))]
    for row, column, terms in encoded["entries"]:
        for exponent, coefficient in terms:
            for j, cell in enumerate(cells):
                shifted = tuple((a + b) % length
                                for a, b, length in zip(cell, exponent, periods))
                i = positions[shifted]
                matrix[i * rows + row][j * cols + column] = (
                    matrix[i * rows + row][j * cols + column] + coefficient
                ) % 3
    return matrix


def transpose(matrix):
    return [list(column) for column in zip(*matrix)]


def matvec(matrix, vector):
    return tuple(sum(a * b for a, b in zip(row, vector)) % 3 for row in matrix)


def matmul(left, right):
    return transpose([matvec(left, column) for column in transpose(right)])


def identity(size):
    return [[int(i == j) for j in range(size)] for i in range(size)]


def inverse_gates(gates):
    result = deepcopy(gates[::-1])
    for gate in result:
        if gate["kind"] == "SUM":
            gate["weight"] = -gate["weight"] % 3
    return result


def replay(error, gates, qudits, periods):
    """Direct local Pauli conjugation, independently of Laurent products."""
    cells = list(product(*(range(length) for length in periods)))
    positions = {cell: i for i, cell in enumerate(cells)}
    x, z = list(error.x), list(error.z)
    for gate in gates:
        old_x, old_z = x[:], z[:]
        for i, cell in enumerate(cells):
            if gate["kind"] == "SWAP":
                a, b = i * qudits + gate["left"], i * qudits + gate["right"]
                x[a], x[b] = old_x[b], old_x[a]
                z[a], z[b] = old_z[b], old_z[a]
            elif gate["kind"] == "SUM":
                shifted = tuple((a + b) % length for a, b, length in
                                zip(cell, gate["displacement"], periods))
                control = positions[shifted] * qudits + gate["control"]
                target = i * qudits + gate["target"]
                x[target] = (old_x[target] + gate["weight"] * old_x[control]) % 3
                z[control] = (old_z[control] - gate["weight"] * old_z[target]) % 3
            else:
                raise ValueError("unsupported fixture gate")
    return Pauli(tuple(x), tuple(z))


def replay_matrix(case, periods, action, *, inverse=False):
    size = case["qudits"] * periods[0] * periods[1]
    zero = (0,) * size
    columns = []
    gates = inverse_gates(case["gates"]) if inverse else case["gates"]
    for unit in identity(size):
        error = Pauli(unit, zero) if action == "X" else Pauli(zero, unit)
        output = replay(error, gates, case["qudits"], periods)
        columns.append(output.x if action == "X" else output.z)
    return transpose(columns)


def independent_rows(rows):
    basis = []
    for row in rows:
        if rank(basis + [row]) > len(basis):
            basis.append(row)
    return basis


def toric_code(periods):
    hx = finite_matrix(DATA["toric"]["HX"], periods)
    hz = finite_matrix(DATA["toric"]["HZ"], periods)
    zero = [0] * len(hx[0])
    generators = [row + zero for row in independent_rows(hx)]
    generators += [zero + row for row in independent_rows(hz)]
    return QutritStabilizerCode(tuple(tuple(row) for row in generators)), hx, hz


def test_fixture_scope_and_generator_binding():
    assert DATA["schema"] == "qec.qutrit-css-decoupling-conformance.v1"
    assert (DATA["prime"], DATA["dimension"]) == (3, 2)
    assert DATA["provenance"]["donor_commit"] == "a48048c1097f4d1b37a55eca9a1796a823f0c124"
    assert DATA["provenance"]["generator_sha256"] == hashlib.sha256(
        (ROOT / "scripts/generate_qutrit_css_fixtures.py").read_bytes()
    ).hexdigest()
    assert [case["name"] for case in CASES] == [
        "identity", "sum_weight_one", "sum_weight_two", "swap", "mixed", "toric_general"
    ]
    assert DATA["claims"] == {
        "finite_software_conformance_only": True, "python_verified_by_lean": False,
        "polynomial_runtime_demonstrated": False, "hardware_demonstrated": False,
        "decoder_promoted": False,
    }
    assert len(CASES[-1]["gates"]) == 17
    assert DATA["toric"]["metadata"]["qca_input_ancillas"] == 4
    assert DATA["toric"]["metadata"]["target"]["toric_multiplicities"] == [[1, 1]]
    assert FIXTURE.read_bytes() == (
        json.dumps(DATA, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()


@pytest.mark.parametrize("periods", [(1, 1), (3, 2), (2, 3)])
@pytest.mark.parametrize("case", CASES, ids=lambda case: case["name"])
def test_exact_forward_inverse_and_symplectic_actions(case, periods):
    actions = {}
    for action in ("X", "Z"):
        forward = replay_matrix(case, periods, action)
        inverse = replay_matrix(case, periods, action, inverse=True)
        assert forward == finite_matrix(case["actions"][action], periods)
        assert inverse == finite_matrix(case["inverse_actions"][action], periods)
        assert matmul(inverse, forward) == identity(len(forward))
        assert matmul(forward, inverse) == identity(len(forward))
        actions[action] = forward
    # For CSS actions, X^T Z = I is the exact commutation-preservation test.
    assert matmul(transpose(actions["X"]), actions["Z"]) == identity(len(forward))
    zero = [0] * len(forward)
    for i, x in enumerate(transpose(actions["X"])):
        for j, z in enumerate(transpose(actions["Z"])):
            assert symplectic(x + zero, zero + z) == int(i == j)


@pytest.mark.parametrize("mutation", ["weight", "displacement", "order", "missing_swap"])
def test_negative_controls_detect_wrong_gate_conventions(mutation):
    case = deepcopy(next(case for case in CASES if case["name"] == "mixed"))
    if mutation == "weight":
        case["gates"][0]["weight"] = 1
    elif mutation == "displacement":
        case["gates"][0]["displacement"][0] *= -1
    elif mutation == "order":
        case["gates"].reverse()
    else:
        case["gates"].pop(1)
    # Even another invertible circuit must fail comparison with the saved action.
    assert replay_matrix(case, (3, 2), "X") != finite_matrix(case["actions"]["X"], (3, 2))


@pytest.mark.parametrize("count", DATA["toric"]["periodic"], ids=lambda c: str(c["periods"]))
def test_toric_rank_and_syndrome_covariance_in_qec(count):
    periods = count["periods"]
    code, hx, hz = toric_code(periods)
    assert rank(hx) == count["rank_x"] == periods[0] * periods[1] - 1
    assert rank(hz) == count["rank_z"] == periods[0] * periods[1] - 1
    assert code.n == count["physical_qudits"] == 2 * periods[0] * periods[1]
    assert code.k == count["logical_qudits"] == 2
    circuit = next(case for case in CASES if case["name"] == "mixed")
    moved = [replay(Pauli(row[:code.n], row[code.n:]), circuit["gates"], 2, periods)
             for row in code.stabilizers]
    transformed = QutritStabilizerCode(tuple(p.symplectic_vector for p in moved))
    for error in paulis_of_weight(code.n, 1):
        observed = replay(error, circuit["gates"], 2, periods)
        assert code.syndrome(error) == transformed.syndrome(observed)
    assert (transformed.n, transformed.k) == (code.n, code.k)


def test_every_single_site_qutrit_error_on_three_by_three_torus_is_corrected():
    code, _, _ = toric_code((3, 3))
    decoder = ExactDecoder(code, max_weight=1)
    errors = tuple(paulis_of_weight(code.n, 1))
    assert len(errors) == 144
    assert all(decoder.correct(error).success for error in errors)
