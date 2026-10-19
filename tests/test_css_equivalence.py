# SPDX-License-Identifier: MPL-2.0
"""Independent finite-field and strict replay checks for CSS witnesses."""
from copy import deepcopy
from fractions import Fraction
from itertools import permutations, product
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from qec.experiments.css_equivalence import (
    MAX_CHECKS_PER_SECTOR,
    MAX_ERROR_SAMPLES,
    MAX_QUBITS,
    build_css_equivalence_witness,
    verify_css_equivalence_witness,
)
from qec.sonify.canonical import canonical_json, canonical_sha256


def _inputs():
    # Repeated Z generators deliberately exercise rank, not raw row counts.
    return {
        "hx": [[1, 1, 0, 0], [0, 0, 1, 1]],
        "hz": [[1, 1, 1, 1], [1, 1, 1, 1]],
        "x_row_transform": [[1, 1], [1, 0]],
        "z_row_transform": [[0, 1], [1, 1]],
        "qubit_permutation": [2, 0, 3, 1],
        "errors": [([1, 0, 0, 1], [1, 1, 0, 1]), ([1, 0, 0, 0], [0, 0, 1, 0])],
    }


def _dot(left, right):
    return sum(a * b for a, b in zip(left, right)) % 2


def _multiply(left, right):
    return [[_dot(row, column) for column in zip(*right)] for row in left]


def _span(rows, width):
    return {tuple(sum(c * row[j] for c, row in zip(coefficients, rows)) % 2
                  for j in range(width))
            for coefficients in product((0, 1), repeat=len(rows))}


def _rehash(witness):
    witness["sha256"] = canonical_sha256({k: v for k, v in witness.items() if k != "sha256"})


def test_nontrivial_basis_and_coordinate_witness():
    inputs = _inputs()
    before = deepcopy(inputs)
    witness = build_css_equivalence_witness(**inputs)
    assert inputs == before
    assert witness["field"] == "GF(2)"
    assert witness["scope"] == "supplied_generator_basis_and_coordinate_witness_only"
    assert witness["transformed"] == {
        "hx": [[1, 1, 1, 1], [0, 1, 0, 1]],
        "hz": [[1, 1, 1, 1], [0, 0, 0, 0]],
    }
    parameters = {"commuting": True, "x_rank": 2, "z_rank": 1,
                  "logical_qubits": 1, "codespace_dimension": 2}
    assert witness["invariants"] == {
        "original": parameters, "transformed": parameters,
        "x_row_transform_rank": 2, "z_row_transform_rank": 2,
        "rank_preserved": True, "dimension_preserved": True,
        "sample_count": 2, "all_sampled_syndromes_covariant": True,
    }
    assert witness["samples"] == [
        {"index": 0, "transformed_error": {"x": [0, 1, 1, 0], "z": [0, 1, 1, 1]},
         "original_syndrome": {"hx": [0, 1], "hz": [0, 0]},
         "transformed_syndrome": {"hx": [1, 0], "hz": [0, 0]},
         "expected_syndrome": {"hx": [1, 0], "hz": [0, 0]}, "covariant": True},
        {"index": 1, "transformed_error": {"x": [0, 1, 0, 0], "z": [1, 0, 0, 0]},
         "original_syndrome": {"hx": [0, 1], "hz": [1, 1]},
         "transformed_syndrome": {"hx": [1, 0], "hz": [1, 0]},
         "expected_syndrome": {"hx": [1, 0], "hz": [1, 0]}, "covariant": True},
    ]
    assert verify_css_equivalence_witness(witness, **inputs) is None


def test_canonical_roundtrip_hash_replay_and_no_input_aliasing():
    inputs = _inputs()
    witness = build_css_equivalence_witness(**inputs)
    payload = {key: value for key, value in witness.items() if key != "sha256"}
    assert witness["sha256"] == canonical_sha256(payload)
    assert witness["sha256"] == "9dc3ec5a6d4df8dbcbc6c1ca2acce1345c91be20416ae7895bb95d7cc49abb9b"
    assert build_css_equivalence_witness(**deepcopy(inputs)) == witness
    assert canonical_json(witness) == canonical_json(build_css_equivalence_witness(**inputs))
    assert verify_css_equivalence_witness(json.loads(canonical_json(witness)), **inputs) is None
    inputs["hx"][0][0] = 0
    inputs["errors"][0][0][0] = 0
    assert witness["inputs"]["hx"][0][0] == 1
    assert witness["inputs"]["errors"][0]["x"][0] == 1
    witness["inputs"]["hz"][0][0] = 0
    assert inputs["hz"][0][0] == 1


def test_lists_and_tuples_share_canonical_results():
    def tuples(value):
        if type(value) in (list, tuple):
            return tuple(tuples(item) for item in value)
        return value
    inputs = _inputs()
    assert build_css_equivalence_witness(**inputs) == build_css_equivalence_witness(
        **{key: tuples(value) for key, value in inputs.items()})


def test_ordered_samples_are_bound_and_not_silently_sorted_or_deduplicated():
    inputs = _inputs()
    first = build_css_equivalence_witness(**inputs)
    inputs["errors"] = list(reversed(inputs["errors"]))
    second = build_css_equivalence_witness(**inputs)
    assert first["sha256"] != second["sha256"]
    assert first["samples"][0]["transformed_error"] == second["samples"][1]["transformed_error"]
    inputs["errors"] *= 2
    assert build_css_equivalence_witness(**inputs)["invariants"]["sample_count"] == 4


def test_all_two_generator_bases_and_four_coordinate_permutations():
    # Independent exhaustive GL(2, 2) enumeration by determinant; no production
    # rank/transform helper is used by this oracle.
    bases = [[[a, b], [c, d]] for a, b, c, d in product((0, 1), repeat=4)
             if (a * d + b * c) % 2 == 1]
    assert len(bases) == 6
    inputs = _inputs()
    for ux, uz, permutation in product(bases, bases, permutations(range(4))):
        inputs.update(x_row_transform=ux, z_row_transform=uz, qubit_permutation=permutation)
        witness = build_css_equivalence_witness(**inputs)
        for sector, basis in (("hx", ux), ("hz", uz)):
            combined = _multiply(basis, inputs[sector])
            expected = [[row[i] for i in permutation] for row in combined]
            assert witness["transformed"][sector] == expected
            # Same stabilizer row space after coordinate relabelling.
            original_permuted = [[row[i] for i in permutation] for row in inputs[sector]]
            assert _span(expected, 4) == _span(original_permuted, 4)
        assert witness["invariants"]["original"] == witness["invariants"]["transformed"]


def test_syndrome_covariance_for_every_four_qubit_pauli_component_pair():
    inputs = _inputs()
    vectors = list(product((0, 1), repeat=4))
    inputs["errors"] = list(product(vectors, repeat=2))
    witness = build_css_equivalence_witness(**inputs)
    assert len(witness["samples"]) == MAX_ERROR_SAMPLES == 256
    for (x, z), sample in zip(inputs["errors"], witness["samples"]):
        permuted_x = [x[i] for i in inputs["qubit_permutation"]]
        permuted_z = [z[i] for i in inputs["qubit_permutation"]]
        for sector, old_error, new_error, basis_name in (
            ("hx", z, permuted_z, "x_row_transform"),
            ("hz", x, permuted_x, "z_row_transform"),
        ):
            original = [_dot(row, old_error) for row in inputs[sector]]
            transformed = [_dot(row, new_error) for row in witness["transformed"][sector]]
            expected = [_dot(row, original) for row in inputs[basis_name]]
            assert sample["original_syndrome"][sector] == original
            assert sample["transformed_syndrome"][sector] == transformed == expected
            assert sample["expected_syndrome"][sector] == expected


@pytest.mark.parametrize("hx,hz,ux,uz,logical", [
    ([], [], [], [], 2),
    ([[1, 0]], [], [[1]], [], 1),
    ([], [[0, 1]], [], [[1]], 1),
    ([[1, 0]], [[0, 1]], [[1]], [[1]], 0),
    ([[0, 0], [0, 0]], [[0, 0]], [[1, 1], [0, 1]], [[1]], 2),
])
def test_empty_zero_and_full_rank_sectors(hx, hz, ux, uz, logical):
    inputs = dict(hx=hx, hz=hz, x_row_transform=ux, z_row_transform=uz,
                  qubit_permutation=[1, 0], errors=[([1, 0], [0, 1])])
    witness = build_css_equivalence_witness(**inputs)
    assert witness["invariants"]["original"]["logical_qubits"] == logical
    assert witness["invariants"]["original"]["codespace_dimension"] == 2 ** logical
    assert verify_css_equivalence_witness(witness, **inputs) is None



def test_binary_rank_differs_from_real_rank():
    # This matrix has real rank 3, but its three rows sum to zero over GF(2).
    checks = [[1, 1, 0], [1, 0, 1], [0, 1, 1]]
    witness = build_css_equivalence_witness(
        checks, [], x_row_transform=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        z_row_transform=[], qubit_permutation=[0, 1, 2],
    )
    assert len(_span(checks, 3)) == 4
    assert witness["invariants"]["original"]["x_rank"] == 2
    assert witness["invariants"]["original"]["logical_qubits"] == 1


def test_exact_rank_matches_exhaustive_small_row_spaces():
    for bits in product((0, 1), repeat=6):
        checks = [bits[:3], bits[3:]]
        witness = build_css_equivalence_witness(
            checks, [], x_row_transform=[[1, 1], [0, 1]],
            z_row_transform=[], qubit_permutation=[1, 2, 0],
        )
        expected_rank = len(_span(checks, 3)).bit_length() - 1
        assert witness["invariants"]["original"]["x_rank"] == expected_rank
        assert witness["invariants"]["transformed"]["x_rank"] == expected_rank


def test_identity_witness_and_explicit_zero_sample_scope():
    inputs = _inputs()
    inputs.update(x_row_transform=[[1, 0], [0, 1]], z_row_transform=[[1, 0], [0, 1]],
                  qubit_permutation=[0, 1, 2, 3], errors=[])
    witness = build_css_equivalence_witness(**inputs)
    assert witness["transformed"] == {key: inputs[key] for key in ("hx", "hz")}
    assert witness["samples"] == []
    assert witness["invariants"]["sample_count"] == 0
    assert witness["syndrome_scope"] == "supplied_errors_only"


@pytest.mark.parametrize("field,value", [
    ("hx", "1100"), ("hx", {0: [1, 1, 0, 0]}), ("hx", [[1, 1]]),
    ("hx", [[1, 1, 0, 0], [1]]), ("hz", [[1, 0, 0, 0]]),
    ("x_row_transform", [[1, 0]]), ("x_row_transform", [[1, 0], [1, 0]]),
    ("x_row_transform", [[1, 0, 0], [0, 1, 0]]),
    ("z_row_transform", [[1, 0], [1, 0]]),
    ("qubit_permutation", []), ("qubit_permutation", [0, 0, 2, 3]),
    ("qubit_permutation", [0, 1, 2, 4]), ("qubit_permutation", [-1, 1, 2, 3]),
    ("qubit_permutation", [False, 1, 2, 3]), ("qubit_permutation", [0.0, 1, 2, 3]),
    ("qubit_permutation", [0, 1]), ("qubit_permutation", {0, 1, 2, 3}),
    ("errors", None), ("errors", [[0, 1]]),
    ("errors", [([0, 0, 0], [0, 0, 0, 0])]),
    ("errors", [([0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0])]),
])
def test_invalid_shapes_transforms_commutation_and_permutations(field, value):
    inputs = _inputs()
    inputs[field] = value
    with pytest.raises(ValueError, match="^INVALID_INPUT$"):
        build_css_equivalence_witness(**inputs)


@pytest.mark.parametrize("bad_bit", [True, False, 0.0, 1.0, 2, -1, "1", None, Fraction(1)])
@pytest.mark.parametrize("target", ["hx", "hz", "x_row_transform", "z_row_transform", "errors"])
def test_binary_values_require_exact_ints_without_modulo_reduction(bad_bit, target):
    inputs = _inputs()
    if target == "errors":
        inputs[target][0][0][0] = bad_bit
    else:
        inputs[target][0][0] = bad_bit
    with pytest.raises(ValueError, match="^INVALID_INPUT$"):
        build_css_equivalence_witness(**inputs)


def test_iterators_and_container_subclasses_are_not_consumed_or_coerced():
    class ListSubclass(list):
        pass
    class IntSubclass(int):
        pass
    inputs = _inputs()
    for invalid in (iter(inputs["hx"]), ListSubclass(inputs["hx"]),
                    [[IntSubclass(1), 1, 0, 0], [0, 0, 1, 1]]):
        inputs["hx"] = invalid
        with pytest.raises(ValueError, match="^INVALID_INPUT$"):
            build_css_equivalence_witness(**inputs)


def test_resource_limits_reject_before_unbounded_work():
    inputs = _inputs()
    for field, invalid in (
        ("qubit_permutation", list(range(MAX_QUBITS + 1))),
        ("hx", [[0] * 4] * (MAX_CHECKS_PER_SECTOR + 1)),
        ("hz", [[0] * 4] * (MAX_CHECKS_PER_SECTOR + 1)),
        ("errors", [([0] * 4, [0] * 4)] * (MAX_ERROR_SAMPLES + 1)),
    ):
        with pytest.raises(ValueError, match="^INVALID_INPUT$"):
            build_css_equivalence_witness(**{**inputs, field: invalid})


def test_maximum_dimensions_are_supported():
    n, m = MAX_QUBITS, MAX_CHECKS_PER_SECTOR
    zero_checks = [[0] * n for _ in range(m)]
    identity = [[int(i == j) for j in range(m)] for i in range(m)]
    witness = build_css_equivalence_witness(
        zero_checks, zero_checks, x_row_transform=identity, z_row_transform=identity,
        qubit_permutation=list(reversed(range(n))), errors=[([0] * n, [0] * n)],
    )
    assert witness["invariants"]["original"]["logical_qubits"] == n
    assert witness["invariants"]["original"]["codespace_dimension"] == 1 << n


@pytest.mark.parametrize("path,value", [
    (("transformed", "hx", 0, 0), 0),
    (("transformed", "hz", 1, 0), 1),
    (("inputs", "hx", 0, 0), 0),
    (("inputs", "x_row_transform", 0, 0), 0),
    (("inputs", "qubit_permutation", 0), 0),
    (("inputs", "errors", 0, "x", 0), 0),
    (("invariants", "original", "logical_qubits"), 3),
    (("invariants", "transformed", "x_rank"), 1),
    (("invariants", "dimension_preserved"), False),
    (("samples", 0, "transformed_syndrome", "hx", 0), 0),
    (("samples", 0, "original_syndrome", "hx", 0), 1),
    (("samples", 0, "expected_syndrome", "hx", 0), 0),
    (("samples", 0, "transformed_error", "x", 0), 1),
    (("scope",), "arbitrary_code_equivalence_proven"),
    (("field",), "GF(3)"),
])
@pytest.mark.parametrize("rehash", [False, True])
def test_exact_verifier_rejects_tampering_even_with_a_recomputed_hash(path, value, rehash):
    inputs = _inputs()
    witness = build_css_equivalence_witness(**inputs)
    target = witness
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    if rehash:
        _rehash(witness)
    with pytest.raises(ValueError, match="^INVALID_INPUT$"):
        verify_css_equivalence_witness(witness, **inputs)


def test_exact_verifier_requires_trusted_external_inputs():
    inputs = _inputs()
    witness = build_css_equivalence_witness(**inputs)
    alternative = {**inputs, "qubit_permutation": [0, 1, 2, 3]}
    with pytest.raises(ValueError, match="^INVALID_INPUT$"):
        verify_css_equivalence_witness(witness, **alternative)
    alternative["z_row_transform"] = [[1, 0], [1, 0]]
    with pytest.raises(ValueError, match="^INVALID_INPUT$"):
        verify_css_equivalence_witness(witness, **alternative)


@pytest.mark.parametrize("mutation", ["extra", "missing", "hash", "bool_alias", "float", "tuple", "cycle"])
def test_exact_verifier_rejects_schema_aliases_and_malformed_trees(mutation):
    inputs = _inputs()
    witness = build_css_equivalence_witness(**inputs)
    if mutation == "extra":
        witness["extra"] = "untrusted"
    elif mutation == "missing":
        del witness["scope"]
    elif mutation == "hash":
        witness["sha256"] = "0" * 64
    elif mutation == "bool_alias":
        witness["inputs"]["hx"][0][0] = True
    elif mutation == "float":
        witness["inputs"]["hx"][0][0] = 1.0
    elif mutation == "tuple":
        witness["inputs"]["hx"] = tuple(witness["inputs"]["hx"])
    elif mutation == "cycle":
        witness["samples"][0]["transformed_error"]["x"] = witness
    with pytest.raises(ValueError, match="^INVALID_INPUT$"):
        verify_css_equivalence_witness(witness, **inputs)


@pytest.mark.parametrize("witness", [None, [], "witness", 0, True])
def test_exact_verifier_rejects_non_dict_values(witness):
    with pytest.raises(ValueError, match="^INVALID_INPUT$"):
        verify_css_equivalence_witness(witness, **_inputs())


def test_replay_is_identical_across_python_hash_seeds():
    source_root = Path(__file__).resolve().parents[1] / "src"
    script = (
        "from qec.experiments.css_equivalence import build_css_equivalence_witness\n"
        "from qec.sonify.canonical import canonical_json\n"
        f"print(canonical_json(build_css_equivalence_witness(**{_inputs()!r})))\n"
    )
    outputs = []
    for seed in ("0", "173"):
        environment = {**os.environ, "PYTHONHASHSEED": seed, "PYTHONPATH": str(source_root)}
        outputs.append(subprocess.run([sys.executable, "-c", script], env=environment,
                                      check=True, capture_output=True, text=True, timeout=10).stdout)
    assert outputs[0] == outputs[1] == canonical_json(build_css_equivalence_witness(**_inputs())) + "\n"
