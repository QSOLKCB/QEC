# SPDX-License-Identifier: MPL-2.0
"""Opt-in, bounded binary CSS presentation witnesses over GF(2).

Given unsigned commuting binary X/Z check matrices, invertible changes of
generator basis, one shared coordinate permutation, and supplied Pauli error
components, this experiment derives the new presentation and checks its
algebraic invariants.
For the convention ``new[j] = old[permutation[j]]``, it computes
``Hx' = Ux Hx P`` and ``Hz' = Uz Hz P``. Permuting both error components by
that same convention gives ``Hx' z' = Ux (Hx z)`` and ``Hz' x' = Uz (Hz x)``.

Inputs are bounded plain lists/tuples of exact binary Python ints; no modulo
reduction, numerical rank tolerance, or array coercion is performed. Empty
and redundant check sectors are supported. The permutation determines n,
including when both sectors are empty. Outputs are canonical JSON-safe
witness dictionaries, with a SHA-256 excluding its own field. Verification
recomputes the complete witness against separately supplied trusted inputs.

This is an isolated algebraic presentation experiment, not a search for
arbitrary code equivalence, a nonbinary construction, or a reproduction of
an external paper's algorithm. Checks represent conventional +1 X-only and
Z-only stabilizer generators; signed stabilizer phases and general local-
Clifford equivalence are unsupported. Generator-basis changes need not preserve
check weights or Tanner-graph degrees. No decoder or graph is mutated;
code distance, decoder performance, and physical behavior are not measured.
"""
from __future__ import annotations

from qec.sonify.canonical import canonical_sha256

MAX_QUBITS = 256
MAX_CHECKS_PER_SECTOR = 128
MAX_ERROR_SAMPLES = 256

Vector = tuple[int, ...]
Matrix = tuple[Vector, ...]


def _sequence(value: object, maximum: int) -> tuple:
    if type(value) not in (list, tuple) or len(value) > maximum:
        raise ValueError("INVALID_INPUT")
    return tuple(value)


def _vector(value: object, width: int) -> Vector:
    result = _sequence(value, width)
    if len(result) != width or any(type(bit) is not int or bit not in (0, 1) for bit in result):
        raise ValueError("INVALID_INPUT")
    return result


def _matrix(value: object, width: int, maximum_rows: int) -> Matrix:
    return tuple(_vector(row, width) for row in _sequence(value, maximum_rows))


def _packed(row: Vector) -> int:
    return sum(bit << column for column, bit in enumerate(row))


def _rank(rows: Matrix) -> int:
    """Exact GF(2) elimination, with deterministic left-to-right pivots."""
    work = [_packed(row) for row in rows]
    rank = 0
    for column in range(len(rows[0]) if rows else 0):
        pivot = next((i for i in range(rank, len(work)) if work[i] & (1 << column)), None)
        if pivot is None:
            continue
        work[rank], work[pivot] = work[pivot], work[rank]
        for i in range(rank + 1, len(work)):
            if work[i] & (1 << column):
                work[i] ^= work[rank]
        rank += 1
        if rank == len(work):
            break
    return rank


def _transform(rows: Matrix, basis: Matrix, permutation: tuple[int, ...]) -> Matrix:
    packed = tuple(_packed(row) for row in rows)
    result = []
    for coefficients in basis:
        combined = 0
        for bit, row in zip(coefficients, packed):
            if bit:
                combined ^= row
        result.append(tuple((combined >> old_column) & 1 for old_column in permutation))
    return tuple(result)


def _syndrome(rows: Matrix, error: Vector) -> Vector:
    packed_error = _packed(error)
    return tuple((_packed(row) & packed_error).bit_count() & 1 for row in rows)


def _parameters(hx: Matrix, hz: Matrix, n: int) -> dict:
    x_rank, z_rank = _rank(hx), _rank(hz)
    commuting = all(not any(_syndrome(hz, row)) for row in hx)
    if not commuting or x_rank + z_rank > n:
        raise ValueError("INVALID_INPUT")
    logical_qubits = n - x_rank - z_rank
    return {"commuting": commuting, "x_rank": x_rank, "z_rank": z_rank,
            "logical_qubits": logical_qubits, "codespace_dimension": 1 << logical_qubits}


def _rows(rows: Matrix) -> list[list[int]]:
    return [list(row) for row in rows]


def build_css_equivalence_witness(
    hx: object,
    hz: object,
    *,
    x_row_transform: object,
    z_row_transform: object,
    qubit_permutation: object,
    errors: object = (),
) -> dict:
    """Build a deterministic witness for an explicitly supplied relabelling.

    ``hx`` and ``hz`` have n columns and 0..128 rows each. Each row
    transformation is a square invertible binary matrix of the corresponding
    row count (the empty 0-by-0 transformation is valid). The permutation
    contains each integer in 0..n-1 exactly once, with 1 <= n <= 256.
    ``errors`` contains at most 256 ordered ``(x_component, z_component)``
    pairs of n bits. Hx detects the Z component, and Hz detects X.

    All invalid inputs raise ``ValueError("INVALID_INPUT")``. Sample order is
    significant and is retained. Zero samples assert no sampled observations.
    The returned hash binds this complete presentation and its ordered samples;
    it is not a canonical identifier for an equivalence class of codes.
    """
    permutation = _sequence(qubit_permutation, MAX_QUBITS)
    n = len(permutation)
    if not n or any(type(i) is not int or not 0 <= i < n for i in permutation):
        raise ValueError("INVALID_INPUT")
    if len(set(permutation)) != n:
        raise ValueError("INVALID_INPUT")
    x_checks = _matrix(hx, n, MAX_CHECKS_PER_SECTOR)
    z_checks = _matrix(hz, n, MAX_CHECKS_PER_SECTOR)
    ux = _matrix(x_row_transform, len(x_checks), len(x_checks))
    uz = _matrix(z_row_transform, len(z_checks), len(z_checks))
    if len(ux) != len(x_checks) or len(uz) != len(z_checks):
        raise ValueError("INVALID_INPUT")
    ux_rank, uz_rank = _rank(ux), _rank(uz)
    if ux_rank != len(ux) or uz_rank != len(uz):
        raise ValueError("INVALID_INPUT")
    samples = []
    for error in _sequence(errors, MAX_ERROR_SAMPLES):
        pair = _sequence(error, 2)
        if len(pair) != 2:
            raise ValueError("INVALID_INPUT")
        samples.append((_vector(pair[0], n), _vector(pair[1], n)))

    original_parameters = _parameters(x_checks, z_checks, n)
    transformed_x = _transform(x_checks, ux, permutation)
    transformed_z = _transform(z_checks, uz, permutation)
    transformed_parameters = _parameters(transformed_x, transformed_z, n)
    rank_preserved = all(original_parameters[key] == transformed_parameters[key]
                         for key in ("x_rank", "z_rank"))
    dimension_preserved = all(original_parameters[key] == transformed_parameters[key]
                              for key in ("logical_qubits", "codespace_dimension"))
    if not rank_preserved or not dimension_preserved:
        raise ValueError("INVALID_INPUT")

    observations = []
    for index, (x, z) in enumerate(samples):
        new_x = tuple(x[i] for i in permutation)
        new_z = tuple(z[i] for i in permutation)
        sx, sz = _syndrome(x_checks, z), _syndrome(z_checks, x)
        new_sx, new_sz = _syndrome(transformed_x, new_z), _syndrome(transformed_z, new_x)
        expected_sx, expected_sz = _syndrome(ux, sx), _syndrome(uz, sz)
        covariant = new_sx == expected_sx and new_sz == expected_sz
        if not covariant:
            raise ValueError("INVALID_INPUT")
        observations.append({
            "index": index,
            "transformed_error": {"x": list(new_x), "z": list(new_z)},
            "original_syndrome": {"hx": list(sx), "hz": list(sz)},
            "transformed_syndrome": {"hx": list(new_sx), "hz": list(new_sz)},
            "expected_syndrome": {"hx": list(expected_sx), "hz": list(expected_sz)},
            "covariant": covariant,
        })
    payload = {
        "schema": "qec.css-equivalence-witness.v1",
        "field": "GF(2)",
        "scope": "supplied_generator_basis_and_coordinate_witness_only",
        "permutation_convention": "new[j] = old[permutation[j]]",
        "syndrome_scope": "supplied_errors_only",
        "inputs": {"hx": _rows(x_checks), "hz": _rows(z_checks),
                   "x_row_transform": _rows(ux), "z_row_transform": _rows(uz),
                   "qubit_permutation": list(permutation),
                   "errors": [{"x": list(x), "z": list(z)} for x, z in samples]},
        "transformed": {"hx": _rows(transformed_x), "hz": _rows(transformed_z)},
        "invariants": {"original": original_parameters, "transformed": transformed_parameters,
                       "x_row_transform_rank": ux_rank, "z_row_transform_rank": uz_rank,
                       "rank_preserved": rank_preserved, "dimension_preserved": dimension_preserved,
                       "sample_count": len(samples),
                       "all_sampled_syndromes_covariant": all(item["covariant"] for item in observations)},
        "samples": observations,
    }
    return {**payload, "sha256": canonical_sha256(payload)}


def _require_exact(actual: object, expected: object) -> None:
    """Compare only the bounded expected shape; never serialize untrusted trees."""
    if type(actual) is not type(expected):
        raise ValueError("INVALID_INPUT")
    if type(expected) is dict:
        if len(actual) != len(expected) or any(type(key) is not str for key in actual):
            raise ValueError("INVALID_INPUT")
        if actual.keys() != expected.keys():
            raise ValueError("INVALID_INPUT")
        for key in expected:
            _require_exact(actual[key], expected[key])
    elif type(expected) is list:
        if len(actual) != len(expected):
            raise ValueError("INVALID_INPUT")
        for item, reference in zip(actual, expected):
            _require_exact(item, reference)
    elif actual != expected:
        raise ValueError("INVALID_INPUT")


def verify_css_equivalence_witness(
    witness: object,
    hx: object,
    hz: object,
    *,
    x_row_transform: object,
    z_row_transform: object,
    qubit_permutation: object,
    errors: object = (),
) -> None:
    """Recompute and require an exact witness, including its canonical hash.

    Pass the original inputs from a separately trusted source, not values
    extracted from the witness under verification. Rehashed modifications to
    matrices, transformations, samples, claims, or hashes are rejected. Success
    returns None; every invalid witness/input raises ValueError("INVALID_INPUT").
    This establishes exact replay, not source authenticity or external authority.
    """
    expected = build_css_equivalence_witness(
        hx, hz, x_row_transform=x_row_transform, z_row_transform=z_row_transform,
        qubit_permutation=qubit_permutation, errors=errors,
    )
    _require_exact(witness, expected)
