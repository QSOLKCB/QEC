"""Declared qutrit and qubit comparison codes."""

from __future__ import annotations

from functools import lru_cache

from qec.decoder.qutrit import (
    cyclic_five_qutrit_code,
    shor_nine_qutrit_code,
    ternary_golay_qutrit_code,
)

from .prime import Matrix, PrimeStabilizerModel

TERNARY_GOLAY_SOURCE = "https://arxiv.org/abs/2003.02717"
QSOL_QEC_SOURCE = "https://github.com/QSOLKCB/QEC"
FIVE_QUBIT_SOURCE = "https://doi.org/10.1103/PhysRevLett.77.198"
STEANE_SOURCE = "https://arxiv.org/abs/quant-ph/9601029"
SURFACE_SOURCE = "https://www.nature.com/articles/s41586-024-08449-y"
REED_MULLER_SOURCE = "https://arxiv.org/abs/2506.14169"


def _qutrit_model(factory, code_id: str, label: str, radius: int) -> PrimeStabilizerModel:
    code = factory()
    return PrimeStabilizerModel(
        code_id=code_id,
        label=label,
        family="qutrit_stabilizer",
        origin="qsol_new_decoder",
        modulus=3,
        n=code.n,
        k=code.k,
        distance=code.distance_hint or 0,
        radius=radius,
        checks=code.stabilizers,
        source_url=(
            TERNARY_GOLAY_SOURCE
            if code_id == "qutrit_golay_11"
            else QSOL_QEC_SOURCE
        ),
    )


def _css_checks(x_rows: tuple[tuple[int, ...], ...], z_rows: tuple[tuple[int, ...], ...]) -> Matrix:
    width = len(x_rows[0] if x_rows else z_rows[0])
    zero = (0,) * width
    return tuple(row + zero for row in x_rows) + tuple(
        zero + row for row in z_rows
    )


def _indicator_rows(width: int, supports: tuple[tuple[int, ...], ...]) -> tuple[tuple[int, ...], ...]:
    return tuple(
        tuple(int(site in support) for site in range(width))
        for support in supports
    )


def _binary_nullspace(rows: tuple[tuple[int, ...], ...]) -> tuple[tuple[int, ...], ...]:
    work = [list(row) for row in rows]
    width = len(work[0])
    pivots: list[int] = []
    pivot_row = 0
    for column in range(width):
        pivot = next(
            (row for row in range(pivot_row, len(work)) if work[row][column]),
            None,
        )
        if pivot is None:
            continue
        work[pivot_row], work[pivot] = work[pivot], work[pivot_row]
        for row, values in enumerate(work):
            if row != pivot_row and values[column]:
                work[row] = [a ^ b for a, b in zip(values, work[pivot_row])]
        pivots.append(column)
        pivot_row += 1

    basis = []
    for free in (column for column in range(width) if column not in pivots):
        vector = [0] * width
        vector[free] = 1
        for row, pivot in enumerate(pivots):
            vector[pivot] = work[row][free]
        basis.append(tuple(vector))
    return tuple(basis)


def _binary_model(
    *,
    code_id: str,
    label: str,
    family: str,
    n: int,
    k: int,
    distance: int,
    checks: Matrix,
    source_url: str,
) -> PrimeStabilizerModel:
    return PrimeStabilizerModel(
        code_id=code_id,
        label=label,
        family=family,
        origin="industry_reference_code",
        modulus=2,
        n=n,
        k=k,
        distance=distance,
        radius=(distance - 1) // 2,
        checks=checks,
        source_url=source_url,
    )


def _five_qubit() -> PrimeStabilizerModel:
    rows = []
    for shift in range(4):
        x, z = [0] * 5, [0] * 5
        for site in (0, 3):
            x[(site + shift) % 5] = 1
        for site in (1, 2):
            z[(site + shift) % 5] = 1
        rows.append(tuple(x + z))
    return _binary_model(
        code_id="five_qubit_5",
        label="Five-qubit [[5,1,3]]",
        family="perfect_stabilizer",
        n=5,
        k=1,
        distance=3,
        checks=tuple(rows),
        source_url=FIVE_QUBIT_SOURCE,
    )


def _steane() -> PrimeStabilizerModel:
    hamming = (
        (1, 0, 0, 1, 1, 0, 1),
        (0, 1, 0, 1, 0, 1, 1),
        (0, 0, 1, 0, 1, 1, 1),
    )
    return _binary_model(
        code_id="steane_7",
        label="Steane [[7,1,3]]",
        family="color_css",
        n=7,
        k=1,
        distance=3,
        checks=_css_checks(hamming, hamming),
        source_url=STEANE_SOURCE,
    )


def _surface_d3() -> PrimeStabilizerModel:
    x_rows = _indicator_rows(9, ((0, 1, 3, 4), (4, 5, 7, 8), (1, 2), (6, 7)))
    z_rows = _indicator_rows(9, ((1, 2, 4, 5), (3, 4, 6, 7), (0, 3), (5, 8)))
    return _binary_model(
        code_id="surface_rotated_d3",
        label="Rotated surface [[9,1,3]]",
        family="surface",
        n=9,
        k=1,
        distance=3,
        checks=_css_checks(x_rows, z_rows),
        source_url=SURFACE_SOURCE,
    )


def _reed_muller() -> PrimeStabilizerModel:
    points = tuple(
        tuple((value >> bit) & 1 for bit in range(4))
        for value in range(1, 16)
    )
    x_rows = tuple(tuple(point[bit] for point in points) for bit in range(4))
    z_rows = _binary_nullspace(((1,) * 15,) + x_rows)
    return _binary_model(
        code_id="reed_muller_15",
        label="Quantum Reed-Muller [[15,1,3]]",
        family="reed_muller_css",
        n=15,
        k=1,
        distance=3,
        checks=_css_checks(x_rows, z_rows),
        source_url=REED_MULLER_SOURCE,
    )


@lru_cache(maxsize=1)
def benchmark_models() -> tuple[PrimeStabilizerModel, ...]:
    """Return models in canonical report order."""
    return (
        _qutrit_model(cyclic_five_qutrit_code, "qutrit_cyclic_5", "Qutrit cyclic [[5,1,3]]₃", 1),
        _qutrit_model(shor_nine_qutrit_code, "qutrit_shor_9", "Qutrit Shor [[9,1,3]]₃", 1),
        _qutrit_model(ternary_golay_qutrit_code, "qutrit_golay_11", "Ternary Golay [[11,1,5]]₃", 2),
        _five_qubit(),
        _steane(),
        _surface_d3(),
        _reed_muller(),
    )
