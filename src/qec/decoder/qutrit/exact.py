"""Exact bounded-weight qutrit decoding, with no score heuristics."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping, Sequence

from .gf3 import vector
from .stabilizer import (
    Pauli,
    QutritStabilizerCode,
    Syndrome,
    paulis_of_weight,
)


class UncorrectableSyndrome(ValueError):
    """Raised when no bounded-weight representative has the syndrome."""


def _one_site_syndromes(code: QutritStabilizerCode) -> tuple[tuple[Syndrome, ...], ...]:
    """Prepare immutable GF(3) contributions, indexed by site and ``3*x+z``.

    The sign matches symplectic(check, error): check_x*z - check_z*x.
    This plan is local to one construction, never cached by a code's name.
    """
    return tuple(
        tuple(
            tuple((check[site] * z - check[code.n + site] * x) % 3
                  for check in code.stabilizers)
            for x in range(3) for z in range(3)
        )
        for site in range(code.n)
    )


def _sparse_syndrome(error: Pauli, plan: tuple[tuple[Syndrome, ...], ...]) -> Syndrome:
    """Combine nonzero-site contributions for a Pauli of the plan's width."""
    contributions = [
        site[3 * x + z]
        for site, x, z in zip(plan, error.x, error.z)
        if x or z
    ]
    if not contributions:
        return plan[0][0]
    if len(contributions) == 1:
        return contributions[0]
    return tuple(sum(values) % 3 for values in zip(*contributions))


@dataclass(frozen=True)
class DecodeResult:
    syndrome: Syndrome
    correction: Pauli
    residual: Pauli
    success: bool


@dataclass(frozen=True)
class ExactDecoder:
    """Exact bounded-weight coset-leader decoder."""

    code: QutritStabilizerCode
    max_weight: int = 1
    _table: Mapping[Syndrome, Pauli] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.max_weight < 0:
            raise ValueError("max_weight must be non-negative")
        identity = Pauli((0,) * self.code.n, (0,) * self.code.n)
        table: dict[Syndrome, Pauli] = {self.code.syndrome(identity): identity}
        # Preserve custom syndrome semantics on subclasses. The public scalar
        # syndrome path (including correct()) remains unchanged and independent.
        plan = (
            _one_site_syndromes(self.code)
            if type(self.code) is QutritStabilizerCode and self.max_weight > 0
            else None
        )
        for weight in range(1, self.max_weight + 1):
            for candidate in paulis_of_weight(self.code.n, weight):
                syndrome = (
                    _sparse_syndrome(candidate, plan)
                    if plan is not None else self.code.syndrome(candidate)
                )
                previous = table.get(syndrome)
                if previous is None:
                    table[syndrome] = candidate
                    continue
                difference = candidate.compose(previous.inverse())
                if not self.code.is_stabilizer(difference):
                    raise ValueError(
                        "bounded error set violates exact correction conditions"
                    )
        object.__setattr__(self, "_table", MappingProxyType(table))

    def decode(self, syndrome: Sequence[int]) -> Pauli:
        key = vector(syndrome, width=len(self.code.stabilizers))
        try:
            return self._table[key].inverse()
        except KeyError as error:
            raise UncorrectableSyndrome(
                f"no error of weight <= {self.max_weight} has syndrome {key}"
            ) from error

    def correct(self, error: Pauli) -> DecodeResult:
        syndrome = self.code.syndrome(error)
        correction = self.decode(syndrome)
        residual = error.compose(correction)
        return DecodeResult(
            syndrome=syndrome,
            correction=correction,
            residual=residual,
            success=self.code.is_stabilizer(residual),
        )
