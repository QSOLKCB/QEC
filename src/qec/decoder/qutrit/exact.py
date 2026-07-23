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
        for weight in range(1, self.max_weight + 1):
            for candidate in paulis_of_weight(self.code.n, weight):
                syndrome = self.code.syndrome(candidate)
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
