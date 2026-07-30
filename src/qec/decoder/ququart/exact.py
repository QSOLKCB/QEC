"""Exact bounded-weight decoding for the packed four-state code."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping, Sequence

from .gf2 import vector
from .packed import (
    BinarySyndrome,
    PackedPauli,
    PackedQuquartCode,
    identity,
    paulis_of_weight,
)


class UncorrectableSyndrome(ValueError):
    """Raised when no bounded-weight representative has the syndrome."""


@dataclass(frozen=True)
class DecodeResult:
    syndrome: BinarySyndrome
    correction: PackedPauli
    residual: PackedPauli
    success: bool


@dataclass(frozen=True)
class ExactDecoder:
    """Exact bounded-weight coset-leader decoder for packed ququart errors."""

    code: PackedQuquartCode
    max_weight: int = 1
    _table: Mapping[BinarySyndrome, PackedPauli] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if isinstance(self.max_weight, bool) or not isinstance(self.max_weight, int):
            raise TypeError("max_weight must be an integer")
        if self.max_weight < 0:
            raise ValueError("max_weight must be non-negative")
        ident = identity(self.code.n)
        table: dict[BinarySyndrome, PackedPauli] = {
            self.code.syndrome(ident): ident
        }
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

    @property
    def table_size(self) -> int:
        return len(self._table)

    def decode(self, syndrome: Sequence[int]) -> PackedPauli:
        key = vector(syndrome, width=self.code.syndrome_width)
        try:
            return self._table[key].inverse()
        except KeyError as error:
            raise UncorrectableSyndrome(
                f"no error of ququart weight <= {self.max_weight} has syndrome {key}"
            ) from error

    def correct(self, error: PackedPauli) -> DecodeResult:
        syndrome = self.code.syndrome(error)
        correction = self.decode(syndrome)
        residual = error.compose(correction)
        return DecodeResult(
            syndrome=syndrome,
            correction=correction,
            residual=residual,
            success=self.code.is_stabilizer(residual),
        )
