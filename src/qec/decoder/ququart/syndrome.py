"""Bijective packing between paired binary checks and four-state symbols."""

from __future__ import annotations

from numbers import Integral
from typing import Iterable, Sequence

from .gf2 import Vector, vector

QuquartSyndrome = tuple[int, ...]


def symbol(value: int) -> int:
    """Return an exact four-state symbol in {0,1,2,3}."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError("ququart symbols must be integers")
    result = int(value)
    if result < 0 or result > 3:
        raise ValueError("ququart symbols must be in the range 0..3")
    return result


def symbols(values: Iterable[int], *, width: int | None = None) -> QuquartSyndrome:
    result = tuple(symbol(value) for value in values)
    if width is not None and len(result) != width:
        raise ValueError(f"expected syndrome width {width}, got {len(result)}")
    return result


def pack_binary_syndrome(syndrome: Sequence[int]) -> QuquartSyndrome:
    """Pack two equal GF(2) syndrome lanes as |q0 q1> -> 2*q0+q1."""
    bits = vector(syndrome)
    if not bits or len(bits) % 2:
        raise ValueError("binary syndrome must have equal non-empty lane halves")
    middle = len(bits) // 2
    return tuple(2 * left + right for left, right in zip(bits[:middle], bits[middle:]))


def unpack_ququart_syndrome(syndrome: Sequence[int]) -> Vector:
    """Unpack four-state syndrome symbols into lane-0 then lane-1 bits."""
    packed = symbols(syndrome)
    if not packed:
        raise ValueError("ququart syndrome must be non-empty")
    lane0 = tuple(value // 2 for value in packed)
    lane1 = tuple(value % 2 for value in packed)
    return lane0 + lane1
