"""Typed ETQ-303 addressing for 101 sites with a three-state fibre."""

from __future__ import annotations

SITE_COUNT = 101
FIBRE_SIZE = 3
STATE_COUNT = SITE_COUNT * FIBRE_SIZE


def etq_index(site: int, symbol: int) -> int:
    """Map (site, qutrit symbol) bijectively into [0, 303)."""
    if isinstance(site, bool) or not isinstance(site, int):
        raise TypeError("ETQ site must be an integer")
    if isinstance(symbol, bool) or not isinstance(symbol, int):
        raise TypeError("ETQ symbol must be an integer")
    if not 0 <= site < SITE_COUNT:
        raise ValueError("ETQ site must be in [0, 101)")
    if not 0 <= symbol < FIBRE_SIZE:
        raise ValueError("ETQ symbol must be in [0, 3)")
    return FIBRE_SIZE * site + symbol


def etq_address(index: int) -> tuple[int, int]:
    """Invert ``etq_index`` exactly."""
    if isinstance(index, bool) or not isinstance(index, int):
        raise TypeError("ETQ index must be an integer")
    if not 0 <= index < STATE_COUNT:
        raise ValueError("ETQ index must be in [0, 303)")
    return divmod(index, FIBRE_SIZE)
