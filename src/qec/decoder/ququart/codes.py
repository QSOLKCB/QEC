"""Reference additive ququart codes built from packed qubit lanes."""

from __future__ import annotations

from .packed import PackedQuquartCode


def packed_five_ququart_code() -> PackedQuquartCode:
    """Return the packed [[5,1,3]]_4 code.

    Each physical ququart stores two physical qubits. Each lane uses the
    perfect five-qubit [[5,1,3]]_2 stabilizer code, so the joint code stores two
    logical qubits, equivalently one logical ququart, in five physical
    ququarts. The 8 stabilizer generators leave a four-dimensional codespace.
    """

    # Five-qubit perfect code generators: XZZXI and cyclic shifts.
    labels = (
        "XZZXI",
        "IXZZX",
        "XIXZZ",
        "ZXIXZ",
    )
    rows: list[tuple[int, ...]] = []
    for word in labels:
        x = tuple(int(label in {"X", "Y"}) for label in word)
        z = tuple(int(label in {"Z", "Y"}) for label in word)
        rows.append(x + z)
    return PackedQuquartCode(
        tuple(rows),
        name="packed-[[5,1,3]]_4",
        distance_hint=3,
    )
