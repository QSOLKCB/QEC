from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

from qec.analysis.canonical_hashing import sha256_hex
from qec.analysis.layer_spec_contract import _deep_freeze, _ensure_json_safe

READOUT_MATRIX_VERSION = "v153.8"
MATRIX_CONSTRUCTION_RULE = "ORDERED_SHELL_KERNEL_PRODUCT_V1"
MARKOV_BASIS_RULE = "DETERMINISTIC_TRANSITION_BASIS_V1"
TRANSITION_RULE = "EXPLICIT_STATE_IDENTITY_TRANSITION_V1"

MAX_MATRIX_ROWS = 128
MAX_MATRIX_COLUMNS = 128
MAX_MATRIX_CELLS = 16384
MAX_TRANSITIONS = 16384

_ALLOWED_ROW_SOURCE_TYPES = {
    "READOUT_SHELL_STACK",
    "READOUT_SHELL_SET",
    "COMPOSITION_RECEIPT_SET",
    "GENERIC_ORDERED_READOUT_SET",
}
_ALLOWED_COLUMN_SOURCE_TYPES = {
    "CORE_KERNEL_SET",
    "DERIVED_KERNEL_SET",
    "KERNEL_STATE_SET",
    "GENERIC_ORDERED_KERNEL_SET",
}
_ALLOWED_TRANSITION_REASONS = {
    "EXPLICIT_ADJACENT_TRANSITION",
    "INVALID_TRANSITION_INDEX",
    "STATE_HASH_MISMATCH",
}


def _is_sha256_hex(v: str) -> bool:
    return isinstance(v, str) and len(v) == 64 and all(c in "0123456789abcdef" for c in v)


def _thaw(v: Any) -> Any:
    if isinstance(v, MappingProxyType):
        return {k: _thaw(x) for k, x in v.items()}
    if isinstance(v, tuple):
        return [_thaw(x) for x in v]
    return v


def _freeze_cells(cells: tuple[Mapping[str, Any], ...]) -> tuple[Mapping[str, Any], ...]:
    return tuple(MappingProxyType(dict(_deep_freeze(dict(c)))) for c in cells)


def _row_order_hash(row_ids: tuple[str, ...], row_hashes: tuple[str, ...]) -> str:
    return sha256_hex({"ordered_row_ids": list(row_ids), "ordered_row_hashes": list(row_hashes)})


def _column_order_hash(column_ids: tuple[str, ...], column_hashes: tuple[str, ...]) -> str:
    return sha256_hex({"ordered_column_ids": list(column_ids), "ordered_column_hashes": list(column_hashes)})


def _cell_identity(r: int, row_id: str, row_hash: str, c: int, column_id: str, column_hash: str) -> str:
    return sha256_hex(
        {
            "matrix_rule": MATRIX_CONSTRUCTION_RULE,
            "row_index": r,
            "row_id": row_id,
            "row_hash": row_hash,
            "column_index": c,
            "column_id": column_id,
            "column_hash": column_hash,
        }
    )


def _transition_identity(from_state_hash: str, to_state_hash: str, transition_index: int) -> str:
    return sha256_hex(
        {
            "transition_rule": TRANSITION_RULE,
            "from_state_hash": from_state_hash,
            "to_state_hash": to_state_hash,
            "transition_index": transition_index,
        }
    )


@dataclass(frozen=True)
class ReadoutCombinationMatrix:
    matrix_id: str
    matrix_version: str
    row_source_type: str
    column_source_type: str
    matrix_construction_rule: str
    ordered_row_ids: tuple[str, ...]
    ordered_row_hashes: tuple[str, ...]
    ordered_column_ids: tuple[str, ...]
    ordered_column_hashes: tuple[str, ...]
    row_count: int
    column_count: int
    cell_count: int
    cells: tuple[Mapping[str, Any], ...]
    matrix_hash: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "ordered_row_ids", tuple(self.ordered_row_ids))
        object.__setattr__(self, "ordered_row_hashes", tuple(self.ordered_row_hashes))
        object.__setattr__(self, "ordered_column_ids", tuple(self.ordered_column_ids))
        object.__setattr__(self, "ordered_column_hashes", tuple(self.ordered_column_hashes))
        object.__setattr__(self, "cells", _freeze_cells(tuple(self.cells)))
        if not self.matrix_id or self.matrix_version != READOUT_MATRIX_VERSION:
            raise ValueError("INVALID_INPUT")
        if self.row_source_type not in _ALLOWED_ROW_SOURCE_TYPES or self.column_source_type not in _ALLOWED_COLUMN_SOURCE_TYPES:
            raise ValueError("INVALID_INPUT")
        if self.matrix_construction_rule != MATRIX_CONSTRUCTION_RULE:
            raise ValueError("INVALID_INPUT")
        if len(self.ordered_row_ids) != len(self.ordered_row_hashes) or len(self.ordered_column_ids) != len(self.ordered_column_hashes):
            raise ValueError("INVALID_INPUT")
        if any(not isinstance(x, str) or not x for x in self.ordered_row_ids + self.ordered_column_ids):
            raise ValueError("INVALID_INPUT")
        if any(not _is_sha256_hex(x) for x in self.ordered_row_hashes + self.ordered_column_hashes):
            raise ValueError("INVALID_INPUT")
        if len(set(self.ordered_row_ids)) != len(self.ordered_row_ids) or len(set(self.ordered_row_hashes)) != len(self.ordered_row_hashes):
            raise ValueError("INVALID_INPUT")
        if len(set(self.ordered_column_ids)) != len(self.ordered_column_ids) or len(set(self.ordered_column_hashes)) != len(self.ordered_column_hashes):
            raise ValueError("INVALID_INPUT")
        if not (0 < len(self.ordered_row_ids) <= MAX_MATRIX_ROWS and 0 < len(self.ordered_column_ids) <= MAX_MATRIX_COLUMNS):
            raise ValueError("INVALID_INPUT")
        if self.row_count != len(self.ordered_row_ids) or self.column_count != len(self.ordered_column_ids):
            raise ValueError("INVALID_INPUT")
        if self.row_count * self.column_count > MAX_MATRIX_CELLS or self.cell_count != self.row_count * self.column_count:
            raise ValueError("INVALID_INPUT")
        if len(self.cells) != self.cell_count:
            raise ValueError("INVALID_INPUT")
        for i, cell in enumerate(self.cells):
            if not isinstance(cell, Mapping):
                raise ValueError("INVALID_INPUT")
            r = i // self.column_count
            c = i % self.column_count
            if cell.get("row_index") != r or cell.get("column_index") != c:
                raise ValueError("INVALID_INPUT")
            row_id = self.ordered_row_ids[r]
            col_id = self.ordered_column_ids[c]
            if cell.get("row_id") != row_id or cell.get("column_id") != col_id:
                raise ValueError("INVALID_INPUT")
            if cell.get("row_hash") != self.ordered_row_hashes[r] or cell.get("column_hash") != self.ordered_column_hashes[c]:
                raise ValueError("INVALID_INPUT")
            if cell.get("cell_id") != f"{row_id}::{col_id}":
                raise ValueError("INVALID_INPUT")
            expected = _cell_identity(r, row_id, self.ordered_row_hashes[r], c, col_id, self.ordered_column_hashes[c])
            if cell.get("cell_output_identity_hash") != expected:
                raise ValueError("INVALID_INPUT")
        if self.matrix_hash != self.stable_hash():
            raise ValueError("INVALID_INPUT")

    def _canonical_payload(self) -> dict[str, Any]:
        payload = {
            "matrix_id": self.matrix_id,
            "matrix_version": self.matrix_version,
            "row_source_type": self.row_source_type,
            "column_source_type": self.column_source_type,
            "matrix_construction_rule": self.matrix_construction_rule,
            "ordered_row_ids": list(self.ordered_row_ids),
            "ordered_row_hashes": list(self.ordered_row_hashes),
            "ordered_column_ids": list(self.ordered_column_ids),
            "ordered_column_hashes": list(self.ordered_column_hashes),
            "row_count": self.row_count,
            "column_count": self.column_count,
            "cell_count": self.cell_count,
            "cells": [_thaw(c) for c in self.cells],
        }
        _ensure_json_safe(payload)
        return payload

    def stable_hash(self) -> str:
        return sha256_hex(self._canonical_payload())

    def to_dict(self) -> dict[str, Any]:
        return dict(self._canonical_payload(), matrix_hash=self.matrix_hash)


@dataclass(frozen=True)
class ReadoutMatrixReceipt:
    matrix_receipt_id: str
    matrix_version: str
    matrix_hash: str
    matrix_construction_rule: str
    row_source_type: str
    column_source_type: str
    row_count: int
    column_count: int
    cell_count: int
    row_order_hash: str
    column_order_hash: str
    cell_identity_hashes: tuple[str, ...]
    matrix_receipt_hash: str
    receipt_hash: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "cell_identity_hashes", tuple(self.cell_identity_hashes))
        if not self.matrix_receipt_id or self.matrix_version != READOUT_MATRIX_VERSION:
            raise ValueError("INVALID_INPUT")
        if self.matrix_construction_rule != MATRIX_CONSTRUCTION_RULE:
            raise ValueError("INVALID_INPUT")
        if self.row_source_type not in _ALLOWED_ROW_SOURCE_TYPES or self.column_source_type not in _ALLOWED_COLUMN_SOURCE_TYPES:
            raise ValueError("INVALID_INPUT")
        if any(not _is_sha256_hex(x) for x in (self.matrix_hash, self.row_order_hash, self.column_order_hash, self.matrix_receipt_hash, self.receipt_hash)):
            raise ValueError("INVALID_INPUT")
        if any(not _is_sha256_hex(x) for x in self.cell_identity_hashes):
            raise ValueError("INVALID_INPUT")
        if self.matrix_receipt_hash != sha256_hex(self._d()):
            raise ValueError("INVALID_INPUT")
        if self.receipt_hash != self.stable_hash():
            raise ValueError("INVALID_INPUT")

    def _d(self) -> dict[str, Any]:
        return {
            "matrix_receipt_id": self.matrix_receipt_id,
            "matrix_version": self.matrix_version,
            "matrix_hash": self.matrix_hash,
            "matrix_construction_rule": self.matrix_construction_rule,
            "row_source_type": self.row_source_type,
            "column_source_type": self.column_source_type,
            "row_count": self.row_count,
            "column_count": self.column_count,
            "cell_count": self.cell_count,
            "row_order_hash": self.row_order_hash,
            "column_order_hash": self.column_order_hash,
            "cell_identity_hashes": list(self.cell_identity_hashes),
        }

    def _c(self) -> dict[str, Any]:
        payload = dict(self._d(), matrix_receipt_hash=self.matrix_receipt_hash)
        _ensure_json_safe(payload)
        return payload

    def stable_hash(self) -> str:
        return sha256_hex(self._c())

    def to_dict(self) -> dict[str, Any]:
        return dict(self._c(), receipt_hash=self.receipt_hash)


@dataclass(frozen=True)
class MarkovBasisReceipt:
    markov_basis_id: str
    matrix_version: str
    readout_matrix_receipt_hash: str
    markov_basis_rule: str
    state_identity_hashes: tuple[str, ...]
    transition_identity_hashes: tuple[str, ...]
    transition_count: int
    deterministic_transition_basis_hash: str
    receipt_hash: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "state_identity_hashes", tuple(self.state_identity_hashes))
        object.__setattr__(self, "transition_identity_hashes", tuple(self.transition_identity_hashes))
        if not self.markov_basis_id or self.matrix_version != READOUT_MATRIX_VERSION or self.markov_basis_rule != MARKOV_BASIS_RULE:
            raise ValueError("INVALID_INPUT")
        if not _is_sha256_hex(self.readout_matrix_receipt_hash):
            raise ValueError("INVALID_INPUT")
        if any(not _is_sha256_hex(x) for x in self.state_identity_hashes + self.transition_identity_hashes):
            raise ValueError("INVALID_INPUT")
        if len(set(self.state_identity_hashes)) != len(self.state_identity_hashes):
            raise ValueError("INVALID_INPUT")
        if self.transition_count != len(self.transition_identity_hashes):
            raise ValueError("INVALID_INPUT")
        if self.transition_count != max(0, len(self.state_identity_hashes) - 1) or self.transition_count > MAX_TRANSITIONS:
            raise ValueError("INVALID_INPUT")
        if self.deterministic_transition_basis_hash != sha256_hex(self._d()) or self.receipt_hash != self.stable_hash():
            raise ValueError("INVALID_INPUT")

    def _d(self) -> dict[str, Any]:
        return {
            "markov_basis_id": self.markov_basis_id,
            "matrix_version": self.matrix_version,
            "readout_matrix_receipt_hash": self.readout_matrix_receipt_hash,
            "markov_basis_rule": self.markov_basis_rule,
            "state_identity_hashes": list(self.state_identity_hashes),
            "transition_identity_hashes": list(self.transition_identity_hashes),
            "transition_count": self.transition_count,
        }

    def _c(self) -> dict[str, Any]:
        payload = dict(self._d(), deterministic_transition_basis_hash=self.deterministic_transition_basis_hash)
        _ensure_json_safe(payload)
        return payload

    def stable_hash(self) -> str:
        return sha256_hex(self._c())

    def to_dict(self) -> dict[str, Any]:
        return dict(self._c(), receipt_hash=self.receipt_hash)


@dataclass(frozen=True)
class ReadoutTransitionReceipt:
    transition_receipt_id: str
    matrix_version: str
    markov_basis_receipt_hash: str
    transition_rule: str
    transition_index: int
    from_state_hash: str
    to_state_hash: str
    transition_identity_hash: str
    transition_valid: bool
    transition_reason: str
    receipt_hash: str

    def __post_init__(self) -> None:
        if not self.transition_receipt_id or self.matrix_version != READOUT_MATRIX_VERSION or self.transition_rule != TRANSITION_RULE:
            raise ValueError("INVALID_INPUT")
        if isinstance(self.transition_index, bool) or not isinstance(self.transition_index, int) or self.transition_index < 0:
            raise ValueError("INVALID_INPUT")
        if any(not _is_sha256_hex(x) for x in (self.markov_basis_receipt_hash, self.from_state_hash, self.to_state_hash, self.transition_identity_hash, self.receipt_hash)):
            raise ValueError("INVALID_INPUT")
        if self.transition_reason not in _ALLOWED_TRANSITION_REASONS:
            raise ValueError("INVALID_INPUT")
        if self.transition_identity_hash != _transition_identity(self.from_state_hash, self.to_state_hash, self.transition_index):
            raise ValueError("INVALID_INPUT")
        if self.receipt_hash != self.stable_hash():
            raise ValueError("INVALID_INPUT")

    def _c(self) -> dict[str, Any]:
        payload = {
            "transition_receipt_id": self.transition_receipt_id,
            "matrix_version": self.matrix_version,
            "markov_basis_receipt_hash": self.markov_basis_receipt_hash,
            "transition_rule": self.transition_rule,
            "transition_index": self.transition_index,
            "from_state_hash": self.from_state_hash,
            "to_state_hash": self.to_state_hash,
            "transition_identity_hash": self.transition_identity_hash,
            "transition_valid": self.transition_valid,
            "transition_reason": self.transition_reason,
        }
        _ensure_json_safe(payload)
        return payload

    def stable_hash(self) -> str:
        return sha256_hex(self._c())

    def to_dict(self) -> dict[str, Any]:
        return dict(self._c(), receipt_hash=self.receipt_hash)


def build_readout_combination_matrix(matrix_id: str, row_source_type: str, column_source_type: str, ordered_row_ids: tuple[str, ...], ordered_row_hashes: tuple[str, ...], ordered_column_ids: tuple[str, ...], ordered_column_hashes: tuple[str, ...]) -> ReadoutCombinationMatrix:
    row_ids = tuple(ordered_row_ids)
    row_hashes = tuple(ordered_row_hashes)
    col_ids = tuple(ordered_column_ids)
    col_hashes = tuple(ordered_column_hashes)
    cells: list[dict[str, Any]] = []
    for r, (rid, rh) in enumerate(zip(row_ids, row_hashes)):
        for c, (cid, ch) in enumerate(zip(col_ids, col_hashes)):
            cells.append({"cell_id": f"{rid}::{cid}", "row_index": r, "row_id": rid, "row_hash": rh, "column_index": c, "column_id": cid, "column_hash": ch, "cell_output_identity_hash": _cell_identity(r, rid, rh, c, cid, ch)})
    base = {
        "matrix_id": matrix_id,
        "matrix_version": READOUT_MATRIX_VERSION,
        "row_source_type": row_source_type,
        "column_source_type": column_source_type,
        "matrix_construction_rule": MATRIX_CONSTRUCTION_RULE,
        "ordered_row_ids": row_ids,
        "ordered_row_hashes": row_hashes,
        "ordered_column_ids": col_ids,
        "ordered_column_hashes": col_hashes,
        "row_count": len(row_ids),
        "column_count": len(col_ids),
        "cell_count": len(cells),
        "cells": tuple(cells),
    }
    payload = {**base, "ordered_row_ids": list(row_ids), "ordered_row_hashes": list(row_hashes), "ordered_column_ids": list(col_ids), "ordered_column_hashes": list(col_hashes), "cells": cells}
    matrix_hash = sha256_hex(payload)
    return ReadoutCombinationMatrix(**base, matrix_hash=matrix_hash)


def build_readout_matrix_receipt(matrix_receipt_id: str, matrix: ReadoutCombinationMatrix) -> ReadoutMatrixReceipt:
    d = {
        "matrix_receipt_id": matrix_receipt_id,
        "matrix_version": READOUT_MATRIX_VERSION,
        "matrix_hash": matrix.stable_hash(),
        "matrix_construction_rule": MATRIX_CONSTRUCTION_RULE,
        "row_source_type": matrix.row_source_type,
        "column_source_type": matrix.column_source_type,
        "row_count": matrix.row_count,
        "column_count": matrix.column_count,
        "cell_count": matrix.cell_count,
        "row_order_hash": _row_order_hash(matrix.ordered_row_ids, matrix.ordered_row_hashes),
        "column_order_hash": _column_order_hash(matrix.ordered_column_ids, matrix.ordered_column_hashes),
        "cell_identity_hashes": tuple(c["cell_output_identity_hash"] for c in matrix.cells),
    }
    matrix_receipt_hash = sha256_hex({**d, "cell_identity_hashes": list(d["cell_identity_hashes"])})
    receipt_payload = {**d, "matrix_receipt_hash": matrix_receipt_hash}
    receipt_hash = sha256_hex(receipt_payload)
    return ReadoutMatrixReceipt(**d, matrix_receipt_hash=matrix_receipt_hash, receipt_hash=receipt_hash)


def validate_readout_matrix_receipt(receipt: ReadoutMatrixReceipt, matrix: ReadoutCombinationMatrix) -> None:
    expected = build_readout_matrix_receipt(receipt.matrix_receipt_id, matrix)
    if receipt.to_dict() != expected.to_dict():
        raise ValueError("INVALID_INPUT")


def build_markov_basis_receipt(markov_basis_id: str, matrix: ReadoutCombinationMatrix, matrix_receipt: ReadoutMatrixReceipt) -> MarkovBasisReceipt:
    validate_readout_matrix_receipt(matrix_receipt, matrix)
    states = tuple(c["cell_output_identity_hash"] for c in matrix.cells)
    transitions = tuple(_transition_identity(states[i], states[i + 1], i) for i in range(max(0, len(states) - 1)))
    d = {
        "markov_basis_id": markov_basis_id,
        "matrix_version": READOUT_MATRIX_VERSION,
        "readout_matrix_receipt_hash": matrix_receipt.receipt_hash,
        "markov_basis_rule": MARKOV_BASIS_RULE,
        "state_identity_hashes": states,
        "transition_identity_hashes": transitions,
        "transition_count": len(transitions),
    }
    basis_hash = sha256_hex({**d, "state_identity_hashes": list(states), "transition_identity_hashes": list(transitions)})
    receipt_hash = sha256_hex({**d, "state_identity_hashes": list(states), "transition_identity_hashes": list(transitions), "deterministic_transition_basis_hash": basis_hash})
    return MarkovBasisReceipt(**d, deterministic_transition_basis_hash=basis_hash, receipt_hash=receipt_hash)


def validate_markov_basis_receipt(receipt: MarkovBasisReceipt, matrix: ReadoutCombinationMatrix, matrix_receipt: ReadoutMatrixReceipt) -> None:
    expected = build_markov_basis_receipt(receipt.markov_basis_id, matrix, matrix_receipt)
    if receipt.to_dict() != expected.to_dict():
        raise ValueError("INVALID_INPUT")


def build_readout_transition_receipt(transition_receipt_id: str, markov_basis: MarkovBasisReceipt, transition_index: int) -> ReadoutTransitionReceipt:
    if isinstance(transition_index, bool) or not isinstance(transition_index, int):
        raise ValueError("INVALID_INPUT")
    if transition_index < 0 or transition_index >= markov_basis.transition_count:
        raise ValueError("INVALID_INPUT")
    from_state = markov_basis.state_identity_hashes[transition_index]
    to_state = markov_basis.state_identity_hashes[transition_index + 1]
    tid = _transition_identity(from_state, to_state, transition_index)
    payload = {
        "transition_receipt_id": transition_receipt_id,
        "matrix_version": READOUT_MATRIX_VERSION,
        "markov_basis_receipt_hash": markov_basis.receipt_hash,
        "transition_rule": TRANSITION_RULE,
        "transition_index": transition_index,
        "from_state_hash": from_state,
        "to_state_hash": to_state,
        "transition_identity_hash": tid,
        "transition_valid": tid == markov_basis.transition_identity_hashes[transition_index],
        "transition_reason": "EXPLICIT_ADJACENT_TRANSITION" if tid == markov_basis.transition_identity_hashes[transition_index] else "STATE_HASH_MISMATCH",
    }
    return ReadoutTransitionReceipt(**payload, receipt_hash=sha256_hex(payload))


def validate_readout_transition_receipt(receipt: ReadoutTransitionReceipt, markov_basis: MarkovBasisReceipt) -> None:
    expected = build_readout_transition_receipt(receipt.transition_receipt_id, markov_basis, receipt.transition_index)
    if receipt.to_dict() != expected.to_dict():
        raise ValueError("INVALID_INPUT")


_FORBIDDEN_ATTR_NAMES = (
    "apply",
    "execute",
    "run",
    "dispatch",
    "route",
    "traverse",
    "pathfind",
    "resolve",
    "project",
    "search",
    "filter",
    "shift",
    "sample",
    "random",
    "replay",
    "drift",
)


def _assert_no_v153_8_forbidden_scope() -> None:
    for _klass in (ReadoutCombinationMatrix, ReadoutMatrixReceipt, MarkovBasisReceipt, ReadoutTransitionReceipt):
        for _name in _FORBIDDEN_ATTR_NAMES:
            if hasattr(_klass, _name):
                raise RuntimeError("INVALID_STATE")


_assert_no_v153_8_forbidden_scope()
