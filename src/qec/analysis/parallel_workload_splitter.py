from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

PARALLEL_WORKLOAD_SPLITTER_SCHEMA_VERSION = "v137.11.3"

_ALLOWED_SPLIT_STRATEGIES = frozenset(
    {"fixed_tiles", "row_chunks", "column_chunks", "scanline_split", "identity_pass"}
)


def _canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def _sha256_hex_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _normalize_token(value: Any, *, name: str) -> str:
    if value is None or callable(value):
        raise ValueError(f"{name} must be non-empty")
    token = str(value).strip()
    if not token:
        raise ValueError(f"{name} must be non-empty")
    return token


def _normalize_positive_int(value: Any, *, name: str) -> int:
    if callable(value) or isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer")
    if isinstance(value, int) and value > 0:
        return value
    raise ValueError(f"{name} must be a positive integer")


def _normalize_scalar(value: Any) -> str:
    if callable(value):
        raise ValueError("callable leakage")
    if isinstance(value, (str, int, float, bool)) or value is None:
        return _canonical_json(value)
    raise ValueError("malformed payload")


def _normalize_payload(payload: Any) -> tuple[tuple[str, ...], ...]:
    if callable(payload):
        raise ValueError("callable leakage")
    if isinstance(payload, (str, bytes)) or not isinstance(payload, Sequence):
        raise ValueError("malformed payload")
    rows: list[tuple[str, ...]] = []
    expected_width: int | None = None
    for row in list(payload):
        if callable(row):
            raise ValueError("callable leakage")
        if isinstance(row, (str, bytes)) or not isinstance(row, Sequence):
            raise ValueError("malformed payload")
        normalized_row = tuple(_normalize_scalar(value) for value in list(row))
        if not normalized_row:
            raise ValueError("malformed payload")
        if expected_width is None:
            expected_width = len(normalized_row)
        elif len(normalized_row) != expected_width:
            raise ValueError("malformed payload")
        rows.append(normalized_row)
    if not rows:
        raise ValueError("malformed payload")
    return tuple(rows)


def _restore_payload(payload: tuple[tuple[str, ...], ...]) -> list[list[Any]]:
    return [[json.loads(value) for value in row] for row in payload]


def _shard_hash_payload(payload: Mapping[str, Any]) -> str:
    return _sha256_hex_bytes(_canonical_json(payload).encode("utf-8"))


@dataclass(frozen=True)
class WorkloadDescriptor:
    workload_id: str
    payload: tuple[tuple[str, ...], ...]
    split_strategy: str
    target_shard_count: int
    max_shard_size: int
    epoch_id: str
    schema_version: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "workload_id": self.workload_id,
            "payload": _restore_payload(self.payload),
            "split_strategy": self.split_strategy,
            "target_shard_count": self.target_shard_count,
            "max_shard_size": self.max_shard_size,
            "epoch_id": self.epoch_id,
            "schema_version": self.schema_version,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")


@dataclass(frozen=True)
class WorkShard:
    shard_id: str
    workload_id: str
    shard_index: int
    shard_payload: Mapping[str, Any]
    epoch_id: str
    stable_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "shard_id": self.shard_id,
            "workload_id": self.workload_id,
            "shard_index": self.shard_index,
            "shard_payload": self.shard_payload,
            "epoch_id": self.epoch_id,
            "stable_hash": self.stable_hash,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")


@dataclass(frozen=True)
class MergeReceipt:
    receipt_hash: str
    workload_id: str
    shard_count: int
    merge_order: tuple[str, ...]
    output_hash: str
    validation_passed: bool
    schema_version: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "receipt_hash": self.receipt_hash,
            "workload_id": self.workload_id,
            "shard_count": self.shard_count,
            "merge_order": list(self.merge_order),
            "output_hash": self.output_hash,
            "validation_passed": self.validation_passed,
            "schema_version": self.schema_version,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")


@dataclass(frozen=True)
class SplitReport:
    descriptor: WorkloadDescriptor
    shards: tuple[WorkShard, ...]
    receipt: MergeReceipt
    stable_hash: str
    schema_version: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "descriptor": self.descriptor.to_dict(),
            "shards": [shard.to_dict() for shard in self.shards],
            "receipt": self.receipt.to_dict(),
            "stable_hash": self.stable_hash,
            "schema_version": self.schema_version,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")


def normalize_workload_descriptor(raw_input: Mapping[str, Any]) -> WorkloadDescriptor:
    if not isinstance(raw_input, Mapping):
        raise ValueError("raw_input must be a mapping")
    schema_version = _normalize_token(raw_input.get("schema_version", PARALLEL_WORKLOAD_SPLITTER_SCHEMA_VERSION), name="schema_version")
    return WorkloadDescriptor(
        workload_id=_normalize_token(raw_input.get("workload_id"), name="workload_id"),
        payload=_normalize_payload(raw_input.get("payload")),
        split_strategy=_normalize_token(raw_input.get("split_strategy"), name="split_strategy"),
        target_shard_count=_normalize_positive_int(raw_input.get("target_shard_count"), name="target_shard_count"),
        max_shard_size=_normalize_positive_int(raw_input.get("max_shard_size"), name="max_shard_size"),
        epoch_id=_normalize_token(raw_input.get("epoch_id"), name="epoch_id"),
        schema_version=schema_version,
    )


def validate_workload_descriptor(descriptor: WorkloadDescriptor) -> None:
    if descriptor.schema_version != PARALLEL_WORKLOAD_SPLITTER_SCHEMA_VERSION:
        raise ValueError("unsupported schema version")
    if not descriptor.workload_id:
        raise ValueError("empty workload_id")
    if descriptor.split_strategy not in _ALLOWED_SPLIT_STRATEGIES:
        raise ValueError("unsupported split_strategy")
    if descriptor.target_shard_count <= 0:
        raise ValueError("invalid shard count")
    if descriptor.max_shard_size <= 0:
        raise ValueError("zero / negative max_shard_size")


def _build_shard(descriptor: WorkloadDescriptor, shard_index: int, shard_payload: Mapping[str, Any]) -> WorkShard:
    identity_payload = {
        "workload_id": descriptor.workload_id,
        "shard_index": shard_index,
        "shard_payload": shard_payload,
        "epoch_id": descriptor.epoch_id,
    }
    shard_id = _shard_hash_payload(identity_payload)
    stable_hash = _shard_hash_payload({"shard_id": shard_id, **identity_payload})
    return WorkShard(
        shard_id=shard_id,
        workload_id=descriptor.workload_id,
        shard_index=shard_index,
        shard_payload=shard_payload,
        epoch_id=descriptor.epoch_id,
        stable_hash=stable_hash,
    )


def split_workload(descriptor: WorkloadDescriptor) -> tuple[WorkShard, ...]:
    validate_workload_descriptor(descriptor)
    rows = len(descriptor.payload)
    cols = len(descriptor.payload[0])
    total_cells = rows * cols
    strategy = descriptor.split_strategy

    payloads: list[dict[str, Any]] = []

    if strategy == "identity_pass":
        if total_cells > descriptor.max_shard_size:
            raise ValueError("payload exceeds max_shard_size")
        payloads.append({"strategy": strategy, "rows": _restore_payload(descriptor.payload)})
    elif strategy == "row_chunks":
        if cols > descriptor.max_shard_size:
            raise ValueError("payload exceeds max_shard_size")
        max_rows_by_size = descriptor.max_shard_size // cols
        rows_per_shard = min(math.ceil(rows / descriptor.target_shard_count), max_rows_by_size)
        for start in range(0, rows, rows_per_shard):
            stop = min(rows, start + rows_per_shard)
            payloads.append(
                {
                    "strategy": strategy,
                    "row_start": start,
                    "row_stop": stop,
                    "rows": _restore_payload(descriptor.payload[start:stop]),
                }
            )
    elif strategy == "column_chunks":
        if rows > descriptor.max_shard_size:
            raise ValueError("payload exceeds max_shard_size")
        max_cols_by_size = descriptor.max_shard_size // rows
        cols_per_shard = min(math.ceil(cols / descriptor.target_shard_count), max_cols_by_size)
        for start in range(0, cols, cols_per_shard):
            stop = min(cols, start + cols_per_shard)
            payloads.append(
                {
                    "strategy": strategy,
                    "col_start": start,
                    "col_stop": stop,
                    "rows": _restore_payload(tuple(row[start:stop] for row in descriptor.payload)),
                }
            )
    elif strategy == "scanline_split":
        flat = [value for row in descriptor.payload for value in row]
        cells_per_shard = min(math.ceil(total_cells / descriptor.target_shard_count), descriptor.max_shard_size)
        for start in range(0, total_cells, cells_per_shard):
            stop = min(total_cells, start + cells_per_shard)
            payloads.append(
                {
                    "strategy": strategy,
                    "scan_start": start,
                    "scan_stop": stop,
                    "cells": [json.loads(value) for value in flat[start:stop]],
                    "shape": [rows, cols],
                }
            )
    else:  # fixed_tiles
        row_splits = max(1, min(rows, int(math.sqrt(descriptor.target_shard_count))))
        col_splits = max(1, min(cols, math.ceil(descriptor.target_shard_count / row_splits)))
        tile_rows = max(1, math.ceil(rows / row_splits))
        tile_cols = max(1, math.ceil(cols / col_splits))
        while tile_rows * tile_cols > descriptor.max_shard_size:
            if tile_rows >= tile_cols and tile_rows > 1:
                tile_rows -= 1
            elif tile_cols > 1:
                tile_cols -= 1
            else:
                raise ValueError("payload exceeds max_shard_size")
        for row_start in range(0, rows, tile_rows):
            row_stop = min(rows, row_start + tile_rows)
            for col_start in range(0, cols, tile_cols):
                col_stop = min(cols, col_start + tile_cols)
                tile = tuple(row[col_start:col_stop] for row in descriptor.payload[row_start:row_stop])
                payloads.append(
                    {
                        "strategy": strategy,
                        "row_start": row_start,
                        "row_stop": row_stop,
                        "col_start": col_start,
                        "col_stop": col_stop,
                        "rows": _restore_payload(tile),
                    }
                )

    shards = tuple(_build_shard(descriptor, index, payload) for index, payload in enumerate(payloads))
    if not shards:
        raise ValueError("invalid shard count")
    return shards


def merge_shards(shards: Sequence[WorkShard]) -> bytes:
    if not shards:
        raise ValueError("invalid shard count")
    ordered = tuple(sorted(shards, key=lambda shard: (shard.shard_index, shard.shard_id)))
    workload_ids = {shard.workload_id for shard in ordered}
    if len(workload_ids) != 1:
        raise ValueError("shards must share workload_id")

    strategy = str(ordered[0].shard_payload.get("strategy", ""))
    if strategy not in _ALLOWED_SPLIT_STRATEGIES:
        raise ValueError("unsupported split_strategy")
    if any(str(shard.shard_payload.get("strategy", "")) != strategy for shard in ordered):
        raise ValueError("shards must share strategy")

    if strategy == "identity_pass":
        if len(ordered) != 1:
            raise ValueError("identity_pass requires exactly 1 shard")
        merged_rows = ordered[0].shard_payload["rows"]
    elif strategy == "row_chunks":
        merged_rows = []
        for shard in ordered:
            merged_rows.extend(shard.shard_payload["rows"])
    elif strategy == "column_chunks":
        base_rows = len(ordered[0].shard_payload["rows"])
        merged_rows = [[] for _ in range(base_rows)]
        for shard in ordered:
            for row_index, row in enumerate(shard.shard_payload["rows"]):
                merged_rows[row_index].extend(row)
    elif strategy == "scanline_split":
        shape = tuple(int(x) for x in ordered[0].shard_payload["shape"])
        rows, cols = shape
        cells: list[Any] = []
        for shard in ordered:
            cells.extend(shard.shard_payload["cells"])
        merged_rows = [cells[offset : offset + cols] for offset in range(0, rows * cols, cols)]
    else:
        max_row = max(int(shard.shard_payload["row_stop"]) for shard in ordered)
        max_col = max(int(shard.shard_payload["col_stop"]) for shard in ordered)
        merged_rows = [[None for _ in range(max_col)] for _ in range(max_row)]
        for shard in ordered:
            row_start = int(shard.shard_payload["row_start"])
            col_start = int(shard.shard_payload["col_start"])
            tile_rows = shard.shard_payload["rows"]
            for row_offset, row in enumerate(tile_rows):
                for col_offset, value in enumerate(row):
                    merged_rows[row_start + row_offset][col_start + col_offset] = value

    output = {
        "workload_id": ordered[0].workload_id,
        "merge_order": [shard.shard_id for shard in ordered],
        "merged_rows": merged_rows,
    }
    return _canonical_json(output).encode("utf-8")


def build_merge_receipt(shards: Sequence[WorkShard], merged_output: bytes) -> MergeReceipt:
    if not shards:
        raise ValueError("invalid shard count")
    ordered = tuple(sorted(shards, key=lambda shard: (shard.shard_index, shard.shard_id)))
    workload_ids = {shard.workload_id for shard in ordered}
    if len(workload_ids) != 1:
        raise ValueError("shards must share workload_id")
    workload_id = ordered[0].workload_id
    payload = {
        "workload_id": workload_id,
        "shard_count": len(ordered),
        "merge_order": [shard.shard_id for shard in ordered],
        "output_hash": _sha256_hex_bytes(merged_output),
        "validation_passed": True,
        "schema_version": PARALLEL_WORKLOAD_SPLITTER_SCHEMA_VERSION,
    }
    receipt_hash = _sha256_hex_bytes(_canonical_json(payload).encode("utf-8"))
    return MergeReceipt(
        receipt_hash=receipt_hash,
        workload_id=workload_id,
        shard_count=len(ordered),
        merge_order=tuple(payload["merge_order"]),
        output_hash=payload["output_hash"],
        validation_passed=True,
        schema_version=PARALLEL_WORKLOAD_SPLITTER_SCHEMA_VERSION,
    )


def stable_split_report_hash(report: SplitReport) -> str:
    payload = {
        "descriptor": report.descriptor.to_dict(),
        "shards": [shard.to_dict() for shard in sorted(report.shards, key=lambda item: (item.shard_index, item.shard_id))],
        "receipt": report.receipt.to_dict(),
        "schema_version": report.schema_version,
    }
    return _sha256_hex_bytes(_canonical_json(payload).encode("utf-8"))


def compile_split_report(raw_input: Mapping[str, Any]) -> SplitReport:
    descriptor = normalize_workload_descriptor(raw_input)
    validate_workload_descriptor(descriptor)
    shards = split_workload(descriptor)
    merged_output = merge_shards(shards)
    receipt = build_merge_receipt(shards, merged_output)
    interim = SplitReport(
        descriptor=descriptor,
        shards=shards,
        receipt=receipt,
        stable_hash="",
        schema_version=PARALLEL_WORKLOAD_SPLITTER_SCHEMA_VERSION,
    )
    stable_hash = stable_split_report_hash(interim)
    return SplitReport(
        descriptor=descriptor,
        shards=shards,
        receipt=receipt,
        stable_hash=stable_hash,
        schema_version=PARALLEL_WORKLOAD_SPLITTER_SCHEMA_VERSION,
    )
