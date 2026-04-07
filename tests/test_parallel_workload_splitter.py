from __future__ import annotations

import dataclasses

import pytest

from qec.analysis.parallel_workload_splitter import (
    PARALLEL_WORKLOAD_SPLITTER_SCHEMA_VERSION,
    MergeReceipt,
    WorkShard,
    build_merge_receipt,
    compile_split_report,
    merge_shards,
    normalize_workload_descriptor,
    split_workload,
    validate_workload_descriptor,
)


def _base_raw(strategy: str = "fixed_tiles") -> dict[str, object]:
    max_shard_size = 4
    if strategy == "identity_pass":
        max_shard_size = 12
    return {
        "workload_id": "workload-A",
        "payload": [
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
        ],
        "split_strategy": strategy,
        "target_shard_count": 4,
        "max_shard_size": max_shard_size,
        "epoch_id": "epoch-01",
        "schema_version": PARALLEL_WORKLOAD_SPLITTER_SCHEMA_VERSION,
    }


def test_frozen_models() -> None:
    report = compile_split_report(_base_raw("identity_pass"))
    with pytest.raises(dataclasses.FrozenInstanceError):
        report.descriptor.workload_id = "other"  # type: ignore[misc]


def test_fixed_tile_splitting_determinism() -> None:
    descriptor = normalize_workload_descriptor(_base_raw("fixed_tiles"))
    shards_a = split_workload(descriptor)
    shards_b = split_workload(descriptor)
    assert tuple(s.to_canonical_json() for s in shards_a) == tuple(s.to_canonical_json() for s in shards_b)


def test_row_chunk_splitting() -> None:
    descriptor = normalize_workload_descriptor(_base_raw("row_chunks"))
    shards = split_workload(descriptor)
    assert len(shards) == 3
    assert shards[0].shard_payload["rows"] == [[1, 2, 3, 4]]


def test_column_chunk_splitting() -> None:
    descriptor = normalize_workload_descriptor(_base_raw("column_chunks"))
    shards = split_workload(descriptor)
    assert len(shards) == 4
    assert shards[0].shard_payload["rows"] == [[1], [5], [9]]


def test_scanline_split_determinism() -> None:
    descriptor = normalize_workload_descriptor(_base_raw("scanline_split"))
    shards_a = split_workload(descriptor)
    shards_b = split_workload(descriptor)
    assert [s.shard_payload["cells"] for s in shards_a] == [s.shard_payload["cells"] for s in shards_b]


def test_merge_determinism() -> None:
    descriptor = normalize_workload_descriptor(_base_raw("fixed_tiles"))
    shards = split_workload(descriptor)
    merged_a = merge_shards(shards)
    merged_b = merge_shards(tuple(reversed(shards)))
    assert merged_a == merged_b


def test_stable_shard_hashes() -> None:
    descriptor = normalize_workload_descriptor(_base_raw("row_chunks"))
    shards = split_workload(descriptor)
    replay = split_workload(descriptor)
    assert [s.stable_hash for s in shards] == [s.stable_hash for s in replay]


def test_stable_merge_receipts() -> None:
    descriptor = normalize_workload_descriptor(_base_raw("scanline_split"))
    shards = split_workload(descriptor)
    merged = merge_shards(shards)
    receipt_a = build_merge_receipt(shards, merged)
    receipt_b = build_merge_receipt(tuple(reversed(shards)), merged)
    assert isinstance(receipt_a, MergeReceipt)
    assert receipt_a.to_canonical_json() == receipt_b.to_canonical_json()


def test_repeated_run_byte_identity() -> None:
    report_a = compile_split_report(_base_raw("fixed_tiles"))
    report_b = compile_split_report(_base_raw("fixed_tiles"))
    assert report_a.to_canonical_bytes() == report_b.to_canonical_bytes()


def test_invalid_split_rejection() -> None:
    raw = _base_raw("not_supported")
    descriptor = normalize_workload_descriptor(raw)
    with pytest.raises(ValueError, match="unsupported split_strategy"):
        validate_workload_descriptor(descriptor)


def test_malformed_payload_rejection() -> None:
    raw = _base_raw("row_chunks")
    raw["payload"] = [object()]
    with pytest.raises(ValueError, match="malformed payload"):
        normalize_workload_descriptor(raw)


def test_schema_rejection() -> None:
    raw = _base_raw("row_chunks")
    raw["schema_version"] = "v0"
    with pytest.raises(ValueError, match="unsupported schema version"):
        compile_split_report(raw)


def test_shard_ordering_stability() -> None:
    report = compile_split_report(_base_raw("column_chunks"))
    reordered = tuple(sorted(report.shards, key=lambda shard: shard.shard_id, reverse=True))
    merged_reordered = merge_shards(reordered)
    merged_original = merge_shards(report.shards)
    assert merged_reordered == merged_original


def test_callable_leakage_rejection() -> None:
    raw = _base_raw("identity_pass")
    raw["payload"] = [[lambda: 1]]
    with pytest.raises(ValueError, match="callable leakage"):
        normalize_workload_descriptor(raw)


def test_invalid_shard_count_rejection() -> None:
    raw = _base_raw("identity_pass")
    raw["target_shard_count"] = 0
    with pytest.raises(ValueError, match="target_shard_count must be a positive integer"):
        normalize_workload_descriptor(raw)


def test_merge_requires_single_workload_id() -> None:
    report = compile_split_report(_base_raw("identity_pass"))
    shard = report.shards[0]
    rogue = WorkShard(
        shard_id="x" * 64,
        workload_id="other",
        shard_index=1,
        shard_payload=shard.shard_payload,
        epoch_id=shard.epoch_id,
        stable_hash="y" * 64,
    )
    with pytest.raises(ValueError, match="shards must share workload_id"):
        merge_shards((shard, rogue))
