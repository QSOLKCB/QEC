from __future__ import annotations

import ast
from pathlib import Path

import pytest

from qec.analysis.memory_compression_context_ledger import (
    ContextLedger,
    ContextLedgerEntry,
    append_context_ledger_entry,
    build_canonical_memory_snapshot,
    build_context_compaction_plan,
    compute_context_compression_report,
    deduplicate_context_items,
    empty_context_ledger,
    minimize_context_ledger,
    normalize_context_items,
    run_memory_compression_context_ledger,
    validate_context_ledger,
)


def _sample_items() -> list[dict[str, object]]:
    return [
        {
            "item_id": "a1",
            "category": "fact",
            "content": "alpha",
            "priority": 3,
            "source_ref": "doc-A",
            "tags": ["z", "a", "a"],
            "bounded": True,
        },
        {
            "item_id": "b2",
            "category": "fact",
            "content": "alpha",
            "priority": 1,
            "source_ref": "doc-A",
            "tags": ["a", "z"],
            "bounded": True,
        },
        {
            "item_id": "c3",
            "category": "todo",
            "content": "beta",
            "priority": 2,
            "source_ref": "doc-B",
            "tags": ["m"],
            "bounded": False,
        },
    ]


def test_normalize_context_items_is_deterministic() -> None:
    n1 = normalize_context_items(_sample_items())
    n2 = normalize_context_items(_sample_items())
    assert n1 == n2


def test_normalize_rejects_empty_content() -> None:
    with pytest.raises(ValueError, match="content"):
        normalize_context_items(
            [
                {
                    "item_id": "x",
                    "category": "fact",
                    "content": "   ",
                    "priority": 1,
                    "source_ref": "s",
                    "tags": [],
                    "bounded": True,
                }
            ]
        )


def test_normalize_rejects_duplicate_item_id() -> None:
    items = _sample_items()
    items[1]["item_id"] = "a1"
    with pytest.raises(ValueError, match="duplicate item_id"):
        normalize_context_items(items)


def test_deduplicate_context_items_is_deterministic() -> None:
    normalized = normalize_context_items(_sample_items())
    r1 = deduplicate_context_items(normalized)
    r2 = deduplicate_context_items(normalized)
    assert r1 == r2


def test_deduplicate_retains_highest_priority_then_lexicographic_id() -> None:
    items = [
        {
            "item_id": "z2",
            "category": "fact",
            "content": "same",
            "priority": 5,
            "source_ref": "s",
            "tags": ["t"],
            "bounded": True,
        },
        {
            "item_id": "a1",
            "category": "fact",
            "content": "same",
            "priority": 5,
            "source_ref": "s",
            "tags": ["t"],
            "bounded": True,
        },
    ]
    retained, compacted = deduplicate_context_items(normalize_context_items(items))
    assert retained[0].item_id == "a1"
    assert compacted == ("z2",)


def test_snapshot_hash_is_stable() -> None:
    normalized = normalize_context_items(_sample_items())
    retained, compacted = deduplicate_context_items(normalized)
    s1 = build_canonical_memory_snapshot(retained, compacted, len(normalized))
    s2 = build_canonical_memory_snapshot(retained, compacted, len(normalized))
    assert s1.snapshot_hash == s2.snapshot_hash


def test_compression_report_is_bounded() -> None:
    report = compute_context_compression_report(original_count=3, retained_count=2, compacted_count=1)
    assert 0.0 <= report.compression_ratio <= 1.0
    assert report.compression_ratio == pytest.approx(1 / 3, rel=0.0, abs=1e-12)


def test_compression_report_rejects_impossible_counts() -> None:
    with pytest.raises(ValueError):
        compute_context_compression_report(original_count=1, retained_count=1, compacted_count=1)


def test_context_ledger_chain_is_stable() -> None:
    ledger = empty_context_ledger()
    ledger = append_context_ledger_entry(ledger, snapshot_hash="s1", retained_count=2, compression_ratio=0.5)
    ledger = append_context_ledger_entry(ledger, snapshot_hash="s2", retained_count=1, compression_ratio=0.0)
    assert validate_context_ledger(ledger)


def test_context_ledger_detects_corruption() -> None:
    ledger = empty_context_ledger()
    ledger = append_context_ledger_entry(ledger, snapshot_hash="s1", retained_count=2, compression_ratio=0.5)
    bad = ContextLedger(
        entries=(
            ContextLedgerEntry(
                sequence_id=0,
                snapshot_hash="tampered",
                parent_hash=ledger.entries[0].parent_hash,
                retained_count=ledger.entries[0].retained_count,
                compression_ratio=ledger.entries[0].compression_ratio,
                entry_hash=ledger.entries[0].entry_hash,
            ),
        ),
        head_hash=ledger.head_hash,
        chain_valid=True,
    )
    assert not validate_context_ledger(bad)


def test_append_rejects_malformed_context_ledger() -> None:
    malformed = ContextLedger(entries=(), head_hash="bad", chain_valid=True)
    with pytest.raises(ValueError, match="malformed"):
        append_context_ledger_entry(malformed, snapshot_hash="s", retained_count=1, compression_ratio=0.0)


def test_minimize_context_ledger_is_deterministic() -> None:
    ledger = empty_context_ledger()
    for i in range(6):
        ledger = append_context_ledger_entry(ledger, snapshot_hash=f"s{i}", retained_count=1, compression_ratio=0.2)
    m1 = minimize_context_ledger(ledger, threshold=3)
    m2 = minimize_context_ledger(ledger, threshold=3)
    assert m1 == m2


def test_minimize_preserves_replay_safe_head_semantics_or_documented_rebuild_rule() -> None:
    ledger = empty_context_ledger()
    for i in range(5):
        ledger = append_context_ledger_entry(ledger, snapshot_hash=f"s{i}", retained_count=1, compression_ratio=0.1)
    minimized = minimize_context_ledger(ledger, threshold=3)
    assert validate_context_ledger(minimized)
    assert len(minimized.entries) == 3


def test_same_input_same_bytes() -> None:
    result1 = run_memory_compression_context_ledger(_sample_items())
    result2 = run_memory_compression_context_ledger(_sample_items())
    bytes1 = tuple(x.to_canonical_json() for x in result1[1:])
    bytes2 = tuple(x.to_canonical_json() for x in result2[1:])
    assert tuple(i.to_canonical_json() for i in result1[0]) == tuple(i.to_canonical_json() for i in result2[0])
    assert bytes1 == bytes2


def test_no_decoder_imports() -> None:
    source = Path("src/qec/analysis/memory_compression_context_ledger.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
            continue
        if isinstance(node, ast.ImportFrom) and node.module is not None:
            imports.append(node.module)
            imports.extend(f"{node.module}.{alias.name}" for alias in node.names)
    assert not any(name.startswith("qec.decoder") for name in imports)


def test_insertion_order_independence() -> None:
    items = _sample_items()
    reversed_items = list(reversed(items))
    r1 = run_memory_compression_context_ledger(items)
    r2 = run_memory_compression_context_ledger(reversed_items)
    assert tuple(item.item_id for item in r1[0]) == tuple(item.item_id for item in r2[0])
    assert r1[1].snapshot_hash == r2[1].snapshot_hash


def test_tag_sorting_and_dedup() -> None:
    normalized = normalize_context_items(_sample_items())
    assert normalized[0].tags == ("a", "z")


def test_zero_item_context_is_valid() -> None:
    retained, snapshot, report, transition, updated, minimized = run_memory_compression_context_ledger([])
    assert retained == ()
    assert snapshot.retention_ratio == 0.0
    assert report.compression_ratio == 0.0
    assert transition.total_items == 0
    assert validate_context_ledger(updated)
    assert validate_context_ledger(minimized)


def test_minimization_threshold_noop() -> None:
    ledger = empty_context_ledger()
    for i in range(2):
        ledger = append_context_ledger_entry(ledger, snapshot_hash=f"s{i}", retained_count=1, compression_ratio=0.0)
    minimized = minimize_context_ledger(ledger, threshold=2)
    assert minimized == ledger


def test_snapshot_hash_independent_of_input_order() -> None:
    left = run_memory_compression_context_ledger(_sample_items())[1]
    right = run_memory_compression_context_ledger(list(reversed(_sample_items())))[1]
    assert left.snapshot_hash == right.snapshot_hash


def test_build_context_compaction_plan_deterministic() -> None:
    normalized = normalize_context_items(_sample_items())
    retained, compacted = deduplicate_context_items(normalized)
    plan1 = build_context_compaction_plan(retained, compacted)
    plan2 = build_context_compaction_plan(retained, compacted)
    assert plan1 == plan2
    assert plan1.plan_hash == plan2.plan_hash


def test_build_context_compaction_plan_ordering() -> None:
    normalized = normalize_context_items(_sample_items())
    retained, compacted = deduplicate_context_items(normalized)
    plan = build_context_compaction_plan(retained, compacted)
    assert plan.retained_ids == tuple(sorted(plan.retained_ids))
    assert plan.removed_ids == tuple(sorted(plan.removed_ids))
    assert plan.canonical_order == tuple(sorted(plan.retained_ids + plan.removed_ids))


def test_build_context_compaction_plan_hash_is_stable_across_input_order() -> None:
    normalized = normalize_context_items(_sample_items())
    retained, compacted = deduplicate_context_items(normalized)
    plan_fwd = build_context_compaction_plan(retained, compacted)
    plan_rev = build_context_compaction_plan(tuple(reversed(retained)), compacted)
    assert plan_fwd.plan_hash == plan_rev.plan_hash


def test_append_context_ledger_entry_rejects_empty_snapshot_hash() -> None:
    ledger = empty_context_ledger()
    with pytest.raises(ValueError, match="snapshot_hash"):
        append_context_ledger_entry(ledger, snapshot_hash="   ", retained_count=1, compression_ratio=0.0)


def test_append_context_ledger_entry_rejects_non_string_snapshot_hash() -> None:
    ledger = empty_context_ledger()
    with pytest.raises(ValueError, match="snapshot_hash"):
        append_context_ledger_entry(ledger, snapshot_hash=123, retained_count=1, compression_ratio=0.0)  # type: ignore[arg-type]
