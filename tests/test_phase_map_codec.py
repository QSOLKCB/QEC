# SPDX-License-Identifier: MIT
"""Tests for phase map export codec — v133.7.0."""

from __future__ import annotations

import json

import pytest

from qec.sims.phase_map_codec import (
    PhaseMapExportBundle,
    PhaseMapExportMetadata,
    compute_phase_map_hash,
    export_phase_map_to_json,
    load_phase_map_from_json,
    with_computed_phase_hash,
)
from qec.sims.phase_map_generator import PhaseCell, PhaseMap


def _make_phase_map() -> PhaseMap:
    cells = (
        PhaseCell(
            decay=0.01,
            coupling_profile=(0.1, 0.2, 0.3),
            regime_label="stable",
            divergence_score=0.05,
        ),
        PhaseCell(
            decay=0.01,
            coupling_profile=(0.4, 0.5, 0.6),
            regime_label="critical",
            divergence_score=0.50,
        ),
        PhaseCell(
            decay=0.02,
            coupling_profile=(0.1, 0.2, 0.3),
            regime_label="critical",
            divergence_score=0.45,
        ),
        PhaseCell(
            decay=0.02,
            coupling_profile=(0.4, 0.5, 0.6),
            regime_label="divergent",
            divergence_score=0.95,
        ),
    )
    return PhaseMap(
        cells=cells,
        num_rows=2,
        num_cols=2,
        stable_count=1,
        critical_count=2,
        divergent_count=1,
        max_divergence=0.95,
    )


def _make_bundle() -> PhaseMapExportBundle:
    return PhaseMapExportBundle(
        phase_map=_make_phase_map(),
        metadata=PhaseMapExportMetadata(
            schema_version="1.0.0",
            created_by_release="v133.7.0",
            trace_hash="0" * 64,
        ),
        ascii_render="S C\nC D",
    )


class TestFrozenDataclasses:
    def test_metadata_is_frozen(self):
        md = PhaseMapExportMetadata(
            schema_version="1.0.0",
            created_by_release="v133.7.0",
            trace_hash="abc",
        )
        with pytest.raises(AttributeError):
            md.schema_version = "2.0.0"  # type: ignore[misc]

    def test_bundle_is_frozen(self):
        bundle = _make_bundle()
        with pytest.raises(AttributeError):
            bundle.ascii_render = "X"  # type: ignore[misc]


class TestCanonicalRoundTrip:
    def test_export_and_load_roundtrip(self):
        bundle = _make_bundle()
        text = export_phase_map_to_json(bundle)
        restored = load_phase_map_from_json(text)

        assert restored.phase_map.num_rows == bundle.phase_map.num_rows
        assert restored.phase_map.num_cols == bundle.phase_map.num_cols
        assert restored.phase_map.stable_count == bundle.phase_map.stable_count
        assert restored.phase_map.critical_count == bundle.phase_map.critical_count
        assert restored.phase_map.divergent_count == bundle.phase_map.divergent_count
        assert restored.phase_map.max_divergence == bundle.phase_map.max_divergence
        assert len(restored.phase_map.cells) == len(bundle.phase_map.cells)
        for orig, rest in zip(bundle.phase_map.cells, restored.phase_map.cells):
            assert orig.decay == rest.decay
            assert orig.coupling_profile == rest.coupling_profile
            assert orig.regime_label == rest.regime_label
            assert orig.divergence_score == rest.divergence_score

        assert restored.metadata.schema_version == bundle.metadata.schema_version
        assert restored.metadata.created_by_release == bundle.metadata.created_by_release
        assert restored.metadata.trace_hash == bundle.metadata.trace_hash
        assert restored.ascii_render == bundle.ascii_render

    def test_json_is_canonical(self):
        bundle = _make_bundle()
        text = export_phase_map_to_json(bundle)
        d = json.loads(text)
        # Keys must be sorted at top level
        assert list(d.keys()) == sorted(d.keys())
        # No whitespace padding (compact separators)
        assert "  " not in text
        assert ": " not in text


class TestHashIdempotency:
    def test_same_hash_twice(self):
        bundle = _make_bundle()
        h1 = compute_phase_map_hash(bundle)
        h2 = compute_phase_map_hash(bundle)
        assert h1 == h2

    def test_hash_is_hex_sha256(self):
        bundle = _make_bundle()
        h = compute_phase_map_hash(bundle)
        assert len(h) == 64
        int(h, 16)  # must be valid hex

    def test_finalized_hash_is_idempotent(self):
        bundle = _make_bundle()
        finalized1 = with_computed_phase_hash(bundle)
        finalized2 = with_computed_phase_hash(finalized1)
        assert finalized1.metadata.trace_hash == finalized2.metadata.trace_hash

    def test_hash_ignores_existing_trace_hash(self):
        bundle = _make_bundle()
        h_placeholder = compute_phase_map_hash(bundle)

        # Bundle with a different trace_hash should produce same hash
        bundle_with_hash = PhaseMapExportBundle(
            phase_map=bundle.phase_map,
            metadata=PhaseMapExportMetadata(
                schema_version=bundle.metadata.schema_version,
                created_by_release=bundle.metadata.created_by_release,
                trace_hash="abcd1234" * 8,
            ),
            ascii_render=bundle.ascii_render,
        )
        h_different = compute_phase_map_hash(bundle_with_hash)
        assert h_placeholder == h_different


class TestRepeatedExportEquality:
    def test_repeated_export_produces_identical_json(self):
        bundle = _make_bundle()
        text1 = export_phase_map_to_json(bundle)
        text2 = export_phase_map_to_json(bundle)
        assert text1 == text2

    def test_finalized_export_stable(self):
        bundle = _make_bundle()
        finalized = with_computed_phase_hash(bundle)
        text1 = export_phase_map_to_json(finalized)
        text2 = export_phase_map_to_json(finalized)
        assert text1 == text2


class TestMalformedJsonRejection:
    def test_invalid_json_raises(self):
        with pytest.raises(ValueError, match="malformed JSON"):
            load_phase_map_from_json("{not valid json")

    def test_empty_string_raises(self):
        with pytest.raises(ValueError, match="malformed JSON"):
            load_phase_map_from_json("")

    def test_non_object_raises(self):
        with pytest.raises((ValueError, TypeError)):
            load_phase_map_from_json('"just a string"')


class TestMissingKeyRejection:
    def test_missing_metadata_key(self):
        bundle = _make_bundle()
        text = export_phase_map_to_json(bundle)
        d = json.loads(text)
        del d["metadata"]
        with pytest.raises(ValueError, match="missing key.*metadata.*root"):
            load_phase_map_from_json(json.dumps(d))

    def test_missing_phase_map_key(self):
        bundle = _make_bundle()
        text = export_phase_map_to_json(bundle)
        d = json.loads(text)
        del d["phase_map"]
        with pytest.raises(ValueError, match="missing key.*phase_map.*root"):
            load_phase_map_from_json(json.dumps(d))

    def test_missing_ascii_render_key(self):
        bundle = _make_bundle()
        text = export_phase_map_to_json(bundle)
        d = json.loads(text)
        del d["ascii_render"]
        with pytest.raises(ValueError, match="missing key.*ascii_render.*root"):
            load_phase_map_from_json(json.dumps(d))

    def test_missing_metadata_subkey(self):
        bundle = _make_bundle()
        text = export_phase_map_to_json(bundle)
        d = json.loads(text)
        del d["metadata"]["schema_version"]
        with pytest.raises(ValueError, match="missing key.*schema_version.*metadata"):
            load_phase_map_from_json(json.dumps(d))

    def test_missing_phase_map_subkey(self):
        bundle = _make_bundle()
        text = export_phase_map_to_json(bundle)
        d = json.loads(text)
        del d["phase_map"]["cells"]
        with pytest.raises(ValueError, match="missing key.*cells.*phase_map"):
            load_phase_map_from_json(json.dumps(d))

    def test_missing_cell_subkey(self):
        bundle = _make_bundle()
        text = export_phase_map_to_json(bundle)
        d = json.loads(text)
        del d["phase_map"]["cells"][0]["decay"]
        with pytest.raises(ValueError, match=r"missing key.*decay.*cells\[0\]"):
            load_phase_map_from_json(json.dumps(d))

    def test_schema_version_mismatch_rejected(self):
        bundle = _make_bundle()
        text = export_phase_map_to_json(bundle)
        d = json.loads(text)
        d["metadata"]["schema_version"] = "99.0.0"
        with pytest.raises(ValueError, match="unsupported schema_version"):
            load_phase_map_from_json(json.dumps(d))


class TestReplayDeterminism:
    def test_full_replay_cycle(self):
        """Build, export, load, re-export — must be byte-identical."""
        bundle = _make_bundle()
        finalized = with_computed_phase_hash(bundle)

        json1 = export_phase_map_to_json(finalized)
        restored = load_phase_map_from_json(json1)
        json2 = export_phase_map_to_json(restored)

        assert json1 == json2

    def test_hash_survives_roundtrip(self):
        bundle = _make_bundle()
        finalized = with_computed_phase_hash(bundle)
        original_hash = finalized.metadata.trace_hash

        json_text = export_phase_map_to_json(finalized)
        restored = load_phase_map_from_json(json_text)

        assert restored.metadata.trace_hash == original_hash
        recomputed = compute_phase_map_hash(restored)
        assert recomputed == original_hash
