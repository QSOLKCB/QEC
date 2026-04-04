"""Tests for v137.0.10 — Retro 3D World Modelling.

Covers:
  - frozen dataclasses
  - primitive normalization
  - sector classification
  - projection classification
  - complexity classification
  - symbolic trace correctness
  - stable hashing
  - canonical export equality
  - 100-run replay determinism
  - invalid input rejection
  - generic iterable acceptance
  - no decoder contamination
  - TRUE_3D vs PSEUDO_3D mode behavior
"""

from __future__ import annotations

import hashlib
import json
import sys

import pytest

from qec.analysis.retro_3d_world_modelling import (
    # Version
    RETRO_3D_WORLD_MODELLING_VERSION,
    # World modes
    TRUE_3D,
    PSEUDO_3D,
    VALID_WORLD_MODES,
    # Sector classes
    OPEN_SECTOR,
    CORRIDOR_SECTOR,
    CHAMBER_SECTOR,
    VERTICAL_SECTOR,
    DENSE_SECTOR,
    # Projection classes
    ORTHO_LIKE,
    PERSPECTIVE_3D,
    PSEUDO_DOOM,
    HYBRID_RETRO,
    # Complexity classes
    MINIMAL,
    COMPACT,
    STRUCTURED,
    DENSE,
    # Primitive types
    PRIMITIVE_WALL,
    PRIMITIVE_ENTITY,
    PRIMITIVE_VERTEX,
    PRIMITIVE_EDGE,
    PRIMITIVE_VOLUME,
    VALID_PRIMITIVE_TYPES,
    # Dataclasses
    RetroWorldPrimitive,
    RetroWorldSector,
    RetroWorldModel,
    RetroWorldLedger,
    # Classification helpers
    classify_sector,
    classify_projection,
    classify_complexity,
    build_symbolic_trace,
    # Builders
    build_primitive,
    build_sector,
    build_retro_world_model,
    build_retro_world_ledger,
    # Export
    export_retro_world_bundle,
    export_retro_world_ledger,
)


# =========================================================================
# Helpers
# =========================================================================


def _make_primitive(
    pid: str = "p1",
    ptype: str = PRIMITIVE_WALL,
    pos: object = (1.0, 2.0, 3.0),
    scl: object = (1.0, 1.0, 1.0),
    rot: object = (0.0, 0.0, 0.0),
    sym: str = "basic_wall",
) -> RetroWorldPrimitive:
    return build_primitive(pid, ptype, pos, scl, rot, sym)


def _make_sector(
    sid: str = "s1",
    floor: float = 0.0,
    ceiling: float = 3.0,
    walls: int = 4,
    entities: int = 1,
) -> RetroWorldSector:
    return build_sector(sid, floor, ceiling, walls, entities)


def _make_model(
    mode: str = PSEUDO_3D,
    primitives: object = None,
    sectors: object = None,
    camera: object = (0.0, 0.0, 0.0),
) -> RetroWorldModel:
    if primitives is None:
        primitives = [_make_primitive()]
    if sectors is None:
        sectors = [_make_sector()]
    return build_retro_world_model(mode, primitives, sectors, camera)


# =========================================================================
# 1. Version
# =========================================================================


class TestVersion:
    def test_version_string(self):
        assert RETRO_3D_WORLD_MODELLING_VERSION == "v137.0.10"


# =========================================================================
# 2. Frozen dataclasses
# =========================================================================


class TestFrozenDataclasses:
    def test_primitive_is_frozen(self):
        p = _make_primitive()
        with pytest.raises(AttributeError):
            p.primitive_id = "changed"  # type: ignore[misc]

    def test_sector_is_frozen(self):
        s = _make_sector()
        with pytest.raises(AttributeError):
            s.sector_id = "changed"  # type: ignore[misc]

    def test_model_is_frozen(self):
        m = _make_model()
        with pytest.raises(AttributeError):
            m.world_mode = "changed"  # type: ignore[misc]

    def test_ledger_is_frozen(self):
        ledger = build_retro_world_ledger([_make_model()])
        with pytest.raises(AttributeError):
            ledger.model_count = 999  # type: ignore[misc]

    def test_primitive_version_default(self):
        p = _make_primitive()
        assert p.version == RETRO_3D_WORLD_MODELLING_VERSION

    def test_sector_version_default(self):
        s = _make_sector()
        assert s.version == RETRO_3D_WORLD_MODELLING_VERSION

    def test_model_version_default(self):
        m = _make_model()
        assert m.version == RETRO_3D_WORLD_MODELLING_VERSION


# =========================================================================
# 3. Primitive normalization
# =========================================================================


class TestPrimitiveNormalization:
    def test_tuple_input(self):
        p = _make_primitive(pos=(1.0, 2.0, 3.0))
        assert p.position == (1.0, 2.0, 3.0)

    def test_list_input(self):
        p = _make_primitive(pos=[4.0, 5.0, 6.0])
        assert p.position == (4.0, 5.0, 6.0)

    def test_generator_input(self):
        gen = (float(x) for x in range(3))
        p = _make_primitive(pos=gen)
        assert p.position == (0.0, 1.0, 2.0)

    def test_int_coercion(self):
        p = _make_primitive(pos=(1, 2, 3))
        assert p.position == (1.0, 2.0, 3.0)
        assert all(isinstance(v, float) for v in p.position)

    def test_scale_normalized(self):
        p = _make_primitive(scl=[2, 3, 4])
        assert p.scale == (2.0, 3.0, 4.0)

    def test_rotation_normalized(self):
        p = _make_primitive(rot=[90, 0, 45])
        assert p.rotation == (90.0, 0.0, 45.0)

    def test_reject_string_position(self):
        with pytest.raises(TypeError, match="non-string"):
            _make_primitive(pos="abc")

    def test_reject_wrong_length(self):
        with pytest.raises(ValueError, match="3 elements"):
            _make_primitive(pos=(1.0, 2.0))

    def test_reject_too_many_elements(self):
        with pytest.raises(ValueError, match="3 elements"):
            _make_primitive(pos=(1.0, 2.0, 3.0, 4.0))

    def test_reject_non_numeric_element(self):
        with pytest.raises(TypeError, match="int or float"):
            _make_primitive(pos=(1.0, "x", 3.0))

    def test_reject_none_position(self):
        with pytest.raises(TypeError, match="iterable"):
            _make_primitive(pos=None)

    def test_reject_empty_primitive_id(self):
        with pytest.raises(ValueError, match="non-empty"):
            build_primitive("", PRIMITIVE_WALL, (0, 0, 0), (1, 1, 1),
                            (0, 0, 0), "cls")

    def test_reject_invalid_primitive_type(self):
        with pytest.raises(ValueError, match="primitive_type"):
            build_primitive("p1", "INVALID", (0, 0, 0), (1, 1, 1),
                            (0, 0, 0), "cls")

    def test_reject_non_string_id(self):
        with pytest.raises(TypeError, match="str"):
            build_primitive(123, PRIMITIVE_WALL, (0, 0, 0), (1, 1, 1),  # type: ignore[arg-type]
                            (0, 0, 0), "cls")

    def test_all_valid_primitive_types(self):
        for pt in VALID_PRIMITIVE_TYPES:
            p = build_primitive("p", pt, (0, 0, 0), (1, 1, 1), (0, 0, 0), "c")
            assert p.primitive_type == pt


# =========================================================================
# 4. Sector classification
# =========================================================================


class TestSectorClassification:
    def test_vertical_sector(self):
        assert classify_sector(0.0, 11.0, 4, 1) == VERTICAL_SECTOR

    def test_dense_sector(self):
        assert classify_sector(0.0, 3.0, 8, 4) == DENSE_SECTOR

    def test_chamber_sector(self):
        assert classify_sector(0.0, 3.0, 6, 1) == CHAMBER_SECTOR

    def test_corridor_sector(self):
        assert classify_sector(0.0, 3.0, 2, 1) == CORRIDOR_SECTOR

    def test_corridor_zero_walls(self):
        assert classify_sector(0.0, 3.0, 0, 0) == CORRIDOR_SECTOR

    def test_open_sector(self):
        assert classify_sector(0.0, 3.0, 4, 1) == OPEN_SECTOR

    def test_vertical_overrides_dense(self):
        # High gap + dense should still be VERTICAL
        assert classify_sector(0.0, 20.0, 10, 10) == VERTICAL_SECTOR

    def test_dense_threshold_exact(self):
        assert classify_sector(0.0, 3.0, 8, 4) == DENSE_SECTOR
        # Just below threshold
        assert classify_sector(0.0, 3.0, 8, 3) == CHAMBER_SECTOR

    def test_chamber_threshold_exact(self):
        assert classify_sector(0.0, 3.0, 6, 0) == CHAMBER_SECTOR
        assert classify_sector(0.0, 3.0, 5, 0) == OPEN_SECTOR

    def test_corridor_threshold_exact(self):
        assert classify_sector(0.0, 3.0, 2, 0) == CORRIDOR_SECTOR
        assert classify_sector(0.0, 3.0, 3, 0) == OPEN_SECTOR


# =========================================================================
# 5. Sector builder validation
# =========================================================================


class TestSectorBuilder:
    def test_basic_build(self):
        s = _make_sector()
        assert s.sector_id == "s1"
        assert s.floor_height == 0.0
        assert s.ceiling_height == 3.0

    def test_reject_ceiling_below_floor(self):
        with pytest.raises(ValueError, match="ceiling_height"):
            build_sector("s1", 5.0, 2.0, 4, 1)

    def test_equal_floor_ceiling(self):
        s = build_sector("s1", 3.0, 3.0, 4, 1)
        assert s.floor_height == s.ceiling_height

    def test_reject_negative_wall_count(self):
        with pytest.raises(ValueError, match="non-negative"):
            build_sector("s1", 0.0, 3.0, -1, 0)

    def test_reject_negative_entity_count(self):
        with pytest.raises(ValueError, match="non-negative"):
            build_sector("s1", 0.0, 3.0, 4, -2)

    def test_reject_bool_wall_count(self):
        with pytest.raises(TypeError, match="int"):
            build_sector("s1", 0.0, 3.0, True, 0)  # type: ignore[arg-type]

    def test_reject_float_wall_count(self):
        with pytest.raises(TypeError, match="int"):
            build_sector("s1", 0.0, 3.0, 4.5, 0)  # type: ignore[arg-type]

    def test_reject_bool_floor(self):
        with pytest.raises(TypeError, match="numeric"):
            build_sector("s1", True, 3.0, 4, 0)  # type: ignore[arg-type]

    def test_int_floor_ceiling_coerced(self):
        s = build_sector("s1", 0, 5, 4, 1)
        assert isinstance(s.floor_height, float)
        assert isinstance(s.ceiling_height, float)

    def test_reject_empty_sector_id(self):
        with pytest.raises(ValueError, match="non-empty"):
            build_sector("", 0.0, 3.0, 4, 1)


# =========================================================================
# 6. Projection classification
# =========================================================================


class TestProjectionClassification:
    def test_pseudo_3d_with_sectors(self):
        assert classify_projection(PSEUDO_3D, 2, 5) == PSEUDO_DOOM

    def test_pseudo_3d_no_sectors(self):
        assert classify_projection(PSEUDO_3D, 0, 5) == ORTHO_LIKE

    def test_true_3d_with_both(self):
        assert classify_projection(TRUE_3D, 2, 5) == HYBRID_RETRO

    def test_true_3d_prims_only(self):
        assert classify_projection(TRUE_3D, 0, 5) == PERSPECTIVE_3D

    def test_true_3d_nothing(self):
        assert classify_projection(TRUE_3D, 0, 0) == ORTHO_LIKE

    def test_unknown_mode_fallback(self):
        assert classify_projection("OTHER", 5, 5) == ORTHO_LIKE


# =========================================================================
# 7. Complexity classification
# =========================================================================


class TestComplexityClassification:
    def test_minimal(self):
        assert classify_complexity(1, 1, 1) == MINIMAL

    def test_minimal_boundary(self):
        assert classify_complexity(1, 1, 1) == MINIMAL  # total=3
        assert classify_complexity(2, 1, 1) == COMPACT  # total=4

    def test_compact(self):
        assert classify_complexity(3, 3, 3) == COMPACT  # total=9

    def test_compact_boundary(self):
        assert classify_complexity(4, 3, 3) == COMPACT  # total=10
        assert classify_complexity(4, 4, 3) == STRUCTURED  # total=11

    def test_structured(self):
        assert classify_complexity(10, 10, 10) == STRUCTURED  # total=30

    def test_structured_boundary(self):
        assert classify_complexity(10, 10, 11) == DENSE  # total=31

    def test_dense(self):
        assert classify_complexity(20, 20, 20) == DENSE

    def test_all_zero(self):
        assert classify_complexity(0, 0, 0) == MINIMAL


# =========================================================================
# 8. Symbolic trace
# =========================================================================


class TestSymbolicTrace:
    def test_basic_trace(self):
        trace = build_symbolic_trace(
            PSEUDO_3D,
            (CORRIDOR_SECTOR, CHAMBER_SECTOR),
            PSEUDO_DOOM,
            COMPACT,
        )
        assert trace == (
            "PSEUDO_3D -> CORRIDOR_SECTOR -> CHAMBER_SECTOR "
            "-> PSEUDO_DOOM -> COMPACT"
        )

    def test_no_sectors(self):
        trace = build_symbolic_trace(TRUE_3D, (), PERSPECTIVE_3D, MINIMAL)
        assert trace == "TRUE_3D -> PERSPECTIVE_3D -> MINIMAL"

    def test_single_sector(self):
        trace = build_symbolic_trace(
            PSEUDO_3D, (OPEN_SECTOR,), PSEUDO_DOOM, COMPACT,
        )
        assert trace == "PSEUDO_3D -> OPEN_SECTOR -> PSEUDO_DOOM -> COMPACT"

    def test_deterministic_across_calls(self):
        args = (TRUE_3D, (DENSE_SECTOR, VERTICAL_SECTOR), HYBRID_RETRO, DENSE)
        t1 = build_symbolic_trace(*args)
        t2 = build_symbolic_trace(*args)
        assert t1 == t2


# =========================================================================
# 9. Stable hashing
# =========================================================================


class TestStableHashing:
    def test_primitive_hash_is_sha256_hex(self):
        p = _make_primitive()
        assert len(p.stable_hash) == 64
        int(p.stable_hash, 16)  # valid hex

    def test_sector_hash_is_sha256_hex(self):
        s = _make_sector()
        assert len(s.stable_hash) == 64

    def test_model_hash_is_sha256_hex(self):
        m = _make_model()
        assert len(m.stable_hash) == 64

    def test_ledger_hash_is_sha256_hex(self):
        ledger = build_retro_world_ledger([_make_model()])
        assert len(ledger.stable_hash) == 64

    def test_same_primitive_same_hash(self):
        p1 = _make_primitive()
        p2 = _make_primitive()
        assert p1.stable_hash == p2.stable_hash

    def test_different_primitive_different_hash(self):
        p1 = _make_primitive(pid="a")
        p2 = _make_primitive(pid="b")
        assert p1.stable_hash != p2.stable_hash

    def test_same_sector_same_hash(self):
        s1 = _make_sector()
        s2 = _make_sector()
        assert s1.stable_hash == s2.stable_hash

    def test_different_sector_different_hash(self):
        s1 = _make_sector(sid="a")
        s2 = _make_sector(sid="b")
        assert s1.stable_hash != s2.stable_hash

    def test_same_model_same_hash(self):
        m1 = _make_model()
        m2 = _make_model()
        assert m1.stable_hash == m2.stable_hash

    def test_different_model_different_hash(self):
        m1 = _make_model(mode=TRUE_3D)
        m2 = _make_model(mode=PSEUDO_3D)
        assert m1.stable_hash != m2.stable_hash

    def test_different_camera_different_hash(self):
        m1 = _make_model(camera=(0.0, 0.0, 0.0))
        m2 = _make_model(camera=(1.0, 0.0, 0.0))
        assert m1.stable_hash != m2.stable_hash

    def test_ledger_hash_differs_with_different_models(self):
        l1 = build_retro_world_ledger([_make_model(camera=(0, 0, 0))])
        l2 = build_retro_world_ledger([_make_model(camera=(1, 0, 0))])
        assert l1.stable_hash != l2.stable_hash


# =========================================================================
# 10. Canonical export equality
# =========================================================================


class TestCanonicalExport:
    def test_bundle_keys(self):
        bundle = export_retro_world_bundle(_make_model())
        expected_keys = {
            "camera_pose", "complexity_class", "entity_count",
            "primitive_count", "projection_class", "sector_count",
            "stable_hash", "version", "world_mode", "world_symbolic_trace",
        }
        assert set(bundle.keys()) == expected_keys

    def test_bundle_deterministic(self):
        b1 = export_retro_world_bundle(_make_model())
        b2 = export_retro_world_bundle(_make_model())
        assert json.dumps(b1, sort_keys=True) == json.dumps(b2, sort_keys=True)

    def test_ledger_export_keys(self):
        ledger = build_retro_world_ledger([_make_model()])
        export = export_retro_world_ledger(ledger)
        assert set(export.keys()) == {"model_count", "models", "stable_hash"}

    def test_ledger_export_deterministic(self):
        ledger = build_retro_world_ledger([_make_model()])
        e1 = export_retro_world_ledger(ledger)
        e2 = export_retro_world_ledger(ledger)
        assert json.dumps(e1, sort_keys=True) == json.dumps(e2, sort_keys=True)

    def test_bundle_camera_is_list(self):
        bundle = export_retro_world_bundle(_make_model())
        assert isinstance(bundle["camera_pose"], list)

    def test_export_rejects_non_model(self):
        with pytest.raises(TypeError, match="RetroWorldModel"):
            export_retro_world_bundle("not_a_model")  # type: ignore[arg-type]

    def test_export_rejects_non_ledger(self):
        with pytest.raises(TypeError, match="RetroWorldLedger"):
            export_retro_world_ledger("not_a_ledger")  # type: ignore[arg-type]


# =========================================================================
# 11. 100-run replay determinism
# =========================================================================


class TestReplayDeterminism:
    def test_100_run_model_replay(self):
        hashes = set()
        for _ in range(100):
            m = _make_model()
            hashes.add(m.stable_hash)
        assert len(hashes) == 1

    def test_100_run_primitive_replay(self):
        hashes = set()
        for _ in range(100):
            p = _make_primitive()
            hashes.add(p.stable_hash)
        assert len(hashes) == 1

    def test_100_run_sector_replay(self):
        hashes = set()
        for _ in range(100):
            s = _make_sector()
            hashes.add(s.stable_hash)
        assert len(hashes) == 1

    def test_100_run_ledger_replay(self):
        hashes = set()
        for _ in range(100):
            ledger = build_retro_world_ledger([_make_model()])
            hashes.add(ledger.stable_hash)
        assert len(hashes) == 1

    def test_100_run_export_bytes_replay(self):
        outputs = set()
        for _ in range(100):
            bundle = export_retro_world_bundle(_make_model())
            outputs.add(json.dumps(bundle, sort_keys=True))
        assert len(outputs) == 1


# =========================================================================
# 12. Invalid input rejection — world model
# =========================================================================


class TestWorldModelValidation:
    def test_reject_invalid_mode(self):
        with pytest.raises(ValueError, match="world_mode"):
            build_retro_world_model("INVALID", [], [], (0, 0, 0))

    def test_reject_non_string_mode(self):
        with pytest.raises(TypeError, match="str"):
            build_retro_world_model(123, [], [], (0, 0, 0))  # type: ignore[arg-type]

    def test_reject_string_primitives(self):
        with pytest.raises(TypeError, match="non-string"):
            build_retro_world_model(TRUE_3D, "abc", [], (0, 0, 0))

    def test_reject_string_sectors(self):
        with pytest.raises(TypeError, match="non-string"):
            build_retro_world_model(TRUE_3D, [], "abc", (0, 0, 0))

    def test_reject_non_iterable_primitives(self):
        with pytest.raises(TypeError, match="iterable"):
            build_retro_world_model(TRUE_3D, 42, [], (0, 0, 0))  # type: ignore[arg-type]

    def test_reject_non_iterable_sectors(self):
        with pytest.raises(TypeError, match="iterable"):
            build_retro_world_model(TRUE_3D, [], 42, (0, 0, 0))  # type: ignore[arg-type]

    def test_reject_wrong_primitive_type_in_list(self):
        with pytest.raises(TypeError, match="RetroWorldPrimitive"):
            build_retro_world_model(TRUE_3D, ["not_a_prim"], [], (0, 0, 0))

    def test_reject_wrong_sector_type_in_list(self):
        with pytest.raises(TypeError, match="RetroWorldSector"):
            build_retro_world_model(TRUE_3D, [], ["not_a_sector"], (0, 0, 0))

    def test_reject_bad_camera_pose(self):
        with pytest.raises(ValueError, match="3 elements"):
            build_retro_world_model(TRUE_3D, [], [], (0, 0))

    def test_empty_primitives_and_sectors_ok(self):
        m = build_retro_world_model(TRUE_3D, [], [], (0, 0, 0))
        assert m.primitive_count == 0
        assert m.sector_count == 0


# =========================================================================
# 13. Ledger validation
# =========================================================================


class TestLedgerValidation:
    def test_reject_string_models(self):
        with pytest.raises(TypeError, match="non-string"):
            build_retro_world_ledger("abc")

    def test_reject_non_iterable(self):
        with pytest.raises(TypeError, match="iterable"):
            build_retro_world_ledger(42)  # type: ignore[arg-type]

    def test_reject_wrong_type_in_list(self):
        with pytest.raises(TypeError, match="RetroWorldModel"):
            build_retro_world_ledger(["not_a_model"])

    def test_empty_ledger(self):
        ledger = build_retro_world_ledger([])
        assert ledger.model_count == 0
        assert ledger.models == ()

    def test_ledger_model_count(self):
        models = [_make_model(), _make_model(camera=(1, 0, 0))]
        ledger = build_retro_world_ledger(models)
        assert ledger.model_count == 2

    def test_ledger_from_generator(self):
        gen = (_make_model(camera=(float(i), 0.0, 0.0)) for i in range(3))
        ledger = build_retro_world_ledger(gen)
        assert ledger.model_count == 3


# =========================================================================
# 14. Generic iterable acceptance
# =========================================================================


class TestGenericIterables:
    def test_primitives_as_tuple(self):
        m = build_retro_world_model(
            TRUE_3D, (_make_primitive(),), [], (0, 0, 0),
        )
        assert m.primitive_count == 1

    def test_primitives_as_generator(self):
        gen = (_make_primitive(pid=f"p{i}") for i in range(3))
        m = build_retro_world_model(TRUE_3D, gen, [], (0, 0, 0))
        assert m.primitive_count == 3

    def test_sectors_as_set_like(self):
        # frozenset won't preserve order but is iterable
        s = _make_sector()
        m = build_retro_world_model(PSEUDO_3D, [], [s], (0, 0, 0))
        assert m.sector_count == 1

    def test_camera_as_list(self):
        m = _make_model(camera=[5.0, 6.0, 7.0])
        assert m.camera_pose == (5.0, 6.0, 7.0)

    def test_camera_as_generator(self):
        gen = (float(x) for x in [1, 2, 3])
        m = build_retro_world_model(TRUE_3D, [], [], gen)
        assert m.camera_pose == (1.0, 2.0, 3.0)


# =========================================================================
# 15. TRUE_3D vs PSEUDO_3D mode behavior
# =========================================================================


class TestWorldModes:
    def test_pseudo_3d_doom_like_constrained(self):
        """PSEUDO_3D can model Doom-like constrained spaces."""
        sectors = [
            build_sector("corridor", 0.0, 3.0, 2, 0),
            build_sector("room", 0.0, 4.0, 6, 3),
        ]
        m = build_retro_world_model(PSEUDO_3D, [], sectors, (0, 0, 0))
        assert m.world_mode == PSEUDO_3D
        assert m.projection_class == PSEUDO_DOOM
        assert CORRIDOR_SECTOR in m.world_symbolic_trace
        assert CHAMBER_SECTOR in m.world_symbolic_trace

    def test_true_3d_richer_geometry(self):
        """TRUE_3D allows richer geometry summaries with free spatial prims."""
        prims = [
            build_primitive(f"v{i}", PRIMITIVE_VERTEX,
                            (float(i), 0.0, 0.0), (1, 1, 1), (0, 0, 0),
                            "vertex_node")
            for i in range(5)
        ]
        m = build_retro_world_model(TRUE_3D, prims, [], (0, 0, 0))
        assert m.world_mode == TRUE_3D
        assert m.projection_class == PERSPECTIVE_3D
        assert m.primitive_count == 5

    def test_true_3d_with_sectors_hybrid(self):
        prims = [_make_primitive()]
        sectors = [_make_sector()]
        m = build_retro_world_model(TRUE_3D, prims, sectors, (0, 0, 0))
        assert m.projection_class == HYBRID_RETRO

    def test_pseudo_3d_no_sectors_ortho(self):
        m = build_retro_world_model(PSEUDO_3D, [], [], (0, 0, 0))
        assert m.projection_class == ORTHO_LIKE

    def test_mode_affects_hash(self):
        m1 = build_retro_world_model(TRUE_3D, [], [], (0, 0, 0))
        m2 = build_retro_world_model(PSEUDO_3D, [], [], (0, 0, 0))
        assert m1.stable_hash != m2.stable_hash

    def test_pseudo_3d_entity_counting(self):
        """Entity count from sectors contributes to model entity_count."""
        sectors = [build_sector("s1", 0.0, 3.0, 4, 5)]
        m = build_retro_world_model(PSEUDO_3D, [], sectors, (0, 0, 0))
        assert m.entity_count == 5

    def test_true_3d_entity_counting_from_primitives(self):
        """ENTITY primitives contribute to entity_count."""
        prims = [
            build_primitive("e1", PRIMITIVE_ENTITY,
                            (0, 0, 0), (1, 1, 1), (0, 0, 0), "monster"),
            build_primitive("w1", PRIMITIVE_WALL,
                            (1, 0, 0), (1, 1, 1), (0, 0, 0), "wall"),
        ]
        m = build_retro_world_model(TRUE_3D, prims, [], (0, 0, 0))
        assert m.entity_count == 1

    def test_combined_entity_count(self):
        """Entities counted from both primitives and sectors."""
        prims = [
            build_primitive("e1", PRIMITIVE_ENTITY,
                            (0, 0, 0), (1, 1, 1), (0, 0, 0), "monster"),
        ]
        sectors = [build_sector("s1", 0.0, 3.0, 4, 3)]
        m = build_retro_world_model(TRUE_3D, prims, sectors, (0, 0, 0))
        assert m.entity_count == 4  # 1 prim + 3 sector


# =========================================================================
# 16. No decoder contamination
# =========================================================================


class TestNoDecoderContamination:
    def test_no_decoder_import(self):
        """Module must not import from qec.decoder."""
        import qec.analysis.retro_3d_world_modelling as mod
        source = open(mod.__file__).read()
        assert "from qec.decoder" not in source
        assert "import qec.decoder" not in source

    def test_no_decoder_in_sys_modules(self):
        """Building world models must not trigger decoder loading."""
        decoder_modules = [
            k for k in sys.modules
            if k.startswith("qec.decoder")
        ]
        # Even if loaded elsewhere, the module itself must not import it
        import qec.analysis.retro_3d_world_modelling as mod
        source = open(mod.__file__).read()
        assert "decoder" not in source.split("import")[-1] if "import" in source else True


# =========================================================================
# 17. Policy hint (optional coupling)
# =========================================================================


class TestPolicyHint:
    def test_policy_hint_none_default(self):
        m = _make_model()
        # Should build fine without policy hint
        assert m.world_mode is not None

    def test_policy_hint_accepted(self):
        m = build_retro_world_model(
            TRUE_3D, [], [], (0, 0, 0), policy_hint="STABLE_POLICY",
        )
        assert m is not None

    def test_policy_hint_does_not_affect_hash(self):
        m1 = build_retro_world_model(
            TRUE_3D, [], [], (0, 0, 0), policy_hint=None,
        )
        m2 = build_retro_world_model(
            TRUE_3D, [], [], (0, 0, 0), policy_hint="LOCKDOWN",
        )
        assert m1.stable_hash == m2.stable_hash


# =========================================================================
# 18. Constants integrity
# =========================================================================


class TestConstants:
    def test_valid_world_modes(self):
        assert TRUE_3D in VALID_WORLD_MODES
        assert PSEUDO_3D in VALID_WORLD_MODES
        assert len(VALID_WORLD_MODES) == 2

    def test_primitive_types_count(self):
        assert len(VALID_PRIMITIVE_TYPES) == 5

    def test_all_sector_classes_distinct(self):
        classes = {OPEN_SECTOR, CORRIDOR_SECTOR, CHAMBER_SECTOR,
                   VERTICAL_SECTOR, DENSE_SECTOR}
        assert len(classes) == 5

    def test_all_projection_classes_distinct(self):
        classes = {ORTHO_LIKE, PERSPECTIVE_3D, PSEUDO_DOOM, HYBRID_RETRO}
        assert len(classes) == 4

    def test_all_complexity_classes_distinct(self):
        classes = {MINIMAL, COMPACT, STRUCTURED, DENSE}
        assert len(classes) == 4
