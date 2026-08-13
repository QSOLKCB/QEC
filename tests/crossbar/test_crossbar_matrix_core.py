# SPDX-License-Identifier: MPL-2.0
from copy import deepcopy

import pytest

from qec.routing.crossbar import (
    CLAIM_BOUNDARY,
    CrossbarLink,
    CrossbarMatrix,
    demo_matrix,
    validate_matrix_manifest,
)
from qec.sonify.canonical import canonical_sha256


def _resign(value: dict[str, object]) -> dict[str, object]:
    unsigned = dict(value)
    unsigned.pop("sha256", None)
    value["sha256"] = canonical_sha256(unsigned)
    return value


def test_same_declared_matrix_produces_same_manifest_and_hash() -> None:
    first = demo_matrix("matrix-a", horizontal_count=3, vertical_count=2)
    second = demo_matrix("matrix-a", horizontal_count=3, vertical_count=2)
    assert first.as_dict() == second.as_dict()
    assert first.sha256() == second.sha256()


def test_matrix_has_complete_row_major_coordinate_closure() -> None:
    matrix = demo_matrix("matrix-grid", horizontal_count=3, vertical_count=2)
    coordinates = [
        (
            intersection.horizontal_ordinal,
            intersection.vertical_ordinal,
            intersection.horizontal_link_id,
            intersection.vertical_link_id,
        )
        for intersection in matrix.intersections
    ]
    assert coordinates == [
        (0, 0, "H000", "V000"),
        (0, 1, "H000", "V001"),
        (1, 0, "H001", "V000"),
        (1, 1, "H001", "V001"),
        (2, 0, "H002", "V000"),
        (2, 1, "H002", "V001"),
    ]
    assert len({item.intersection_id for item in matrix.intersections}) == 6


def test_intersection_identity_is_bound_to_matrix_and_coordinate() -> None:
    first = demo_matrix("matrix-a", horizontal_count=2, vertical_count=2)
    second = demo_matrix("matrix-b", horizontal_count=2, vertical_count=2)
    assert first.coordinate("H000", "V000").intersection_id != first.coordinate("H000", "V001").intersection_id
    assert first.coordinate("H000", "V000").intersection_id != second.coordinate("H000", "V000").intersection_id


def test_link_state_is_hash_bound_and_vocabulary_is_exact() -> None:
    idle = demo_matrix("matrix-state", horizontal_count=2, vertical_count=2)
    busy = demo_matrix(
        "matrix-state",
        horizontal_count=2,
        vertical_count=2,
        state_overrides={"H001": "busy"},
    )
    assert idle.sha256() != busy.sha256()
    assert busy.horizontal_links[1].state == "busy"
    with pytest.raises(ValueError):
        demo_matrix(
            "matrix-state",
            horizontal_count=2,
            vertical_count=2,
            state_overrides={"H001": "mystery"},
        )


def test_axis_order_and_ids_are_canonical() -> None:
    with pytest.raises(ValueError):
        CrossbarMatrix(
            "bad-order",
            (
                CrossbarLink("horizontal", "H001", 1),
                CrossbarLink("horizontal", "H000", 0),
            ),
            (CrossbarLink("vertical", "V000", 0),),
        )
    with pytest.raises(ValueError):
        CrossbarMatrix(
            "duplicate-id",
            (CrossbarLink("horizontal", "X", 0),),
            (CrossbarLink("vertical", "X", 0),),
        )


def test_resigned_partial_coordinate_manifest_is_rejected() -> None:
    manifest = demo_matrix("matrix-tamper", horizontal_count=2, vertical_count=2).as_dict()
    tampered = deepcopy(manifest)
    tampered["intersections"] = tampered["intersections"][:-1]  # type: ignore[index]
    _resign(tampered)
    with pytest.raises(ValueError, match="complete canonical row-major closure"):
        validate_matrix_manifest(tampered)


def test_resigned_intersection_identity_tamper_is_rejected() -> None:
    manifest = demo_matrix("matrix-tamper-id", horizontal_count=2, vertical_count=2).as_dict()
    tampered = deepcopy(manifest)
    tampered["intersections"][0]["intersection_id"] = "0" * 64  # type: ignore[index]
    _resign(tampered)
    with pytest.raises(ValueError, match="intersection identity mismatch"):
        validate_matrix_manifest(tampered)


def test_validation_replays_manifest_instead_of_trusting_outer_hash() -> None:
    manifest = demo_matrix("matrix-valid", horizontal_count=4, vertical_count=3).as_dict()
    validation = validate_matrix_manifest(manifest)
    assert validation["all_passed"] is True
    assert validation["crossbar_matrix_receipt_hash"] == manifest["sha256"]
    assert validation["intersection_count"] == 12


def test_v172_0_claim_boundary_excludes_future_authority() -> None:
    assert CLAIM_BOUNDARY["marker_authority_present"] is False
    assert CLAIM_BOUNDARY["route_search_present"] is False
    assert CLAIM_BOUNDARY["reservation_present"] is False
    assert CLAIM_BOUNDARY["connection_commit_present"] is False
    assert CLAIM_BOUNDARY["decoder_output_mutation_permitted"] is False
    assert CLAIM_BOUNDARY["payload_mutation_permitted"] is False
