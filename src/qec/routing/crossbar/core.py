# SPDX-License-Identifier: MPL-2.0
"""Deterministic v172.0 Crossbar coordinate-matrix contracts."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Mapping

from qec.sonify.canonical import (
    canonical_sha256,
    require_int,
    require_nonempty_text,
    validate_sha256,
)

QEC_VERSION = "172.0.0"
CONTRACT_VERSION = "172.0"
MATRIX_SCHEMA = "qec.crossbar-matrix-manifest.v1"
INTERSECTION_ID_SCHEMA = "qec.crossbar-intersection-id.v1"
VALIDATION_SCHEMA = "qec.crossbar-matrix-validation.v1"

LINK_STATES: Final[tuple[str, ...]] = ("busy", "idle", "quarantined", "unavailable")
_AXES: Final[tuple[str, ...]] = ("horizontal", "vertical")
_MAX_AXIS_LINKS = 4096
_MAX_INTERSECTIONS = 65536

_CLAIM_BOUNDARY_VALUES: Final = {
    "classical_software_model_only": True,
    "physical_crossbar_fidelity": False,
    "marker_authority_present": False,
    "route_search_present": False,
    "reservation_present": False,
    "connection_commit_present": False,
    "decoder_output_mutation_permitted": False,
    "payload_mutation_permitted": False,
    "browser_demo_is_canonical_evidence": False,
    "receipt_proves": "immutable_matrix_identity_and_declared_initial_link_state",
    "receipt_does_not_prove": "end_to_end_route_continuity_or_physical_network_behavior",
}
CLAIM_BOUNDARY: Final[Mapping[str, object]] = MappingProxyType(_CLAIM_BOUNDARY_VALUES)


def _exact_object(payload: object, *, label: str, fields: set[str]) -> dict[str, object]:
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be an object")
    if set(payload) != fields:
        raise ValueError(f"{label} must contain exactly the canonical fields")
    return payload


def _exact_list(payload: object, *, label: str) -> list[object]:
    if not isinstance(payload, list):
        raise ValueError(f"{label} must be a list")
    return payload


def _validate_hashed_artifact(payload: object, *, schema: str, label: str) -> dict[str, object]:
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be an object")
    if payload.get("schema") != schema:
        raise ValueError(f"unexpected {label} schema")
    observed = validate_sha256(payload.get("sha256"), f"{label}.sha256")
    unsigned = dict(payload)
    unsigned.pop("sha256", None)
    if canonical_sha256(unsigned) != observed:
        raise ValueError(f"{label} hash mismatch")
    return payload


@dataclass(frozen=True)
class CrossbarLink:
    """One immutable horizontal or vertical link record."""

    axis: str
    link_id: str
    ordinal: int
    state: str = "idle"

    def __post_init__(self) -> None:
        if self.axis not in _AXES:
            raise ValueError("link.axis must be horizontal or vertical")
        require_nonempty_text(self.link_id, "link.link_id")
        require_int(self.ordinal, "link.ordinal", minimum=0, maximum=_MAX_AXIS_LINKS - 1)
        if self.state not in LINK_STATES:
            raise ValueError(f"link.state must be one of {LINK_STATES}")

    def as_dict(self) -> dict[str, object]:
        return {
            "axis": self.axis,
            "link_id": self.link_id,
            "ordinal": self.ordinal,
            "state": self.state,
        }

    @classmethod
    def from_dict(cls, payload: object) -> "CrossbarLink":
        item = _exact_object(
            payload,
            label="crossbar link",
            fields={"axis", "link_id", "ordinal", "state"},
        )
        return cls(item["axis"], item["link_id"], item["ordinal"], item["state"])  # type: ignore[arg-type]


@dataclass(frozen=True)
class CrossbarIntersection:
    """Canonical identity for one horizontal/vertical coordinate."""

    matrix_id: str
    horizontal_link_id: str
    horizontal_ordinal: int
    vertical_link_id: str
    vertical_ordinal: int

    def __post_init__(self) -> None:
        require_nonempty_text(self.matrix_id, "intersection.matrix_id")
        require_nonempty_text(self.horizontal_link_id, "intersection.horizontal_link_id")
        require_nonempty_text(self.vertical_link_id, "intersection.vertical_link_id")
        require_int(
            self.horizontal_ordinal,
            "intersection.horizontal_ordinal",
            minimum=0,
            maximum=_MAX_AXIS_LINKS - 1,
        )
        require_int(
            self.vertical_ordinal,
            "intersection.vertical_ordinal",
            minimum=0,
            maximum=_MAX_AXIS_LINKS - 1,
        )

    @property
    def intersection_id(self) -> str:
        return canonical_sha256(
            {
                "schema": INTERSECTION_ID_SCHEMA,
                "contract_version": CONTRACT_VERSION,
                "matrix_id": self.matrix_id,
                "horizontal_link_id": self.horizontal_link_id,
                "horizontal_ordinal": self.horizontal_ordinal,
                "vertical_link_id": self.vertical_link_id,
                "vertical_ordinal": self.vertical_ordinal,
            }
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "horizontal_link_id": self.horizontal_link_id,
            "horizontal_ordinal": self.horizontal_ordinal,
            "vertical_link_id": self.vertical_link_id,
            "vertical_ordinal": self.vertical_ordinal,
            "intersection_id": self.intersection_id,
        }

    @classmethod
    def from_dict(cls, matrix_id: str, payload: object) -> "CrossbarIntersection":
        item = _exact_object(
            payload,
            label="crossbar intersection",
            fields={
                "horizontal_link_id",
                "horizontal_ordinal",
                "vertical_link_id",
                "vertical_ordinal",
                "intersection_id",
            },
        )
        intersection = cls(
            matrix_id=matrix_id,
            horizontal_link_id=item["horizontal_link_id"],  # type: ignore[arg-type]
            horizontal_ordinal=item["horizontal_ordinal"],  # type: ignore[arg-type]
            vertical_link_id=item["vertical_link_id"],  # type: ignore[arg-type]
            vertical_ordinal=item["vertical_ordinal"],  # type: ignore[arg-type]
        )
        observed = validate_sha256(item["intersection_id"], "intersection.intersection_id")
        if intersection.intersection_id != observed:
            raise ValueError("crossbar intersection identity mismatch")
        return intersection


@dataclass(frozen=True)
class CrossbarMatrix:
    """Immutable bounded horizontal/vertical matrix.

    v172.0 intentionally models identity and declared initial state only.
    It contains no marker, search, reservation, closure, or commit authority.
    """

    matrix_id: str
    horizontal_links: tuple[CrossbarLink, ...]
    vertical_links: tuple[CrossbarLink, ...]

    def __post_init__(self) -> None:
        require_nonempty_text(self.matrix_id, "matrix.matrix_id")
        if not self.horizontal_links or not self.vertical_links:
            raise ValueError("crossbar matrix requires horizontal and vertical links")
        if len(self.horizontal_links) > _MAX_AXIS_LINKS or len(self.vertical_links) > _MAX_AXIS_LINKS:
            raise ValueError("crossbar matrix exceeds bounded axis size")
        if len(self.horizontal_links) * len(self.vertical_links) > _MAX_INTERSECTIONS:
            raise ValueError("crossbar matrix exceeds bounded intersection count")
        self._validate_axis(self.horizontal_links, "horizontal")
        self._validate_axis(self.vertical_links, "vertical")
        horizontal_ids = {link.link_id for link in self.horizontal_links}
        vertical_ids = {link.link_id for link in self.vertical_links}
        if horizontal_ids & vertical_ids:
            raise ValueError("crossbar link ids must be globally unique across axes")

    @staticmethod
    def _validate_axis(links: tuple[CrossbarLink, ...], axis: str) -> None:
        if any(link.axis != axis for link in links):
            raise ValueError(f"{axis} link collection contains wrong-axis record")
        expected_ordinals = tuple(range(len(links)))
        observed_ordinals = tuple(link.ordinal for link in links)
        if observed_ordinals != expected_ordinals:
            raise ValueError(f"{axis} links must use contiguous canonical ordinal order")
        link_ids = tuple(link.link_id for link in links)
        if len(set(link_ids)) != len(link_ids):
            raise ValueError(f"{axis} link ids must be unique")

    @property
    def intersections(self) -> tuple[CrossbarIntersection, ...]:
        return tuple(
            CrossbarIntersection(
                matrix_id=self.matrix_id,
                horizontal_link_id=horizontal.link_id,
                horizontal_ordinal=horizontal.ordinal,
                vertical_link_id=vertical.link_id,
                vertical_ordinal=vertical.ordinal,
            )
            for horizontal in self.horizontal_links
            for vertical in self.vertical_links
        )

    def coordinate(self, horizontal_link_id: str, vertical_link_id: str) -> CrossbarIntersection:
        require_nonempty_text(horizontal_link_id, "horizontal_link_id")
        require_nonempty_text(vertical_link_id, "vertical_link_id")
        horizontal = next(
            (link for link in self.horizontal_links if link.link_id == horizontal_link_id),
            None,
        )
        vertical = next(
            (link for link in self.vertical_links if link.link_id == vertical_link_id),
            None,
        )
        if horizontal is None or vertical is None:
            raise ValueError("unknown crossbar coordinate")
        return CrossbarIntersection(
            matrix_id=self.matrix_id,
            horizontal_link_id=horizontal.link_id,
            horizontal_ordinal=horizontal.ordinal,
            vertical_link_id=vertical.link_id,
            vertical_ordinal=vertical.ordinal,
        )

    def as_dict(self) -> dict[str, object]:
        unsigned = {
            "schema": MATRIX_SCHEMA,
            "contract_version": CONTRACT_VERSION,
            "matrix_id": self.matrix_id,
            "link_state_vocabulary": list(LINK_STATES),
            "horizontal_links": [link.as_dict() for link in self.horizontal_links],
            "vertical_links": [link.as_dict() for link in self.vertical_links],
            "intersections": [intersection.as_dict() for intersection in self.intersections],
            "claim_boundary": dict(CLAIM_BOUNDARY),
        }
        return {**unsigned, "sha256": canonical_sha256(unsigned)}

    def sha256(self) -> str:
        return self.as_dict()["sha256"]  # type: ignore[return-value]

    @classmethod
    def from_dict(cls, payload: object) -> "CrossbarMatrix":
        item = _validate_hashed_artifact(
            payload,
            schema=MATRIX_SCHEMA,
            label="crossbar matrix manifest",
        )
        expected_fields = {
            "schema",
            "contract_version",
            "matrix_id",
            "link_state_vocabulary",
            "horizontal_links",
            "vertical_links",
            "intersections",
            "claim_boundary",
            "sha256",
        }
        if set(item) != expected_fields:
            raise ValueError("crossbar matrix manifest fields are not canonical")
        if item["contract_version"] != CONTRACT_VERSION:
            raise ValueError("unexpected crossbar matrix contract version")
        if item["link_state_vocabulary"] != list(LINK_STATES):
            raise ValueError("crossbar link-state vocabulary is not canonical")
        if item["claim_boundary"] != dict(CLAIM_BOUNDARY):
            raise ValueError("crossbar claim boundary mismatch")

        horizontal_values = _exact_list(item["horizontal_links"], label="horizontal links")
        vertical_values = _exact_list(item["vertical_links"], label="vertical links")
        supplied_intersections = _exact_list(item["intersections"], label="crossbar intersections")

        matrix = cls(
            matrix_id=item["matrix_id"],  # type: ignore[arg-type]
            horizontal_links=tuple(CrossbarLink.from_dict(value) for value in horizontal_values),
            vertical_links=tuple(CrossbarLink.from_dict(value) for value in vertical_values),
        )
        parsed_intersections = tuple(
            CrossbarIntersection.from_dict(matrix.matrix_id, value)
            for value in supplied_intersections
        )
        if parsed_intersections != matrix.intersections:
            raise ValueError("crossbar intersections must be complete canonical row-major closure")
        return matrix


def validate_matrix_manifest(payload: object) -> dict[str, object]:
    """Replay the matrix identity from canonical source records."""

    matrix = CrossbarMatrix.from_dict(payload)
    manifest_hash = matrix.sha256()
    unsigned = {
        "schema": VALIDATION_SCHEMA,
        "contract_version": CONTRACT_VERSION,
        "matrix_id": matrix.matrix_id,
        "crossbar_matrix_receipt_hash": manifest_hash,
        "horizontal_link_count": len(matrix.horizontal_links),
        "vertical_link_count": len(matrix.vertical_links),
        "intersection_count": len(matrix.intersections),
        "canonical_ordering_verified": True,
        "full_coordinate_coverage_verified": True,
        "intersection_identities_verified": True,
        "claim_boundary_verified": True,
        "all_passed": True,
    }
    return {**unsigned, "sha256": canonical_sha256(unsigned)}


def demo_matrix(
    matrix_id: str = "crossbar-demo",
    *,
    horizontal_count: int = 4,
    vertical_count: int = 4,
    state_overrides: Mapping[str, str] | None = None,
) -> CrossbarMatrix:
    """Build a deterministic bounded fixture matrix."""

    require_nonempty_text(matrix_id, "matrix_id")
    require_int(horizontal_count, "horizontal_count", minimum=1, maximum=_MAX_AXIS_LINKS)
    require_int(vertical_count, "vertical_count", minimum=1, maximum=_MAX_AXIS_LINKS)
    if horizontal_count * vertical_count > _MAX_INTERSECTIONS:
        raise ValueError("requested demo matrix exceeds bounded intersection count")

    overrides = dict(state_overrides or {})
    horizontal_ids = tuple(f"H{index:03d}" for index in range(horizontal_count))
    vertical_ids = tuple(f"V{index:03d}" for index in range(vertical_count))
    known_ids = set(horizontal_ids) | set(vertical_ids)
    unknown = set(overrides) - known_ids
    if unknown:
        raise ValueError(f"state override references unknown links: {sorted(unknown)}")
    for link_id, state in overrides.items():
        require_nonempty_text(link_id, "state override link id")
        if state not in LINK_STATES:
            raise ValueError(f"state override for {link_id} must be one of {LINK_STATES}")

    horizontal_links = tuple(
        CrossbarLink("horizontal", link_id, index, overrides.get(link_id, "idle"))
        for index, link_id in enumerate(horizontal_ids)
    )
    vertical_links = tuple(
        CrossbarLink("vertical", link_id, index, overrides.get(link_id, "idle"))
        for index, link_id in enumerate(vertical_ids)
    )
    return CrossbarMatrix(matrix_id, horizontal_links, vertical_links)
