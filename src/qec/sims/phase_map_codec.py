# SPDX-License-Identifier: MIT
"""Phase map export codec — v133.7.0.

Canonical export, load, and hashing for PhaseMap artifacts.
Deterministic JSON serialization with SHA-256 content hashing.

All operations are pure, deterministic, and replay-safe.
No file IO. No plotting. No external dependencies.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Tuple

from qec.sims.phase_map_generator import PhaseCell, PhaseMap

_SCHEMA_VERSION = "1.0.0"
_PLACEHOLDER_HASH = "0" * 64


@dataclass(frozen=True)
class PhaseMapExportMetadata:
    """Frozen metadata for a phase map export bundle."""

    schema_version: str
    created_by_release: str
    trace_hash: str


@dataclass(frozen=True)
class PhaseMapExportBundle:
    """Frozen export bundle containing phase map, metadata, and ASCII render."""

    phase_map: PhaseMap
    metadata: PhaseMapExportMetadata
    ascii_render: str


def _format_float(value: float) -> float:
    """Round float to 12 significant decimal places for canonical representation."""
    return round(value, 12)


def _phase_cell_to_dict(cell: PhaseCell) -> dict:
    return {
        "coupling_profile": [_format_float(v) for v in cell.coupling_profile],
        "decay": _format_float(cell.decay),
        "divergence_score": _format_float(cell.divergence_score),
        "regime_label": cell.regime_label,
    }


def _phase_map_to_dict(pm: PhaseMap) -> dict:
    return {
        "cells": [_phase_cell_to_dict(c) for c in pm.cells],
        "critical_count": pm.critical_count,
        "divergent_count": pm.divergent_count,
        "max_divergence": _format_float(pm.max_divergence),
        "num_cols": pm.num_cols,
        "num_rows": pm.num_rows,
        "stable_count": pm.stable_count,
    }


def _bundle_to_dict(bundle: PhaseMapExportBundle) -> dict:
    return {
        "ascii_render": bundle.ascii_render,
        "metadata": {
            "created_by_release": bundle.metadata.created_by_release,
            "schema_version": bundle.metadata.schema_version,
            "trace_hash": bundle.metadata.trace_hash,
        },
        "phase_map": _phase_map_to_dict(bundle.phase_map),
    }


def _canonical_json(d: dict) -> str:
    return json.dumps(d, sort_keys=True, separators=(",", ":"))


def export_phase_map_to_json(bundle: PhaseMapExportBundle) -> str:
    """Export a PhaseMapExportBundle to canonical JSON.

    Parameters
    ----------
    bundle : PhaseMapExportBundle
        The bundle to serialize.

    Returns
    -------
    str
        Canonical JSON string with sorted keys and compact separators.
    """
    return _canonical_json(_bundle_to_dict(bundle))


def load_phase_map_from_json(text: str) -> PhaseMapExportBundle:
    """Reconstruct a PhaseMapExportBundle from canonical JSON.

    Parameters
    ----------
    text : str
        JSON string produced by export_phase_map_to_json.

    Returns
    -------
    PhaseMapExportBundle
        Reconstructed frozen bundle.

    Raises
    ------
    ValueError
        If JSON is malformed or required keys are missing.
    """
    try:
        d = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"malformed JSON: {exc}") from exc

    _require_keys(d, ("ascii_render", "metadata", "phase_map"), "root")

    md = d["metadata"]
    _require_keys(
        md,
        ("created_by_release", "schema_version", "trace_hash"),
        "metadata",
    )

    pm_d = d["phase_map"]
    _require_keys(
        pm_d,
        (
            "cells",
            "critical_count",
            "divergent_count",
            "max_divergence",
            "num_cols",
            "num_rows",
            "stable_count",
        ),
        "phase_map",
    )

    cells = tuple(
        PhaseCell(
            decay=float(c["decay"]),
            coupling_profile=tuple(float(v) for v in c["coupling_profile"]),
            regime_label=str(c["regime_label"]),
            divergence_score=float(c["divergence_score"]),
        )
        for c in pm_d["cells"]
    )

    phase_map = PhaseMap(
        cells=cells,
        num_rows=int(pm_d["num_rows"]),
        num_cols=int(pm_d["num_cols"]),
        stable_count=int(pm_d["stable_count"]),
        critical_count=int(pm_d["critical_count"]),
        divergent_count=int(pm_d["divergent_count"]),
        max_divergence=float(pm_d["max_divergence"]),
    )

    metadata = PhaseMapExportMetadata(
        schema_version=str(md["schema_version"]),
        created_by_release=str(md["created_by_release"]),
        trace_hash=str(md["trace_hash"]),
    )

    return PhaseMapExportBundle(
        phase_map=phase_map,
        metadata=metadata,
        ascii_render=str(d["ascii_render"]),
    )


def compute_phase_map_hash(bundle: PhaseMapExportBundle) -> str:
    """Compute SHA-256 hash over canonical JSON of the bundle.

    The hash is computed over the bundle with trace_hash set to the
    placeholder value, ensuring idempotent hashing regardless of
    whether the bundle already contains a computed hash.

    Parameters
    ----------
    bundle : PhaseMapExportBundle
        The bundle to hash.

    Returns
    -------
    str
        Hex-encoded SHA-256 digest.
    """
    normalized = PhaseMapExportBundle(
        phase_map=bundle.phase_map,
        metadata=PhaseMapExportMetadata(
            schema_version=bundle.metadata.schema_version,
            created_by_release=bundle.metadata.created_by_release,
            trace_hash=_PLACEHOLDER_HASH,
        ),
        ascii_render=bundle.ascii_render,
    )
    canonical = export_phase_map_to_json(normalized)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def with_computed_phase_hash(
    bundle: PhaseMapExportBundle,
) -> PhaseMapExportBundle:
    """Return a finalized bundle with the trace_hash computed and embedded.

    Parameters
    ----------
    bundle : PhaseMapExportBundle
        Bundle (may have placeholder or stale trace_hash).

    Returns
    -------
    PhaseMapExportBundle
        New frozen bundle with trace_hash set to the SHA-256 digest.
    """
    digest = compute_phase_map_hash(bundle)
    return PhaseMapExportBundle(
        phase_map=bundle.phase_map,
        metadata=PhaseMapExportMetadata(
            schema_version=bundle.metadata.schema_version,
            created_by_release=bundle.metadata.created_by_release,
            trace_hash=digest,
        ),
        ascii_render=bundle.ascii_render,
    )


def _require_keys(
    d: dict, keys: tuple, context: str,
) -> None:
    for k in keys:
        if k not in d:
            raise ValueError(f"missing key '{k}' in {context}")
