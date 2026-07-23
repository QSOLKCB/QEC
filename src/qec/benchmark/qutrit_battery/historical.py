"""Immutable v3-era baseline loading and numeric overlays."""

from __future__ import annotations

import csv
import hashlib
import io
from decimal import Decimal
from pathlib import Path

from .curves import decimal_text

V3_HEADER = (
    "error_rate",
    "steane",
    "surface",
    "reed_muller",
    "fusion_qec_photonic",
)
V3_FILE_SHA256 = "80f1f74ad02c2ac7fdaf5e6a6df1611f55df3f4294cc11edfc889f5d7fe41b0a"
V3_GIT_BLOB = "b4a3d4a9f9bb8de9b2ba391406f269b27c6715dc"
V3_FIRST_TAG = "v3.0.0"
V3_LAST_TAG = "v3.9.1"
V3_FIRST_COMMIT = "300b5551d684dc4ec1a430ebe570b77d0b52df39"
V3_LAST_COMMIT = "7deff2e7cdc925fecfceb5c0d2f4f64039076489"


def load_v3_baseline(path: Path) -> tuple[bytes, list[dict[str, str]]]:
    """Load only the byte-identical baseline tracked through the v3 tags."""
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != V3_FILE_SHA256:
        raise ValueError("immutable v3 baseline SHA-256 mismatch")
    reader = csv.DictReader(io.StringIO(payload.decode("utf-8")))
    if tuple(reader.fieldnames or ()) != V3_HEADER:
        raise ValueError("immutable v3 baseline header mismatch")
    rows = [dict(row) for row in reader]
    if len(rows) != 4:
        raise ValueError("immutable v3 baseline row count mismatch")
    return payload, rows


def overlay_rows(
    v3_rows: list[dict[str, str]],
    decoded_rows: list[dict[str, str | int]],
    bound_rows: list[dict[str, str | int]],
) -> list[dict[str, str]]:
    def key(value: str) -> str:
        return str(Decimal(value).normalize())

    decoded = {
        (key(str(row["error_rate"])), str(row["code_id"])): str(
            row["logical_failure_rate"]
        )
        for row in decoded_rows
    }
    bounds = {
        (key(str(row["error_rate"])), str(row["code_id"])): str(
            row["uncorrectable_weight_tail"]
        )
        for row in bound_rows
    }
    result = []
    for old in v3_rows:
        p = old["error_rate"]
        lookup_p = key(p)
        result.append({
            "error_rate": p,
            "v3_steane_illustrative": old["steane"],
            "v3_surface_illustrative": old["surface"],
            "v3_reed_muller_illustrative": old["reed_muller"],
            "v3_fusion_qec_photonic_illustrative": old["fusion_qec_photonic"],
            "new_qutrit_cyclic_exact": decoded[(lookup_p, "qutrit_cyclic_5")],
            "new_qutrit_shor_exact": decoded[(lookup_p, "qutrit_shor_9")],
            "new_qutrit_golay_upper_bound": bounds[
                (lookup_p, "qutrit_golay_11")
            ],
            "comparison_scope": "numeric_overlay_only_models_not_comparable",
        })
    return result


def numeric_delta_rows(overlays: list[dict[str, str]]) -> list[dict[str, str]]:
    """Compute transparent ratios without interpreting them as performance."""
    old_columns = (
        "v3_steane_illustrative",
        "v3_surface_illustrative",
        "v3_reed_muller_illustrative",
        "v3_fusion_qec_photonic_illustrative",
    )
    new_columns = (
        "new_qutrit_cyclic_exact",
        "new_qutrit_shor_exact",
        "new_qutrit_golay_upper_bound",
    )
    rows = []
    for overlay in overlays:
        for new_column in new_columns:
            new_value = Decimal(overlay[new_column])
            for old_column in old_columns:
                old_value = Decimal(overlay[old_column])
                rows.append({
                    "error_rate": overlay["error_rate"],
                    "new_metric": new_column,
                    "new_value": overlay[new_column],
                    "v3_metric": old_column,
                    "v3_value": overlay[old_column],
                    "numeric_ratio_new_over_v3": decimal_text(
                        new_value / old_value
                    ),
                    "interpretation": "none_models_and_claim_scopes_differ",
                })
    return rows


def lineage_record() -> dict[str, str | list[str]]:
    return {
        "baseline_role": "immutable_historical_reference",
        "source_path": "qec_data_prepared.csv",
        "file_sha256": V3_FILE_SHA256,
        "git_blob_sha1": V3_GIT_BLOB,
        "verified_tag_range": [V3_FIRST_TAG, V3_LAST_TAG],
        "tag_commits": [V3_FIRST_COMMIT, V3_LAST_COMMIT],
        "repository_scope": "QSOLKCB/QEC",
        "claim_scope": (
            "Historical illustrative/research-aligned values; not "
            "device-calibrated measurements."
        ),
    }
