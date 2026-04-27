from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from qec.analysis.canonical_hashing import sha256_hex
from qec.analysis.cross_environment_replay_kernel import (
    CrossEnvironmentReplayReceipt,
    EnvironmentReplayArtifact,
    compare_cross_environment_replay,
)


def _h(label: str) -> str:
    return sha256_hex({"label": label})


def _artifact(
    environment_id: str,
    *,
    workload_id: str = "workload-A",
    artifact_hash: str = "artifact-A",
    canonical_payload_hash: str = "payload-A",
    receipt_hash: str = "receipt-A",
    platform_label: str = "linux",
    python_label: str = "3.12.3",
    metadata_hash: str = "metadata-A",
) -> EnvironmentReplayArtifact:
    return EnvironmentReplayArtifact(
        environment_id=environment_id,
        workload_id=workload_id,
        artifact_hash=_h(artifact_hash),
        canonical_payload_hash=_h(canonical_payload_hash),
        receipt_hash=_h(receipt_hash),
        platform_label=platform_label,
        python_label=python_label,
        metadata_hash=_h(metadata_hash),
    )


def test_cross_environment_match() -> None:
    artifacts = (
        _artifact("env-b"),
        _artifact("env-a"),
        _artifact("env-c"),
    )

    receipt = compare_cross_environment_replay(artifacts)

    assert receipt.receipt_status == "CROSS_ENV_MATCH"
    assert receipt.comparison.comparison_status == "MATCH"
    assert receipt.comparison.mismatch_classification == "NONE"
    assert receipt.comparison.reference_environment_id == "env-a"
    assert receipt.comparison.matching_environment_ids == ("env-a", "env-b", "env-c")
    assert receipt.comparison.mismatching_environment_ids == ()
    assert receipt.determinism_preserved is True
    assert receipt.failure_recorded is False
    assert receipt.failure_classified is True


def test_artifact_hash_mismatch_classification() -> None:
    receipt = compare_cross_environment_replay(
        (
            _artifact("env-a"),
            _artifact("env-b", artifact_hash="artifact-B"),
        )
    )

    assert receipt.receipt_status == "CROSS_ENV_MISMATCH"
    assert receipt.comparison.mismatch_classification == "ARTIFACT_HASH_MISMATCH"
    assert receipt.comparison.mismatching_environment_ids == ("env-b",)


def test_canonical_payload_mismatch_classification() -> None:
    receipt = compare_cross_environment_replay(
        (
            _artifact("env-a"),
            _artifact("env-b", canonical_payload_hash="payload-B"),
        )
    )

    assert receipt.comparison.mismatch_classification == "CANONICAL_PAYLOAD_MISMATCH"


def test_receipt_hash_mismatch_classification() -> None:
    receipt = compare_cross_environment_replay(
        (
            _artifact("env-a"),
            _artifact("env-b", receipt_hash="receipt-B"),
        )
    )

    assert receipt.comparison.mismatch_classification == "RECEIPT_HASH_MISMATCH"


def test_mixed_mismatch_classification() -> None:
    receipt = compare_cross_environment_replay(
        (
            _artifact("env-a"),
            _artifact("env-b", artifact_hash="artifact-B", receipt_hash="receipt-B"),
        )
    )

    assert receipt.comparison.mismatch_classification == "MIXED_HASH_MISMATCH"


def test_workload_id_mismatch() -> None:
    receipt = compare_cross_environment_replay(
        (
            _artifact("env-b", workload_id="workload-B"),
            _artifact("env-a", workload_id="workload-A"),
        )
    )

    assert receipt.receipt_status == "CROSS_ENV_MISMATCH"
    assert receipt.comparison.comparison_status == "MISMATCH"
    assert receipt.comparison.mismatch_classification == "WORKLOAD_ID_MISMATCH"
    assert receipt.determinism_preserved is False
    assert receipt.failure_recorded is True
    assert receipt.failure_classified is True


def test_insufficient_environments() -> None:
    receipt = compare_cross_environment_replay((_artifact("env-a"),))

    assert receipt.receipt_status == "INSUFFICIENT_ENVIRONMENTS"
    assert receipt.comparison.comparison_status == "INSUFFICIENT"
    assert receipt.comparison.mismatch_classification == "INSUFFICIENT_ENVIRONMENTS"
    assert receipt.failure_recorded is True
    assert receipt.failure_classified is True


def test_input_order_stability() -> None:
    first = compare_cross_environment_replay(
        (
            _artifact("env-c", artifact_hash="same"),
            _artifact("env-a", artifact_hash="same"),
            _artifact("env-b", artifact_hash="diff"),
        )
    )
    second = compare_cross_environment_replay(
        (
            _artifact("env-b", artifact_hash="diff"),
            _artifact("env-c", artifact_hash="same"),
            _artifact("env-a", artifact_hash="same"),
        )
    )

    assert first.to_canonical_json() == second.to_canonical_json()
    assert first.stable_hash == second.stable_hash


def test_metadata_labels_do_not_affect_equality() -> None:
    receipt = compare_cross_environment_replay(
        (
            _artifact("env-a", platform_label="linux", python_label="3.12.3", metadata_hash="meta-a"),
            _artifact("env-b", platform_label="windows", python_label="3.11.9", metadata_hash="meta-b"),
        )
    )

    assert receipt.receipt_status == "CROSS_ENV_MATCH"
    assert receipt.comparison.mismatch_classification == "NONE"


def test_invalid_hash_rejected() -> None:
    with pytest.raises(ValueError, match="artifact_hash must be a valid SHA-256 hex string"):
        EnvironmentReplayArtifact(
            environment_id="env-a",
            workload_id="w",
            artifact_hash="not-a-hash",
            canonical_payload_hash=_h("payload"),
            receipt_hash=_h("receipt"),
            platform_label="linux",
            python_label="3.12",
            metadata_hash=_h("meta"),
        )


def test_frozen_dataclass_immutability() -> None:
    artifact = _artifact("env-a")
    receipt = compare_cross_environment_replay((_artifact("env-a"), _artifact("env-b")))

    with pytest.raises(FrozenInstanceError):
        artifact.environment_id = "env-x"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        receipt.receipt_status = "INVALID_INPUT"  # type: ignore[misc]


def test_canonical_json_and_hash_stability() -> None:
    receipt_one = compare_cross_environment_replay((_artifact("env-a"), _artifact("env-b")))
    receipt_two = compare_cross_environment_replay((_artifact("env-a"), _artifact("env-b")))

    assert isinstance(receipt_one, CrossEnvironmentReplayReceipt)
    assert receipt_one.to_canonical_json() == receipt_two.to_canonical_json()
    assert receipt_one.to_canonical_bytes() == receipt_two.to_canonical_bytes()
    assert receipt_one.stable_hash == receipt_two.stable_hash


def test_invalid_input_rejected() -> None:
    with pytest.raises(ValueError, match="artifacts must not be empty"):
        compare_cross_environment_replay(())
