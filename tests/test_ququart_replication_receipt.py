import pytest

from qec.benchmark.ququart_battery.replication import (
    ReplicationReceiptError,
    V170_1_0_ARTIFACTS,
    V170_1_0_RELEASE_MANIFEST_SHA256,
    build_replication_receipt,
    qbraid_v170_1_0_receipt,
)


def test_qbraid_receipt_is_parameter_bound_and_does_not_overclaim_hashes():
    receipt = qbraid_v170_1_0_receipt()
    assert receipt["verification"]["deterministic_artifacts"] == "prefix_consistent"
    assert receipt["verification"]["sampled_artifacts"] == "parameter_variant_expected"
    assert receipt["verification"]["full_cross_environment_hash_match_claimed"] is False
    assert len(receipt["sha256"]) == 64


def test_full_hash_match_requires_equal_full_hashes():
    release_sha = V170_1_0_ARTIFACTS["exact_fer_curve.csv"]
    receipt = build_replication_receipt({
        "target": {
            "release": "170.1.0",
            "commit": "32fdf9c88f4ac1b873c8acc4d3d54c50f87d6e0a",
            "release_manifest_sha256": V170_1_0_RELEASE_MANIFEST_SHA256,
        },
        "environment": {"platform": "test"},
        "parameters": {"seed": 1},
        "source": {"document_sha256": "b" * 64},
        "artifacts": [{
            "name": "exact_fer_curve.csv",
            "kind": "deterministic",
            "release_sha256": release_sha,
            "observed_sha256": release_sha,
            "parameters_match": True,
        }],
    })
    assert receipt["verification"]["deterministic_artifacts"] == "full_hash_match"
    assert receipt["verification"]["full_cross_environment_hash_match_claimed"] is True


def test_mismatched_artifact_receipt_is_rejected():
    with pytest.raises(ReplicationReceiptError, match="mismatch"):
        build_replication_receipt({
            "target": {
                "release": "170.1.0",
                "commit": "x",
                "release_manifest_sha256": V170_1_0_RELEASE_MANIFEST_SHA256,
            },
            "environment": {},
            "parameters": {},
            "source": {"document_sha256": "b" * 64},
            "artifacts": [{
                "name": "exact_fer_curve.csv",
                "kind": "deterministic",
                "release_sha256": V170_1_0_ARTIFACTS["exact_fer_curve.csv"],
                "observed_sha256_prefix": "deadbeef",
                "parameters_match": True,
            }],
        })
