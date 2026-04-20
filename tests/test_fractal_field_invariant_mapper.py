from __future__ import annotations

import json

import pytest

from qec.analysis import fractal_field_invariant_mapper as ffim
from qec.analysis.fractal_field_invariant_mapper import map_fractal_field_invariants


def test_determinism_byte_identical_outputs() -> None:
    trajectory = (1, 2, 1, 2, 1, 2, 1, 2, 1)
    receipt_a = map_fractal_field_invariants(trajectory)
    receipt_b = map_fractal_field_invariants(list(trajectory))

    assert receipt_a.to_canonical_bytes() == receipt_b.to_canonical_bytes()
    assert receipt_a.stable_hash() == receipt_b.stable_hash()


def test_invariant_detection_correctness_and_cross_scale_logic() -> None:
    receipt = map_fractal_field_invariants((1, 2, 1, 2, 1, 2, 1, 2, 1))

    motifs = receipt.ordered_invariant_motifs
    assert motifs
    assert motifs[0]["motif"] == (1, 2)
    assert motifs[0]["scale_sizes"] == (2, 4, 8)
    assert receipt.decision["invariant_motif_count"] >= 1


def test_scale_ordering_and_profile_surface() -> None:
    receipt = map_fractal_field_invariants(("a", "b", "a", "b", "a", "b", "a", "b", "a"))
    scales = tuple(item["scale_size"] for item in receipt.ordered_scale_profiles)

    assert scales == (2, 4, 8)
    for profile in receipt.ordered_scale_profiles:
        assert set(profile.keys()) == {
            "scale_size",
            "window_count",
            "dominant_motif",
            "motif_recurrence_ratio",
            "motif_diversity",
            "scale_stability_score",
        }


def test_motif_canonicalization_rejects_non_canonical_representations() -> None:
    with pytest.raises(ValueError, match="must be non-empty tuples"):
        ffim._canonicalize_motif([1, 2])  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="homogeneous canonical type"):
        ffim._canonicalize_motif((1, "2"))  # type: ignore[arg-type]


def test_fragmentation_vs_concentration_behavior() -> None:
    concentrated = map_fractal_field_invariants((1, 1, 1, 1, 1, 1, 1, 1, 1))
    fragmented = map_fractal_field_invariants((1, 2, 3, 4, 5, 6, 7, 8, 9))

    c_score = concentrated.bounded_metric_bundle["fractal_concentration_score"]
    f_score = fragmented.bounded_metric_bundle["fractal_concentration_score"]
    assert c_score > f_score


def test_validation_failures() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        map_fractal_field_invariants(())

    with pytest.raises(ValueError, match="must be int or str"):
        map_fractal_field_invariants((1.0, 2.0))

    with pytest.raises(ValueError, match="must not be bool"):
        map_fractal_field_invariants((True, False, True))


def test_immutability_enforcement_of_precomputed_outputs() -> None:
    trajectory = (1, 2, 1, 2, 1, 2, 1, 2, 1)
    profiles = ffim._precompute_scale_profiles(trajectory)
    signatures = ffim._precompute_motif_signatures(profiles)

    with pytest.raises(AttributeError):
        profiles[0].scale_size = 100  # type: ignore[misc]
    with pytest.raises(AttributeError):
        signatures[0].signature = "tamper"  # type: ignore[misc]


def test_metric_bounds_and_receipt_requirements() -> None:
    receipt = map_fractal_field_invariants((1, 2, 1, 2, 1, 2, 1, 2, 1))

    assert receipt.release_version == "v138.5.3"
    assert receipt.experiment_kind == "fractal_field_invariant_mapper"
    assert receipt.advisory_only is True
    assert receipt.decoder_core_modified is False

    for value in receipt.bounded_metric_bundle.values():
        assert 0.0 <= value <= 1.0


def test_elimination_readiness_precomputation_called_once() -> None:
    trajectory = (1, 2, 1, 2, 1, 2, 1, 2, 1)
    counts = {"scale": 0, "motif": 0}
    original_scale = ffim._precompute_scale_profiles
    original_motif = ffim._precompute_motif_signatures

    def counted_scale(arg):
        counts["scale"] += 1
        return original_scale(arg)

    def counted_motif(arg):
        counts["motif"] += 1
        return original_motif(arg)

    ffim._precompute_scale_profiles = counted_scale
    ffim._precompute_motif_signatures = counted_motif
    try:
        receipt = map_fractal_field_invariants(trajectory)
        assert receipt.input_content_hash
        assert receipt.stable_hash() == receipt.stable_hash()
    finally:
        ffim._precompute_scale_profiles = original_scale
        ffim._precompute_motif_signatures = original_motif

    assert counts == {"scale": 1, "motif": 1}


def test_motif_signature_no_collision_for_pipe_in_string_tokens() -> None:
    """Regression: motifs containing '|' must not collide with multi-element motifs."""
    sig_combined = ffim._motif_signature(("a|b",))
    sig_separate = ffim._motif_signature(("a", "b"))
    assert sig_combined != sig_separate

    sig_abc = ffim._motif_signature(("a|b", "c"))
    sig_a_bc = ffim._motif_signature(("a", "b|c"))
    assert sig_abc != sig_a_bc


def test_canonical_json_is_compact_sorted_and_replay_safe() -> None:
    receipt = map_fractal_field_invariants((1, 2, 1, 2, 1, 2, 1, 2, 1))
    data = receipt.to_canonical_json()

    assert ", " not in data
    assert ": " not in data
    parsed = json.loads(data)
    assert parsed["release_version"] == "v138.5.3"
