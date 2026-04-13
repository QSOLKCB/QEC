import pytest

from qec.control.bounded_fallback_corridor import (
    evaluate_bounded_fallback_corridor,
    normalize_bounded_fallback_corridor,
    validate_bounded_fallback_corridor,
)


def _base_corridor():
    return {
        "corridor_id": "corr-1",
        "segments": [
            {
                "segment_id": "seg-recover",
                "from_state": "blocked",
                "to_state": "safe",
                "max_depth": 4,
                "allowed_lanes": [2, 3],
                "rollback_limit": 3,
                "priority": 1,
                "segment_epoch": 1,
            }
        ],
        "terminal_fallback_state": "safe-terminal",
        "terminal_policy": "prefer_recover",
        "max_total_depth": 5,
        "max_attempts": 5,
    }


def _base_context():
    return {
        "context_id": "ctx-1",
        "source_failure_state": "blocked",
        "collision_receipt_id": "col-1",
        "rollback_receipt_id": "rb-1",
        "active_lane": 2,
        "current_depth": 1,
        "attempt_count": 1,
    }


def test_repeated_run_byte_identity():
    corridor = normalize_bounded_fallback_corridor(_base_corridor())
    receipt_a = evaluate_bounded_fallback_corridor(corridor, _base_context())
    receipt_b = evaluate_bounded_fallback_corridor(corridor, _base_context())
    assert receipt_a.to_canonical_bytes() == receipt_b.to_canonical_bytes()


def test_repeated_run_hash_identity():
    corridor = normalize_bounded_fallback_corridor(_base_corridor())
    receipt_a = evaluate_bounded_fallback_corridor(corridor, _base_context())
    receipt_b = evaluate_bounded_fallback_corridor(corridor, _base_context())
    assert receipt_a.corridor_hash == receipt_b.corridor_hash


def test_duplicate_segment_rejection():
    corridor = _base_corridor()
    corridor["segments"].append(dict(corridor["segments"][0]))
    with pytest.raises(ValueError, match="duplicate segment IDs"):
        normalize_bounded_fallback_corridor(corridor)


def test_invalid_depth_rejection():
    corridor = _base_corridor()
    corridor["segments"][0]["max_depth"] = -1
    with pytest.raises(ValueError, match="negative depth"):
        normalize_bounded_fallback_corridor(corridor)


def test_invalid_lane_rejection():
    corridor = _base_corridor()
    corridor["segments"][0]["allowed_lanes"] = [2, -4]
    with pytest.raises(ValueError, match="invalid lane lists"):
        normalize_bounded_fallback_corridor(corridor)


def test_deterministic_recover_outcome():
    receipt = evaluate_bounded_fallback_corridor(_base_corridor(), _base_context())
    assert receipt.terminal_decision == "recover"
    assert receipt.bounded_stop_reason == "recoverable_segment_found"


def test_deterministic_rollback_outcome():
    corridor = _base_corridor()
    corridor["segments"][0]["rollback_limit"] = 1
    context = _base_context()
    context["attempt_count"] = 1
    receipt = evaluate_bounded_fallback_corridor(corridor, context)
    assert receipt.terminal_decision == "rollback"
    assert receipt.bounded_stop_reason == "rollback_required"


def test_deterministic_halt_outcome():
    corridor = _base_corridor()
    corridor["terminal_policy"] = "halt_on_exhaustion"
    corridor["segments"][0]["allowed_lanes"] = [9]
    context = _base_context()
    context["attempt_count"] = corridor["max_attempts"]
    receipt = evaluate_bounded_fallback_corridor(corridor, context)
    assert receipt.terminal_decision == "halt"
    assert receipt.bounded_stop_reason == "attempt_bound_reached"


def test_deterministic_terminal_fallback_outcome():
    corridor = _base_corridor()
    corridor["segments"][0]["allowed_lanes"] = [9]
    corridor["terminal_policy"] = "force_terminal_fallback"
    receipt = evaluate_bounded_fallback_corridor(corridor, _base_context())
    assert receipt.terminal_decision == "terminal_fallback"
    assert receipt.bounded_stop_reason == "terminal_policy_enforced"


def test_canonical_export_stability():
    corridor_a = normalize_bounded_fallback_corridor(_base_corridor())
    corridor_b = normalize_bounded_fallback_corridor(_base_corridor())

    rec_a = evaluate_bounded_fallback_corridor(corridor_a, _base_context())
    rec_b = evaluate_bounded_fallback_corridor(corridor_b, _base_context())

    assert corridor_a.to_canonical_json() == corridor_b.to_canonical_json()
    assert rec_a.to_canonical_bytes() == rec_b.to_canonical_bytes()
    assert rec_a.as_hash_payload() == rec_a.to_canonical_bytes()


def test_validate_valid_corridor_passes():
    report = validate_bounded_fallback_corridor(_base_corridor())
    assert report.is_valid is True
    assert report.errors == ()


def test_allowed_lanes_order_invariance():
    corridor_a = _base_corridor()
    corridor_b = _base_corridor()
    corridor_b["segments"][0]["allowed_lanes"] = [3, 2]

    norm_a = normalize_bounded_fallback_corridor(corridor_a)
    norm_b = normalize_bounded_fallback_corridor(corridor_b)

    assert norm_a.segments[0].allowed_lanes == norm_b.segments[0].allowed_lanes
    assert norm_a.to_canonical_bytes() == norm_b.to_canonical_bytes()


def test_allowed_lanes_hash_identity_reordered():
    corridor_a = _base_corridor()
    corridor_b = _base_corridor()
    corridor_b["segments"][0]["allowed_lanes"] = [3, 2]

    norm_a = normalize_bounded_fallback_corridor(corridor_a)
    norm_b = normalize_bounded_fallback_corridor(corridor_b)

    assert norm_a.as_hash_payload() == norm_b.as_hash_payload()


def test_allowed_lanes_sorted_across_multiple_segments():
    corridor = _base_corridor()
    corridor["segments"].append(
        {
            "segment_id": "seg-2",
            "from_state": "blocked",
            "to_state": "safe",
            "max_depth": 3,
            "allowed_lanes": [5, 1, 3],
            "rollback_limit": 2,
            "priority": 2,
            "segment_epoch": 1,
        }
    )
    norm = normalize_bounded_fallback_corridor(corridor)
    for seg in norm.segments:
        assert list(seg.allowed_lanes) == sorted(seg.allowed_lanes)


def test_malformed_none_segments_returns_invalid_report():
    corridor = _base_corridor()
    corridor["segments"] = None
    report = validate_bounded_fallback_corridor(corridor)
    assert report.is_valid is False


def test_malformed_none_lanes_returns_invalid_report():
    corridor = _base_corridor()
    corridor["segments"][0]["allowed_lanes"] = None
    report = validate_bounded_fallback_corridor(corridor)
    assert report.is_valid is False


def test_force_terminal_fallback_overrides_matching_segment():
    """force_terminal_fallback must override even when a segment would match."""
    corridor = _base_corridor()
    corridor["terminal_policy"] = "force_terminal_fallback"
    # _base_context has active_lane=2 and source_failure_state="blocked" — segment matches normally
    receipt = evaluate_bounded_fallback_corridor(corridor, _base_context())
    assert receipt.terminal_decision == "terminal_fallback"
    assert receipt.bounded_stop_reason == "terminal_policy_enforced"


def test_validate_duplicate_segment_flag():
    corridor = _base_corridor()
    corridor["segments"].append(dict(corridor["segments"][0]))
    report = validate_bounded_fallback_corridor(corridor)
    assert report.is_valid is False
    assert report.uniqueness is False


def test_validate_negative_depth_flag():
    corridor = _base_corridor()
    corridor["segments"][0]["max_depth"] = -1
    report = validate_bounded_fallback_corridor(corridor)
    assert report.is_valid is False
    assert report.depth_validity is False
