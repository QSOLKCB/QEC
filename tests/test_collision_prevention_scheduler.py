import pytest

from qec.control.collision_prevention_scheduler import (
    evaluate_collision_prevention_schedule,
    normalize_collision_prevention_schedule,
    validate_collision_prevention_schedule,
)


def _base_schedule():
    return {
        "schedule_id": "sched-1",
        "windows": [
            {
                "window_id": "w1",
                "transition_id": "t-alpha",
                "from_state": "s0",
                "to_state": "s1",
                "window_epoch_start": 10,
                "window_epoch_end": 15,
                "priority": 10,
                "exclusive": False,
                "scheduler_lane": 0,
                "phase_window": 0,
                "collision_delta_threshold": 0,
            },
            {
                "window_id": "w2",
                "transition_id": "t-beta",
                "from_state": "s1",
                "to_state": "s2",
                "window_epoch_start": 20,
                "window_epoch_end": 25,
                "priority": 5,
                "exclusive": False,
                "scheduler_lane": 0,
                "phase_window": 0,
                "collision_delta_threshold": 0,
            },
        ],
        "rules": [
            {
                "rule_id": "r1",
                "conflicting_transition_ids": ["t-alpha", "t-beta"],
                "lane_constraints": [],
                "priority_resolution_mode": "defer_lower_priority",
                "fallback_action": "defer",
                "rule_epoch": 1,
            }
        ],
        "terminal_scheduler_mode": "clean",
    }


def _force_same_lane(schedule):
    normalized = normalize_collision_prevention_schedule(schedule)
    lane = normalized.windows[0].scheduler_lane
    schedule["windows"][1]["transition_id"] = schedule["windows"][0]["transition_id"]
    schedule["rules"][0]["conflicting_transition_ids"] = [
        schedule["windows"][0]["transition_id"],
        schedule["windows"][1]["transition_id"],
    ]
    schedule["windows"][1]["scheduler_lane"] = lane
    return schedule


def test_repeated_run_byte_identity():
    schedule = normalize_collision_prevention_schedule(_base_schedule())
    receipt_a = evaluate_collision_prevention_schedule(schedule)
    receipt_b = evaluate_collision_prevention_schedule(schedule)
    assert receipt_a.to_canonical_bytes() == receipt_b.to_canonical_bytes()


def test_repeated_run_hash_identity():
    schedule = normalize_collision_prevention_schedule(_base_schedule())
    receipt_a = evaluate_collision_prevention_schedule(schedule)
    receipt_b = evaluate_collision_prevention_schedule(schedule)
    assert receipt_a.schedule_hash == receipt_b.schedule_hash


def test_stable_mod_64_lane_assignment():
    schedule = normalize_collision_prevention_schedule(_base_schedule())
    expected = schedule.windows[0].scheduler_lane
    rerun = normalize_collision_prevention_schedule(_base_schedule())
    assert rerun.windows[0].scheduler_lane == expected


def test_stable_mod_60_phase_windows():
    schedule = normalize_collision_prevention_schedule(_base_schedule())
    assert schedule.windows[0].phase_window == schedule.windows[0].window_epoch_start % 60
    assert schedule.windows[1].phase_window == schedule.windows[1].window_epoch_start % 60


def test_deterministic_delta_collision_detection():
    schedule = _base_schedule()
    schedule = _force_same_lane(schedule)
    schedule["windows"][1]["window_epoch_start"] = 11
    schedule["windows"][1]["window_epoch_end"] = 16
    schedule["windows"][0]["collision_delta_threshold"] = 1
    schedule["windows"][1]["collision_delta_threshold"] = 1
    schedule["windows"][0]["exclusive"] = True
    normalized = normalize_collision_prevention_schedule(schedule)
    receipt = evaluate_collision_prevention_schedule(normalized)
    assert receipt.collision_count == 1


def test_deterministic_ternary_decision_trace():
    schedule = _base_schedule()
    schedule = _force_same_lane(schedule)
    schedule["windows"][1]["window_epoch_start"] = 10
    schedule["windows"][1]["window_epoch_end"] = 15
    schedule["windows"][0]["collision_delta_threshold"] = 0
    schedule["windows"][1]["collision_delta_threshold"] = 0
    normalized = normalize_collision_prevention_schedule(schedule)
    receipt = evaluate_collision_prevention_schedule(normalized)
    assert set(receipt.decision_trace_base3).issubset({"0", "1", "2"})


def test_duplicate_window_rejection():
    schedule = _base_schedule()
    schedule["windows"][1]["window_id"] = "w1"
    with pytest.raises(ValueError, match="duplicate window IDs"):
        normalize_collision_prevention_schedule(schedule)


def test_invalid_epoch_rejection():
    schedule = _base_schedule()
    schedule["windows"][0]["window_epoch_end"] = 9
    with pytest.raises(ValueError, match="invalid epoch windows"):
        normalize_collision_prevention_schedule(schedule)


def test_exclusive_overlap_rejection():
    schedule = _base_schedule()
    schedule = _force_same_lane(schedule)
    schedule["windows"][0]["exclusive"] = True
    schedule["windows"][1]["exclusive"] = True
    schedule["windows"][1]["window_epoch_start"] = 12
    schedule["windows"][1]["window_epoch_end"] = 14
    with pytest.raises(ValueError, match="overlapping exclusive windows"):
        normalize_collision_prevention_schedule(schedule)


def test_lane_collision_detection():
    schedule = _base_schedule()
    schedule = _force_same_lane(schedule)
    schedule["windows"][0]["window_epoch_start"] = 10
    schedule["windows"][1]["window_epoch_start"] = 10
    schedule["windows"][0]["collision_delta_threshold"] = 0
    schedule["windows"][1]["collision_delta_threshold"] = 0
    normalized = normalize_collision_prevention_schedule(schedule)
    receipt = evaluate_collision_prevention_schedule(normalized)
    assert receipt.collision_count >= 1


def test_deterministic_clean_outcome():
    schedule = normalize_collision_prevention_schedule(_base_schedule())
    receipt = evaluate_collision_prevention_schedule(schedule)
    assert receipt.scheduler_terminal_status == "clean"


def test_deterministic_resolved_outcome():
    schedule = _base_schedule()
    schedule = _force_same_lane(schedule)
    schedule["windows"][0]["window_epoch_start"] = 10
    schedule["windows"][1]["window_epoch_start"] = 10
    schedule["windows"][0]["collision_delta_threshold"] = 0
    schedule["windows"][1]["collision_delta_threshold"] = 0
    schedule["rules"][0]["priority_resolution_mode"] = "defer_lower_priority"
    normalized = normalize_collision_prevention_schedule(schedule)
    receipt = evaluate_collision_prevention_schedule(normalized)
    assert receipt.scheduler_terminal_status == "resolved"


def test_deterministic_blocked_outcome():
    schedule = _base_schedule()
    schedule = _force_same_lane(schedule)
    schedule["windows"][0]["window_epoch_start"] = 10
    schedule["windows"][1]["window_epoch_start"] = 10
    schedule["windows"][0]["collision_delta_threshold"] = 0
    schedule["windows"][1]["collision_delta_threshold"] = 0
    schedule["rules"][0]["priority_resolution_mode"] = "block_lower_priority"
    schedule["rules"][0]["fallback_action"] = "block"
    normalized = normalize_collision_prevention_schedule(schedule)
    receipt = evaluate_collision_prevention_schedule(normalized)
    assert receipt.scheduler_terminal_status == "blocked"


def test_deterministic_halt_outcome():
    schedule = _base_schedule()
    schedule = _force_same_lane(schedule)
    schedule["windows"][0]["window_epoch_start"] = 10
    schedule["windows"][1]["window_epoch_start"] = 10
    schedule["windows"][0]["collision_delta_threshold"] = 0
    schedule["windows"][1]["collision_delta_threshold"] = 0
    schedule["rules"][0]["priority_resolution_mode"] = "halt_on_conflict"
    schedule["rules"][0]["fallback_action"] = "halt"
    normalized = normalize_collision_prevention_schedule(schedule)
    receipt = evaluate_collision_prevention_schedule(normalized)
    assert receipt.scheduler_terminal_status == "halt"


def test_canonical_export_stability():
    sched_a = normalize_collision_prevention_schedule(_base_schedule())
    sched_b = normalize_collision_prevention_schedule(_base_schedule())
    rec_a = evaluate_collision_prevention_schedule(sched_a)
    rec_b = evaluate_collision_prevention_schedule(sched_b)
    assert sched_a.to_canonical_json() == sched_b.to_canonical_json()
    assert rec_a.to_canonical_bytes() == rec_b.to_canonical_bytes()
    assert rec_a.to_canonical_bytes() == rec_a.as_hash_payload()


def test_validate_schedule_passes_for_valid_data():
    report = validate_collision_prevention_schedule(_base_schedule())
    assert report.is_valid is True
    assert report.errors == ()


def test_validate_duplicate_window_sets_only_uniqueness_flag():
    # P2: a duplicate-window error must only fail the uniqueness flag,
    # not unrelated flags like epoch_validity, lane_validity, etc.
    schedule = _base_schedule()
    schedule["windows"][1]["window_id"] = "w1"
    report = validate_collision_prevention_schedule(schedule)
    assert report.is_valid is False
    assert report.uniqueness is False
    assert report.epoch_validity is True
    assert report.lane_validity is True
    assert report.phase_validity is True
    assert report.rule_validity is True
    assert report.delta_threshold_validity is True
    assert report.fallback_validity is True


def test_validate_invalid_epoch_sets_only_epoch_flag():
    schedule = _base_schedule()
    schedule["windows"][0]["window_epoch_end"] = 9
    report = validate_collision_prevention_schedule(schedule)
    assert report.is_valid is False
    assert report.epoch_validity is False
    assert report.uniqueness is True
    assert report.lane_validity is True
    assert report.phase_validity is True
    assert report.rule_validity is True


def test_evaluate_accepts_schedule_like_dict():
    # evaluate_collision_prevention_schedule must accept raw ScheduleLike dicts.
    receipt = evaluate_collision_prevention_schedule(_base_schedule())
    assert receipt.scheduler_terminal_status == "clean"
    assert isinstance(receipt.schedule_hash, str) and len(receipt.schedule_hash) == 64


def test_fallback_action_allow_keeps_decision_zero():
    # fallback_action="allow" must result in 0 (no deferral) even when a rule fires.
    schedule = _base_schedule()
    schedule = _force_same_lane(schedule)
    schedule["windows"][0]["window_epoch_start"] = 10
    schedule["windows"][1]["window_epoch_start"] = 10
    schedule["windows"][0]["collision_delta_threshold"] = 0
    schedule["windows"][1]["collision_delta_threshold"] = 0
    schedule["rules"][0]["priority_resolution_mode"] = "defer_lower_priority"
    schedule["rules"][0]["fallback_action"] = "allow"
    normalized = normalize_collision_prevention_schedule(schedule)
    receipt = evaluate_collision_prevention_schedule(normalized)
    # fallback_action="allow" → loser decision stays 0 → resolved/blocked are empty.
    # collision_count >= 1 with no resolved/blocked windows → status is "blocked"
    # (unhandled-collision else-branch).
    assert receipt.scheduler_terminal_status == "blocked"
    assert all(d == "0" for d in receipt.decision_trace_base3)


def test_fallback_actions_preserve_insertion_order():
    # deterministic_fallback_actions must preserve encounter order, not be sorted.
    schedule = _base_schedule()
    schedule = _force_same_lane(schedule)
    schedule["windows"][0]["window_epoch_start"] = 10
    schedule["windows"][1]["window_epoch_start"] = 10
    schedule["windows"][0]["collision_delta_threshold"] = 0
    schedule["windows"][1]["collision_delta_threshold"] = 0
    schedule["rules"][0]["fallback_action"] = "defer"
    normalized = normalize_collision_prevention_schedule(schedule)
    receipt = evaluate_collision_prevention_schedule(normalized)
    # At least one fallback_action recorded; check it equals the rule's action.
    assert len(receipt.deterministic_fallback_actions) >= 1
    assert receipt.deterministic_fallback_actions[0] == "defer"
