"""Tests for deterministic law engine (v121.0.0)."""

import copy

from qec.analysis.law_engine import Law, LawEngine


def _always_true(_state):
    return True


def _always_false(_state):
    return False


def _tag_action(tag):
    def _action(state):
        out = dict(state)
        out["tag"] = tag
        return out

    return _action


def test_single_law_trigger():
    law = Law(
        law_id="law_single",
        priority=1,
        condition=_always_true,
        action=_tag_action("single"),
    )
    engine = LawEngine({law.law_id: law})

    state = {"x": 1}
    before = copy.deepcopy(state)
    result = engine.evaluate(state)

    assert result["law_triggered"] is True
    assert result["executed_law_id"] == "law_single"
    assert result["priority"] == 1
    assert result["action_result"] == {"x": 1, "tag": "single"}
    assert result["first_matched_law_id"] == "law_single"
    assert result["evaluation_trace"] == ("law_single",)
    assert state == before


def test_multiple_triggered_laws_highest_priority_wins():
    low = Law("law_low", 1, _always_true, _tag_action("low"))
    high = Law("law_high", 3, _always_true, _tag_action("high"))
    engine = LawEngine({low.law_id: low, high.law_id: high})

    result = engine.evaluate({"x": 1})

    assert result["law_triggered"] is True
    assert result["executed_law_id"] == "law_high"
    assert result["priority"] == 3
    assert result["action_result"]["tag"] == "high"
    assert result["first_matched_law_id"] == "law_high"
    assert result["evaluation_trace"] == ("law_high",)


def test_equal_priority_lexicographic_law_id_stable():
    zzz = Law("law_zzz", 2, _always_true, _tag_action("zzz"))
    aaa = Law("law_aaa", 2, _always_true, _tag_action("aaa"))
    engine = LawEngine({zzz.law_id: zzz, aaa.law_id: aaa})

    result = engine.evaluate({"x": 1})

    assert result["law_triggered"] is True
    assert result["executed_law_id"] == "law_aaa"
    assert result["action_result"]["tag"] == "aaa"
    assert result["first_matched_law_id"] == "law_aaa"
    assert result["evaluation_trace"] == ("law_aaa",)


def test_no_laws_triggered():
    law_a = Law("law_a", 4, _always_false, _tag_action("a"))
    law_b = Law("law_b", 2, _always_false, _tag_action("b"))
    engine = LawEngine({law_a.law_id: law_a, law_b.law_id: law_b})

    result = engine.evaluate({"x": 1})

    assert result == {
        "law_triggered": False,
        "executed_law_id": None,
        "priority": None,
        "action_result": {},
        "first_matched_law_id": None,
        "evaluation_trace": ("law_a", "law_b"),
    }


def test_disabled_law_ignored():
    disabled = Law("law_disabled", 5, _always_true, _tag_action("disabled"), enabled=False)
    enabled = Law("law_enabled", 1, _always_true, _tag_action("enabled"), enabled=True)
    engine = LawEngine({disabled.law_id: disabled, enabled.law_id: enabled})

    result = engine.evaluate({"x": 1})

    assert result["executed_law_id"] == "law_enabled"
    assert result["action_result"]["tag"] == "enabled"
    assert result["first_matched_law_id"] == "law_enabled"
    assert result["evaluation_trace"] == ("law_enabled",)


def test_deterministic_repeated_identical_outputs():
    law_a = Law("law_a", 1, _always_false, _tag_action("a"))
    law_b = Law("law_b", 2, _always_true, _tag_action("b"))
    engine = LawEngine({law_a.law_id: law_a, law_b.law_id: law_b})

    state = {"x": 9, "nested": {"y": 1}}
    results = [engine.evaluate(state) for _ in range(5)]

    for res in results[1:]:
        assert res == results[0]


def test_evaluation_trace_stability():
    law_c = Law("law_c", 2, _always_false, _tag_action("c"))
    law_b = Law("law_b", 2, _always_false, _tag_action("b"))
    law_a = Law("law_a", 2, _always_false, _tag_action("a"))
    engine = LawEngine({law_c.law_id: law_c, law_b.law_id: law_b, law_a.law_id: law_a})

    result_one = engine.evaluate({"x": 1})
    result_two = engine.evaluate({"x": 1})

    assert result_one["evaluation_trace"] == ("law_a", "law_b", "law_c")
    assert result_two["evaluation_trace"] == ("law_a", "law_b", "law_c")


def test_first_match_short_circuit_preserved():
    first = Law("law_a", 3, _always_true, _tag_action("a"))

    def _must_not_execute(_state):
        raise AssertionError("short-circuit violated")

    second = Law("law_b", 1, _must_not_execute, _tag_action("b"))
    engine = LawEngine({first.law_id: first, second.law_id: second})

    result = engine.evaluate({"x": 1})

    assert result["executed_law_id"] == "law_a"
    assert result["first_matched_law_id"] == "law_a"
    assert result["evaluation_trace"] == ("law_a",)
