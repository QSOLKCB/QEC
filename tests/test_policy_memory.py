"""Tests for policy memory, replay, and persistence (v103.7.0).

Verifies:
- deterministic storage and retrieval
- correct averaging
- ranking correctness
- replay correctness
- no mutation of inputs
- export/import round-trip
- integration with meta-control via strategy_adapter
"""

from __future__ import annotations

import copy
import json

from qec.analysis.policy import Policy, get_policy
from qec.analysis.policy_memory import (
    DEFAULT_TOP_K,
    ROUND_PRECISION,
    export_policy_memory,
    format_policy_memory_summary,
    get_top_policies,
    import_policy_memory,
    init_policy_memory,
    replay_policies,
    update_policy_memory,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_policy(name: str, priority: str = "balanced") -> Policy:
    """Build a minimal policy."""
    return Policy(
        name=name,
        mode="hybrid",
        priority=priority,
        thresholds={"instability": 0.5, "sync": 0.5},
    )


def _make_state_vector(stability_state=0, strong_attractor=0.3,
                       weak_attractor=0.2, transient=0.1,
                       basin=0.2, neutral=0.2):
    """Build a minimal state vector."""
    return {
        "ternary": {
            "stability_state": stability_state,
            "trend_state": 0,
            "phase_state": 0,
        },
        "membership": {
            "strong_attractor": strong_attractor,
            "weak_attractor": weak_attractor,
            "transient": transient,
            "basin": basin,
            "neutral": neutral,
        },
    }


def _make_multistate(*names, stability_state=0):
    """Build a multistate dict from strategy names."""
    return {
        name: _make_state_vector(stability_state=stability_state)
        for name in names
    }


def _make_run(strategies):
    """Build a run dict from a list of (name, design_score) tuples."""
    return {
        "strategies": [
            {
                "name": name,
                "metrics": {
                    "design_score": score,
                    "confidence_efficiency": 0.5,
                    "consistency_gap": 0.1,
                    "revival_strength": 0.0,
                },
            }
            for name, score in strategies
        ],
    }


# ---------------------------------------------------------------------------
# TestInitPolicyMemory
# ---------------------------------------------------------------------------


class TestInitPolicyMemory:
    """Tests for init_policy_memory."""

    def test_returns_empty_structure(self):
        mem = init_policy_memory()
        assert mem == {"policies": {}}

    def test_returns_new_dict_each_call(self):
        a = init_policy_memory()
        b = init_policy_memory()
        assert a is not b

    def test_deterministic(self):
        assert init_policy_memory() == init_policy_memory()


# ---------------------------------------------------------------------------
# TestUpdatePolicyMemory
# ---------------------------------------------------------------------------


class TestUpdatePolicyMemory:
    """Tests for update_policy_memory."""

    def test_add_new_policy(self):
        mem = init_policy_memory()
        p = _make_policy("alpha")
        result = update_policy_memory(mem, p, 0.8)

        assert "alpha" in result["policies"]
        entry = result["policies"]["alpha"]
        assert entry["scores"] == [0.8]
        assert entry["avg_score"] == 0.8
        assert entry["uses"] == 1

    def test_update_existing_policy(self):
        mem = init_policy_memory()
        p = _make_policy("alpha")
        mem = update_policy_memory(mem, p, 0.8)
        mem = update_policy_memory(mem, p, 0.6)

        entry = mem["policies"]["alpha"]
        assert entry["scores"] == [0.8, 0.6]
        assert entry["avg_score"] == round((0.8 + 0.6) / 2, ROUND_PRECISION)
        assert entry["uses"] == 2

    def test_no_mutation_of_input(self):
        mem = init_policy_memory()
        original = copy.deepcopy(mem)
        p = _make_policy("alpha")
        update_policy_memory(mem, p, 0.9)
        assert mem == original

    def test_multiple_policies(self):
        mem = init_policy_memory()
        mem = update_policy_memory(mem, _make_policy("a"), 0.5)
        mem = update_policy_memory(mem, _make_policy("b"), 0.9)
        mem = update_policy_memory(mem, _make_policy("a"), 0.7)

        assert len(mem["policies"]) == 2
        assert mem["policies"]["a"]["uses"] == 2
        assert mem["policies"]["b"]["uses"] == 1

    def test_deterministic(self):
        """Same sequence of updates produces identical memory."""
        def build():
            mem = init_policy_memory()
            mem = update_policy_memory(mem, _make_policy("x"), 0.3)
            mem = update_policy_memory(mem, _make_policy("y"), 0.7)
            mem = update_policy_memory(mem, _make_policy("x"), 0.5)
            return mem

        assert build() == build()

    def test_stores_policy_dict(self):
        mem = init_policy_memory()
        p = _make_policy("alpha", priority="stability")
        mem = update_policy_memory(mem, p, 0.8)
        stored = mem["policies"]["alpha"]["policy_dict"]
        assert stored == p.to_dict()


# ---------------------------------------------------------------------------
# TestGetTopPolicies
# ---------------------------------------------------------------------------


class TestGetTopPolicies:
    """Tests for get_top_policies."""

    def test_empty_memory_returns_empty(self):
        assert get_top_policies(init_policy_memory()) == []

    def test_single_policy(self):
        mem = init_policy_memory()
        mem = update_policy_memory(mem, _make_policy("only"), 0.5)
        top = get_top_policies(mem, k=3)
        assert len(top) == 1
        assert top[0].name == "only"

    def test_ranking_by_avg_score(self):
        mem = init_policy_memory()
        mem = update_policy_memory(mem, _make_policy("low"), 0.3)
        mem = update_policy_memory(mem, _make_policy("high"), 0.9)
        mem = update_policy_memory(mem, _make_policy("mid"), 0.6)
        top = get_top_policies(mem, k=3)
        names = [p.name for p in top]
        assert names == ["high", "mid", "low"]

    def test_tiebreak_by_uses(self):
        mem = init_policy_memory()
        mem = update_policy_memory(mem, _make_policy("a"), 0.5)
        mem = update_policy_memory(mem, _make_policy("b"), 0.5)
        mem = update_policy_memory(mem, _make_policy("b"), 0.5)
        top = get_top_policies(mem, k=2)
        # b has more uses, should rank first
        assert top[0].name == "b"
        assert top[1].name == "a"

    def test_tiebreak_by_name(self):
        mem = init_policy_memory()
        mem = update_policy_memory(mem, _make_policy("beta"), 0.5)
        mem = update_policy_memory(mem, _make_policy("alpha"), 0.5)
        top = get_top_policies(mem, k=2)
        # Same score, same uses => alphabetical
        assert top[0].name == "alpha"
        assert top[1].name == "beta"

    def test_k_limits_results(self):
        mem = init_policy_memory()
        for i in range(10):
            mem = update_policy_memory(mem, _make_policy(f"p{i:02d}"), 0.1 * i)
        top = get_top_policies(mem, k=3)
        assert len(top) == 3

    def test_reconstructed_policies_are_valid(self):
        mem = init_policy_memory()
        p = _make_policy("test", priority="stability")
        mem = update_policy_memory(mem, p, 0.8)
        top = get_top_policies(mem, k=1)
        assert top[0].name == "test"
        assert top[0].priority == "stability"
        assert top[0].mode == "hybrid"

    def test_deterministic(self):
        def build():
            mem = init_policy_memory()
            mem = update_policy_memory(mem, _make_policy("a"), 0.5)
            mem = update_policy_memory(mem, _make_policy("b"), 0.9)
            return get_top_policies(mem, k=2)

        r1 = build()
        r2 = build()
        assert [p.name for p in r1] == [p.name for p in r2]


# ---------------------------------------------------------------------------
# TestExportImport
# ---------------------------------------------------------------------------


class TestExportImport:
    """Tests for export_policy_memory and import_policy_memory."""

    def test_roundtrip(self):
        mem = init_policy_memory()
        mem = update_policy_memory(mem, _make_policy("a"), 0.5)
        mem = update_policy_memory(mem, _make_policy("b"), 0.9)
        mem = update_policy_memory(mem, _make_policy("a"), 0.7)

        exported = export_policy_memory(mem)
        imported = import_policy_memory(exported)

        assert imported == mem

    def test_json_serializable(self):
        mem = init_policy_memory()
        mem = update_policy_memory(mem, _make_policy("x"), 0.42)
        exported = export_policy_memory(mem)
        # Must not raise.
        serialized = json.dumps(exported, sort_keys=True)
        # Round-trip through JSON.
        imported = import_policy_memory(json.loads(serialized))
        assert imported == mem

    def test_recomputes_average(self):
        """Import recomputes avg_score for integrity."""
        data = {
            "policies": {
                "test": {
                    "policy_dict": {"mode": "hybrid", "priority": "balanced",
                                    "thresholds": {}},
                    "scores": [0.4, 0.6],
                    "avg_score": 999.0,  # Intentionally wrong.
                    "uses": 2,
                },
            },
        }
        imported = import_policy_memory(data)
        assert imported["policies"]["test"]["avg_score"] == round(0.5, ROUND_PRECISION)

    def test_invalid_format_raises(self):
        try:
            import_policy_memory({"wrong": "format"})
            assert False, "Should have raised ValueError"
        except ValueError:
            pass

    def test_invalid_top_level_raises(self):
        try:
            import_policy_memory([1, 2, 3])
            assert False, "Should have raised ValueError"
        except ValueError:
            pass

    def test_no_mutation_of_input(self):
        mem = init_policy_memory()
        mem = update_policy_memory(mem, _make_policy("a"), 0.5)
        original = copy.deepcopy(mem)
        export_policy_memory(mem)
        assert mem == original


# ---------------------------------------------------------------------------
# TestFormatPolicyMemorySummary
# ---------------------------------------------------------------------------


class TestFormatPolicyMemorySummary:
    """Tests for format_policy_memory_summary."""

    def test_empty_memory(self):
        mem = init_policy_memory()
        text = format_policy_memory_summary(mem)
        assert "=== Policy Memory ===" in text
        assert "No policies stored." in text

    def test_with_policies(self):
        mem = init_policy_memory()
        mem = update_policy_memory(mem, _make_policy("balanced"), 0.89)
        mem = update_policy_memory(mem, _make_policy("balanced"), 0.89)
        mem = update_policy_memory(mem, _make_policy("stability_first"), 0.82)
        text = format_policy_memory_summary(mem)
        assert "Stored Policies:" in text
        assert "balanced" in text
        assert "stability_first" in text
        assert "Top Policies:" in text

    def test_with_replay_result(self):
        mem = init_policy_memory()
        mem = update_policy_memory(mem, _make_policy("a"), 0.9)
        replay = {
            "current_best": 0.91,
            "replay_best": 0.92,
            "improved": True,
        }
        text = format_policy_memory_summary(mem, replay)
        assert "Replay Result:" in text
        assert "Current:" in text
        assert "Improved: True" in text

    def test_deterministic(self):
        mem = init_policy_memory()
        mem = update_policy_memory(mem, _make_policy("x"), 0.5)
        assert format_policy_memory_summary(mem) == format_policy_memory_summary(mem)


# ---------------------------------------------------------------------------
# TestReplayPolicies
# ---------------------------------------------------------------------------


class TestReplayPolicies:
    """Tests for replay_policies."""

    def test_empty_memory_returns_no_improvement(self):
        runs = [_make_run([("s1", 0.7), ("s2", 0.5)])]
        mem = init_policy_memory()
        objective = {
            "w_stability": 0.3,
            "w_attractor": 0.3,
            "w_transient": 0.2,
            "w_sync": 0.2,
        }
        result = replay_policies(runs, mem, objective)
        assert result["improved"] is False
        assert result["replayed"] == {}

    def test_replay_produces_scores(self):
        runs = [_make_run([("s1", 0.7), ("s2", 0.5)])]
        mem = init_policy_memory()
        p = _make_policy("balanced")
        mem = update_policy_memory(mem, p, 0.8)
        objective = {
            "w_stability": 0.3,
            "w_attractor": 0.3,
            "w_transient": 0.2,
            "w_sync": 0.2,
        }
        result = replay_policies(runs, mem, objective, k=1)
        assert "balanced" in result["replayed"]
        assert isinstance(result["current_best"], float)
        assert isinstance(result["replay_best"], float)
        assert isinstance(result["improved"], bool)

    def test_deterministic(self):
        runs = [_make_run([("s1", 0.7), ("s2", 0.5)])]
        mem = init_policy_memory()
        mem = update_policy_memory(mem, _make_policy("balanced"), 0.8)
        objective = {
            "w_stability": 0.3,
            "w_attractor": 0.3,
            "w_transient": 0.2,
            "w_sync": 0.2,
        }
        r1 = replay_policies(runs, mem, objective, k=1)
        r2 = replay_policies(runs, mem, objective, k=1)
        assert r1 == r2

    def test_no_mutation_of_memory(self):
        runs = [_make_run([("s1", 0.7)])]
        mem = init_policy_memory()
        mem = update_policy_memory(mem, _make_policy("balanced"), 0.8)
        original = copy.deepcopy(mem)
        objective = {
            "w_stability": 0.3,
            "w_attractor": 0.3,
            "w_transient": 0.2,
            "w_sync": 0.2,
        }
        replay_policies(runs, mem, objective, k=1)
        assert mem == original


# ---------------------------------------------------------------------------
# TestMetaControlMemoryIntegration
# ---------------------------------------------------------------------------


class TestMetaControlMemoryIntegration:
    """Tests for memory integration in run_meta_control."""

    def test_meta_control_with_memory_disabled(self):
        """Memory disabled should behave identically to baseline."""
        from qec.analysis.meta_control import run_meta_control

        runs = [_make_run([("s1", 0.7), ("s2", 0.5)])]
        policies = [get_policy("stability_first"), get_policy("balanced")]
        objective = {
            "w_stability": 0.3,
            "w_attractor": 0.3,
            "w_transient": 0.2,
            "w_sync": 0.2,
        }
        r1 = run_meta_control(runs, policies, objective, max_steps=2)
        r2 = run_meta_control(
            runs, policies, objective, max_steps=2,
            use_memory=False, memory=None,
        )
        assert r1["scores"] == r2["scores"]
        assert r1["policies"] == r2["policies"]

    def test_meta_control_with_memory_enabled(self):
        """Memory enabled should not crash and produce valid output."""
        from qec.analysis.meta_control import run_meta_control

        runs = [_make_run([("s1", 0.7), ("s2", 0.5)])]
        policies = [get_policy("stability_first")]
        mem = init_policy_memory()
        mem = update_policy_memory(mem, _make_policy("balanced"), 0.8)
        objective = {
            "w_stability": 0.3,
            "w_attractor": 0.3,
            "w_transient": 0.2,
            "w_sync": 0.2,
        }
        result = run_meta_control(
            runs, policies, objective, max_steps=2,
            use_memory=True, memory=mem,
        )
        assert "scores" in result
        assert "policies" in result
        assert len(result["scores"]) > 0


# ---------------------------------------------------------------------------
# TestStrategyAdapterIntegration
# ---------------------------------------------------------------------------


class TestStrategyAdapterIntegration:
    """Tests for run_policy_memory_analysis in strategy_adapter."""

    def test_basic_pipeline(self):
        from qec.analysis.strategy_adapter import run_policy_memory_analysis

        runs = [_make_run([("s1", 0.7), ("s2", 0.5)])]
        result = run_policy_memory_analysis(runs, max_steps=2)

        assert "meta_result" in result
        assert "memory" in result
        assert "summary" in result

    def test_pipeline_with_replay(self):
        from qec.analysis.strategy_adapter import run_policy_memory_analysis

        runs = [_make_run([("s1", 0.7), ("s2", 0.5)])]
        # Pre-seed memory.
        mem = init_policy_memory()
        mem = update_policy_memory(mem, _make_policy("balanced"), 0.8)

        result = run_policy_memory_analysis(
            runs, memory=mem, max_steps=2, replay=True, replay_k=1,
        )

        assert "replay_result" in result
        assert isinstance(result["replay_result"]["improved"], bool)

    def test_memory_updated_after_run(self):
        from qec.analysis.strategy_adapter import run_policy_memory_analysis

        runs = [_make_run([("s1", 0.7), ("s2", 0.5)])]
        result = run_policy_memory_analysis(runs, max_steps=2)

        mem = result["memory"]
        assert len(mem["policies"]) > 0

    def test_format_adapter_summary(self):
        from qec.analysis.strategy_adapter import (
            format_policy_memory_adapter_summary,
            run_policy_memory_analysis,
        )

        runs = [_make_run([("s1", 0.7), ("s2", 0.5)])]
        result = run_policy_memory_analysis(runs, max_steps=2)
        text = format_policy_memory_adapter_summary(result)
        assert isinstance(text, str)
        assert "Policy Memory" in text

    def test_deterministic(self):
        from qec.analysis.strategy_adapter import run_policy_memory_analysis

        runs = [_make_run([("s1", 0.7), ("s2", 0.5)])]
        r1 = run_policy_memory_analysis(runs, max_steps=2)
        r2 = run_policy_memory_analysis(runs, max_steps=2)
        assert r1["memory"] == r2["memory"]
        assert r1["meta_result"]["scores"] == r2["meta_result"]["scores"]
