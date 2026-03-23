"""Tests for conjecture execution and validation engine (v97.2.0).

Covers:
  - input normalization
  - conjecture-to-metric mapping
  - execution correctness
  - validation logic
  - scoring rules
  - pipeline integration
  - no mutation
  - determinism
  - print output
  - integration with v97.1 conjecture_engine
"""

import copy

from qec.analysis.conjecture_execution import (
    execute_conjecture,
    normalize_conjecture_inputs,
    print_conjecture_results,
    run_conjecture_execution,
    score_validation,
    validate_conjecture,
)


# ---------------------------------------------------------------------------
# FIXTURES
# ---------------------------------------------------------------------------

def _make_systems():
    """Create test system data."""
    return [
        {
            "system_class": "toroidal",
            "best_mode": "square",
            "friction_score": 1.0,
            "oscillation_ratio": 0.3,
            "churn_score": 0.4,
            "stability_efficiency": 0.8,
            "core_invariants": [{"name": "parity"}],
            "law_matches": [],
        },
        {
            "system_class": "toroidal",
            "best_mode": "d4>e8_like",
            "friction_score": 3.0,
            "oscillation_ratio": 0.7,
            "churn_score": 0.8,
            "stability_efficiency": 0.5,
            "core_invariants": [{"name": "parity"}],
            "law_matches": [],
        },
        {
            "system_class": "hyperbolic",
            "best_mode": "d4",
            "friction_score": 2.0,
            "oscillation_ratio": 0.6,
            "churn_score": 0.3,
            "stability_efficiency": 0.7,
            "core_invariants": [{"name": "symmetry"}],
            "law_matches": [],
        },
    ]


def _make_oscillation_conjecture():
    return {
        "statement": "IF oscillation_ratio is high (0.70) for class 'toroidal' "
                     "THEN simpler modes reduce oscillation",
        "type": "oscillation_reduction",
        "conditions": {
            "oscillation_ratio": 0.7,
            "system_class": "toroidal",
            "threshold": 0.5,
        },
    }


def _make_churn_conjecture():
    return {
        "statement": "IF churn_score is high (0.80) for class 'toroidal' "
                     "THEN curvature-like constraint (1,-2,1) reduces churn",
        "type": "churn_reduction",
        "conditions": {
            "churn_score": 0.8,
            "system_class": "toroidal",
            "threshold": 0.6,
        },
    }


def _make_friction_conjecture():
    return {
        "statement": "IF friction_score is high (3.00) for class 'toroidal' "
                     "THEN reducing invariant conflict improves efficiency",
        "type": "friction_reduction",
        "conditions": {
            "friction_score": 3.0,
            "system_class": "toroidal",
            "threshold": 2.5,
        },
    }


def _make_hierarchy_conjecture():
    return {
        "statement": "IF multi-stage mode 'd4>e8_like' increases friction "
                     "(3.00) for class 'toroidal' "
                     "THEN optimal solution lies in partial hierarchy",
        "type": "hierarchy_optimization",
        "conditions": {
            "mode": "d4>e8_like",
            "friction_score": 3.0,
            "system_class": "toroidal",
            "is_multi_stage": True,
            "threshold": 2.5,
        },
    }


def _make_invariant_conjecture():
    return {
        "statement": "IF invariant 'parity' appears in 2 classes "
                     "THEN it generalizes as a reusable correction rule",
        "type": "invariant_generalization",
        "conditions": {
            "invariant_name": "parity",
            "class_count": 2,
            "classes": ["hyperbolic", "toroidal"],
            "threshold": 2,
        },
    }


# ---------------------------------------------------------------------------
# TEST — INPUT NORMALIZATION
# ---------------------------------------------------------------------------

class TestNormalizeConjectureInputs:

    def test_dict_with_conjectures_and_systems(self):
        data = {
            "conjectures": [_make_oscillation_conjecture()],
            "systems": _make_systems(),
        }
        result = normalize_conjecture_inputs(data)
        assert "conjectures" in result
        assert "systems" in result
        assert len(result["conjectures"]) == 1
        assert len(result["systems"]) == 3

    def test_empty_data(self):
        result = normalize_conjecture_inputs({})
        assert result["conjectures"] == []
        assert result["systems"] == []

    def test_systems_sorted(self):
        data = {
            "conjectures": [],
            "systems": _make_systems(),
        }
        result = normalize_conjecture_inputs(data)
        classes = [s["system_class"] for s in result["systems"]]
        assert classes == sorted(classes)

    def test_conjectures_sorted(self):
        c1 = _make_oscillation_conjecture()
        c2 = _make_churn_conjecture()
        data = {"conjectures": [c1, c2], "systems": []}
        result = normalize_conjecture_inputs(data)
        types = [c["type"] for c in result["conjectures"]]
        assert types == sorted(types)

    def test_no_mutation(self):
        data = {
            "conjectures": [_make_oscillation_conjecture()],
            "systems": _make_systems(),
        }
        original = copy.deepcopy(data)
        normalize_conjecture_inputs(data)
        assert data == original

    def test_candidates_key(self):
        data = {
            "conjectures": [_make_friction_conjecture()],
            "candidates": _make_systems(),
        }
        result = normalize_conjecture_inputs(data)
        assert len(result["systems"]) == 3

    def test_groups_key(self):
        data = {
            "conjectures": [],
            "groups": [{"best": _make_systems()[0]}],
        }
        result = normalize_conjecture_inputs(data)
        assert len(result["systems"]) == 1


# ---------------------------------------------------------------------------
# TEST — EXECUTION (CONJECTURE → METRIC)
# ---------------------------------------------------------------------------

class TestExecuteConjecture:

    def test_oscillation_reduction_with_modes(self):
        """Oscillation conjecture compares simple vs complex modes."""
        conj = _make_oscillation_conjecture()
        systems = _make_systems()
        result = execute_conjecture(conj, systems)
        assert "observed_change" in result
        assert "direction" in result
        # square (0.3) vs d4>e8_like (0.7): simple - complex = -0.4
        assert result["observed_change"] < 0
        assert result["direction"] == "decrease"

    def test_churn_reduction(self):
        conj = _make_churn_conjecture()
        systems = _make_systems()
        result = execute_conjecture(conj, systems)
        assert "observed_change" in result
        assert "direction" in result

    def test_friction_reduction(self):
        conj = _make_friction_conjecture()
        systems = _make_systems()
        result = execute_conjecture(conj, systems)
        assert "observed_change" in result
        assert "direction" in result

    def test_hierarchy_optimization(self):
        conj = _make_hierarchy_conjecture()
        systems = _make_systems()
        result = execute_conjecture(conj, systems)
        assert "observed_change" in result
        assert "direction" in result

    def test_invariant_generalization(self):
        conj = _make_invariant_conjecture()
        systems = _make_systems()
        result = execute_conjecture(conj, systems)
        assert "observed_change" in result

    def test_unknown_type(self):
        conj = {"type": "unknown_type", "conditions": {}}
        systems = _make_systems()
        result = execute_conjecture(conj, systems)
        assert "observed_change" in result

    def test_empty_systems(self):
        conj = _make_oscillation_conjecture()
        result = execute_conjecture(conj, [])
        assert result["observed_change"] == 0.0
        assert result["direction"] == "none"

    def test_no_mutation(self):
        conj = _make_oscillation_conjecture()
        systems = _make_systems()
        orig_conj = copy.deepcopy(conj)
        orig_systems = copy.deepcopy(systems)
        execute_conjecture(conj, systems)
        assert conj == orig_conj
        assert systems == orig_systems

    def test_single_system_fallback(self):
        """Single system uses threshold comparison."""
        conj = _make_friction_conjecture()
        systems = [_make_systems()[1]]  # friction=3.0, threshold=2.5
        result = execute_conjecture(conj, systems)
        assert result["observed_change"] == round(3.0 - 2.5, 6)
        assert result["direction"] == "increase"


# ---------------------------------------------------------------------------
# TEST — VALIDATION
# ---------------------------------------------------------------------------

class TestValidateConjecture:

    def test_confirmed_decrease(self):
        conj = _make_oscillation_conjecture()
        result = {"observed_change": -0.4, "direction": "decrease"}
        v = validate_conjecture(conj, result)
        assert v["status"] == "confirmed"

    def test_rejected_opposite(self):
        conj = _make_oscillation_conjecture()
        result = {"observed_change": 0.3, "direction": "increase"}
        v = validate_conjecture(conj, result)
        assert v["status"] == "rejected"

    def test_inconclusive_no_change(self):
        conj = _make_oscillation_conjecture()
        result = {"observed_change": 0.0, "direction": "none"}
        v = validate_conjecture(conj, result)
        assert v["status"] == "inconclusive"

    def test_confirmed_increase(self):
        conj = _make_invariant_conjecture()
        result = {"observed_change": 0.2, "direction": "increase"}
        v = validate_conjecture(conj, result)
        assert v["status"] == "confirmed"

    def test_rejected_increase_when_decrease_expected(self):
        conj = _make_friction_conjecture()
        result = {"observed_change": 0.5, "direction": "increase"}
        v = validate_conjecture(conj, result)
        assert v["status"] == "rejected"

    def test_unknown_type_inconclusive(self):
        conj = {"type": "nonexistent", "conditions": {}}
        result = {"observed_change": 0.0, "direction": "none"}
        v = validate_conjecture(conj, result)
        assert v["status"] == "inconclusive"

    def test_no_mutation(self):
        conj = _make_oscillation_conjecture()
        result = {"observed_change": -0.4, "direction": "decrease"}
        orig_conj = copy.deepcopy(conj)
        orig_result = copy.deepcopy(result)
        validate_conjecture(conj, result)
        assert conj == orig_conj
        assert result == orig_result


# ---------------------------------------------------------------------------
# TEST — SCORING
# ---------------------------------------------------------------------------

class TestScoreValidation:

    def test_confirmed_strong_effect(self):
        assert score_validation("confirmed", {"observed_change": -0.5}) == 2

    def test_confirmed_weak_effect(self):
        assert score_validation("confirmed", {"observed_change": -0.1}) == 1

    def test_confirmed_threshold_boundary(self):
        assert score_validation("confirmed", {"observed_change": 0.2}) == 2

    def test_inconclusive(self):
        assert score_validation("inconclusive", {"observed_change": 0.0}) == 0

    def test_rejected(self):
        assert score_validation("rejected", {"observed_change": 0.5}) == -1

    def test_no_mutation(self):
        result = {"observed_change": -0.5}
        original = copy.deepcopy(result)
        score_validation("confirmed", result)
        assert result == original


# ---------------------------------------------------------------------------
# TEST — PIPELINE
# ---------------------------------------------------------------------------

class TestRunConjectureExecution:

    def test_full_pipeline(self):
        data = {
            "conjectures": [
                _make_oscillation_conjecture(),
                _make_churn_conjecture(),
                _make_friction_conjecture(),
            ],
            "systems": _make_systems(),
        }
        report = run_conjecture_execution(data)
        assert "results" in report
        assert "summary" in report
        assert len(report["results"]) == 3

        total = sum(report["summary"].values())
        assert total == 3

    def test_output_structure(self):
        data = {
            "conjectures": [_make_oscillation_conjecture()],
            "systems": _make_systems(),
        }
        report = run_conjecture_execution(data)
        r = report["results"][0]
        assert "statement" in r
        assert "status" in r
        assert "score" in r
        assert "observed_change" in r

    def test_summary_counts(self):
        data = {
            "conjectures": [
                _make_oscillation_conjecture(),
                _make_friction_conjecture(),
            ],
            "systems": _make_systems(),
        }
        report = run_conjecture_execution(data)
        s = report["summary"]
        assert "confirmed" in s
        assert "rejected" in s
        assert "inconclusive" in s
        assert sum(s.values()) == 2

    def test_empty_conjectures(self):
        data = {"conjectures": [], "systems": _make_systems()}
        report = run_conjecture_execution(data)
        assert report["results"] == []
        assert report["summary"]["confirmed"] == 0
        assert report["summary"]["rejected"] == 0
        assert report["summary"]["inconclusive"] == 0

    def test_empty_data(self):
        report = run_conjecture_execution({})
        assert report["results"] == []

    def test_deterministic_output(self):
        data = {
            "conjectures": [
                _make_oscillation_conjecture(),
                _make_churn_conjecture(),
                _make_hierarchy_conjecture(),
            ],
            "systems": _make_systems(),
        }
        r1 = run_conjecture_execution(data)
        r2 = run_conjecture_execution(data)
        assert r1 == r2

    def test_no_mutation(self):
        data = {
            "conjectures": [_make_oscillation_conjecture()],
            "systems": _make_systems(),
        }
        original = copy.deepcopy(data)
        run_conjecture_execution(data)
        assert data == original

    def test_results_sorted_by_status(self):
        """Confirmed should appear before rejected before inconclusive."""
        data = {
            "conjectures": [
                _make_oscillation_conjecture(),
                _make_churn_conjecture(),
                _make_friction_conjecture(),
                _make_hierarchy_conjecture(),
            ],
            "systems": _make_systems(),
        }
        report = run_conjecture_execution(data)
        statuses = [r["status"] for r in report["results"]]
        status_order = {"confirmed": 0, "rejected": 1, "inconclusive": 2}
        orders = [status_order.get(s, 3) for s in statuses]
        assert orders == sorted(orders)


# ---------------------------------------------------------------------------
# TEST — PRINT
# ---------------------------------------------------------------------------

class TestPrintConjectureResults:

    def test_basic_output(self):
        report = {
            "results": [
                {
                    "statement": "simpler modes reduce oscillation",
                    "status": "confirmed",
                    "score": 2,
                    "observed_change": -0.32,
                },
            ],
            "summary": {"confirmed": 1, "rejected": 0, "inconclusive": 0},
        }
        text = print_conjecture_results(report)
        assert "=== Conjecture Results ===" in text
        assert "[CONFIRMED]" in text
        assert "simpler modes reduce oscillation" in text
        assert "-0.32" in text

    def test_all_statuses(self):
        report = {
            "results": [
                {
                    "statement": "oscillation",
                    "status": "confirmed",
                    "score": 2,
                    "observed_change": -0.32,
                },
                {
                    "statement": "churn",
                    "status": "rejected",
                    "score": -1,
                    "observed_change": 0.05,
                },
                {
                    "statement": "hierarchy",
                    "status": "inconclusive",
                    "score": 0,
                    "observed_change": 0.0,
                },
            ],
            "summary": {"confirmed": 1, "rejected": 1, "inconclusive": 1},
        }
        text = print_conjecture_results(report)
        assert "[CONFIRMED]" in text
        assert "[REJECTED]" in text
        assert "[INCONCLUSIVE]" in text
        assert "confirmed: 1" in text
        assert "rejected: 1" in text
        assert "inconclusive: 1" in text

    def test_empty_results(self):
        report = {
            "results": [],
            "summary": {"confirmed": 0, "rejected": 0, "inconclusive": 0},
        }
        text = print_conjecture_results(report)
        assert "No conjecture results." in text

    def test_no_mutation(self):
        report = {
            "results": [
                {
                    "statement": "test",
                    "status": "confirmed",
                    "score": 1,
                    "observed_change": -0.1,
                },
            ],
            "summary": {"confirmed": 1, "rejected": 0, "inconclusive": 0},
        }
        original = copy.deepcopy(report)
        print_conjecture_results(report)
        assert report == original

    def test_deterministic_output(self):
        report = {
            "results": [
                {
                    "statement": "test",
                    "status": "confirmed",
                    "score": 2,
                    "observed_change": -0.5,
                },
            ],
            "summary": {"confirmed": 1, "rejected": 0, "inconclusive": 0},
        }
        t1 = print_conjecture_results(report)
        t2 = print_conjecture_results(report)
        assert t1 == t2


# ---------------------------------------------------------------------------
# TEST — INTEGRATION WITH v97.1
# ---------------------------------------------------------------------------

class TestIntegrationWithConjectureEngine:

    def test_engine_output_as_input(self):
        """Output from conjecture_engine can feed into execution pipeline."""
        from qec.analysis.conjecture_engine import run_conjecture_engine

        systems = _make_systems()
        engine_report = run_conjecture_engine(systems)

        # Use engine output + systems as execution input
        data = {
            "conjectures": engine_report["conjectures"],
            "systems": systems,
        }
        exec_report = run_conjecture_execution(data)

        assert "results" in exec_report
        assert "summary" in exec_report
        # Should have at least as many results as engine conjectures
        assert len(exec_report["results"]) == len(engine_report["conjectures"])

    def test_engine_integration_deterministic(self):
        """Full engine -> execution pipeline is deterministic."""
        from qec.analysis.conjecture_engine import run_conjecture_engine

        systems = _make_systems()
        engine_report = run_conjecture_engine(systems)

        data = {
            "conjectures": engine_report["conjectures"],
            "systems": systems,
        }
        r1 = run_conjecture_execution(data)
        r2 = run_conjecture_execution(data)
        assert r1 == r2
