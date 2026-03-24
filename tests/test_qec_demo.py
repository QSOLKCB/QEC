"""Smoke tests for scripts/qec_demo.py."""

from __future__ import annotations

import subprocess
import sys


def test_demo_import():
    """Verify the demo module can be imported without error."""
    import importlib
    import os

    demo_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "scripts",
    )
    sys.path.insert(0, demo_path)
    try:
        mod = importlib.import_module("qec_demo")
        assert hasattr(mod, "run_demo")
        assert hasattr(mod, "main")
    finally:
        sys.path.pop(0)


def test_demo_runs_and_returns_results():
    """Verify the demo runs and returns structured results."""
    import os

    demo_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "scripts",
    )
    sys.path.insert(0, demo_path)
    try:
        from qec_demo import run_demo

        results = run_demo()
        assert "steps" in results
        assert "regime_counts" in results
        assert len(results["steps"]) > 0
        assert results["memory_keys"] > 0
        assert results["transition_entries"] > 0
        assert results["history_length"] > 0
    finally:
        sys.path.pop(0)


def test_demo_determinism():
    """Verify repeated runs produce identical results."""
    import os

    demo_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "scripts",
    )
    sys.path.insert(0, demo_path)
    try:
        from qec_demo import run_demo

        r1 = run_demo()
        r2 = run_demo()
        assert r1 == r2
    finally:
        sys.path.pop(0)


def test_demo_regimes_are_real():
    """Verify regimes come from the actual classification system."""
    import os

    demo_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "scripts",
    )
    sys.path.insert(0, demo_path)
    try:
        from qec_demo import run_demo

        results = run_demo()
        valid_regimes = {"stable", "transitional", "oscillatory", "unstable", "mixed"}
        for step in results["steps"]:
            assert step["regime"] in valid_regimes
    finally:
        sys.path.pop(0)
