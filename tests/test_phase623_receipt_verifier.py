"""Contract tests for Phase 623's generated JUnit inventory verifier.

Synthetic XML in these tests is *not* executed experiment evidence. The real
CI gate runs the signed socket trials first and only then validates JUnit.
"""
from __future__ import annotations

from collections import Counter
import xml.etree.ElementTree as ET

import pytest

from scripts.maintenance.verify_phase623_trial import EXPECTED, inspect

SHA = "a" * 40


def _junit(*, mutate=None) -> bytes:
    root = ET.Element("testsuite")
    for name, count in EXPECTED.items():
        for index in range(count):
            testcase = ET.SubElement(root, "testcase", {
                "classname": "test_phase623_end_to_end",
                "name": f"{name}[{index}]" if count > 1 else name,
            })
            if mutate is not None:
                mutate(testcase, name, index)
    return ET.tostring(root)


def test_phase623_verifier_exact_inventory_contract():
    receipt = inspect(_junit(), SHA)
    assert receipt["test_count"] == 11
    assert receipt["scenario_count"] == 8
    assert receipt["source_commit"] == SHA
    assert receipt["skipped"] == receipt["errors"] == receipt["failures"] == 0
    assert len(receipt["junit_sha256"]) == 64


@pytest.mark.parametrize("tag", ("failure", "error", "skipped"))
def test_phase623_verifier_refuses_nonpassing_trials(tag):
    def mutate(case, name, index):
        if name == next(iter(EXPECTED)) and index == 0:
            ET.SubElement(case, tag, {"message": "controlled synthetic failure"})
    with pytest.raises(ValueError):
        inspect(_junit(mutate=mutate), SHA)


def test_phase623_verifier_refuses_missing_and_repeated_test_names():
    root = ET.fromstring(_junit())
    root.remove(root[-1])
    with pytest.raises(ValueError):
        inspect(ET.tostring(root), SHA)
    root = ET.fromstring(_junit())
    root[-1].set("name", root[0].get("name"))
    with pytest.raises(ValueError):
        inspect(ET.tostring(root), SHA)


def test_phase623_verifier_refuses_wrong_module_or_commit():
    root = ET.fromstring(_junit())
    root[0].set("classname", "unrelated_tests")
    with pytest.raises(ValueError):
        inspect(ET.tostring(root), SHA)
    with pytest.raises(ValueError):
        inspect(_junit(), "not-a-real-commit")
