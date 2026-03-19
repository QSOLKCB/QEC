from __future__ import annotations

from qec.discovery.basin_switch_detector import BasinSwitchDetector


def test_no_basin_switch_when_distance_small() -> None:
    detector = BasinSwitchDetector(threshold=0.5)
    prev = [1.0, 0.2, 0.1]
    new = [1.1, 0.25, 0.15]
    assert detector.detect(prev, new) is False


def test_basin_switch_when_distance_large() -> None:
    detector = BasinSwitchDetector(threshold=0.5)
    prev = [1.0, 0.2, 0.1]
    new = [2.0, 1.0, 0.8]
    assert detector.detect(prev, new) is True
