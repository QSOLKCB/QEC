from __future__ import annotations

from qec.discovery.basin_switch_detector import BasinSwitchDetector
from qec.discovery.mutation_trust_region import SpectralTrustRegion


def test_trust_region_allows_within_radius() -> None:
    trust = SpectralTrustRegion(radius=0.25)
    assert trust.allow([0.0, 0.0, 0.0], [0.1, 0.1, 0.1])


def test_trust_region_rejects_outside_radius() -> None:
    trust = SpectralTrustRegion(radius=0.25)
    assert not trust.allow([0.0, 0.0, 0.0], [0.3, 0.3, 0.0])


def test_basin_switch_detector_threshold() -> None:
    detector = BasinSwitchDetector(threshold=0.5)
    assert detector.detect([0.0, 0.0, 0.0], [0.4, 0.4, 0.0])
    assert not detector.detect([0.0, 0.0, 0.0], [0.2, 0.2, 0.0])


def test_trust_region_determinism() -> None:
    trust = SpectralTrustRegion(radius=0.5)
    old_s = [0.1, 0.2, 0.3]
    new_s = [0.3, 0.2, 0.1]
    assert trust.spectral_distance(old_s, new_s) == trust.spectral_distance(old_s, new_s)
