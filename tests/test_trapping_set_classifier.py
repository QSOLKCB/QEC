from __future__ import annotations

from src.qec.analysis.api import classify_trapping_set


def test_classifier_small_trapping_set() -> None:
    label = classify_trapping_set([0, 1, 2], a=3, b=1, localization_metrics={"ipr": 0.2, "support_fraction": 0.2, "topk_mass_fraction": 0.9})
    assert label == "SMALL_TRAPPING_SET"


def test_classifier_pseudocodeword_basin_stability() -> None:
    metrics = {"ipr": 0.35, "support_fraction": 0.2, "topk_mass_fraction": 0.8}
    l1 = classify_trapping_set([0, 1, 5, 7], a=8, b=4, localization_metrics=metrics)
    l2 = classify_trapping_set([0, 1, 5, 7], a=8, b=4, localization_metrics=metrics)
    assert l1 == l2 == "PSEUDO_CODEWORD_BASIN"
