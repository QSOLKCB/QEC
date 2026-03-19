from __future__ import annotations

import numpy as np

from qec.analysis.nb_trapping_set_predictor import NBTrappingSetPredictor


def test_candidate_scores_rounding_and_formula() -> None:
    predictor = NBTrappingSetPredictor()
    scores = predictor._compute_candidate_scores(  # noqa: SLF001
        [[0, 2], [1], []],
        {0: 1.123456789012, 1: 0.4, 2: 0.6},
        0.2,
    )

    expected0 = round(np.mean([1.123456789012, 0.6]) + 1.123456789012 + 0.2, 12)
    expected1 = round(0.4 + 0.4 + 0.2, 12)

    assert scores == [expected0, expected1]
    for score in scores:
        assert round(score, 12) == score


def test_candidate_scores_align_with_candidate_sets_in_prediction() -> None:
    H = np.array([
        [1, 1, 0, 1, 0, 0],
        [0, 1, 1, 0, 1, 0],
        [1, 0, 1, 0, 0, 1],
    ], dtype=np.float64)
    predictor = NBTrappingSetPredictor()

    r1 = predictor.predict_trapping_regions(H)
    r2 = predictor.predict_trapping_regions(H)

    assert r1 == r2
    assert len(r1["candidate_scores"]) == len(r1["candidate_sets"])
    for value in r1["candidate_scores"]:
        assert round(value, 12) == value
