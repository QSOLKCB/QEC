"""Decoder design metrics — experimental design layer.

Computes evaluation metrics for the ternary bosonic pipeline and
baselines. All metrics are deterministic and bounded in [0, 1].

Dependencies: numpy.
"""

from __future__ import annotations

from typing import Dict

import numpy as np


def neutral_usage(ternary: np.ndarray) -> float:
    """Fraction of ternary values that are neutral (0).

    Parameters
    ----------
    ternary : np.ndarray
        Array with values in {-1, 0, +1}.

    Returns
    -------
    float
        Neutral fraction in [0, 1].
    """
    if ternary.size == 0:
        return 0.0
    return float(np.mean(ternary == 0))


def confidence_efficiency(confidences: np.ndarray) -> float:
    """Mean confidence across all nodes.

    Parameters
    ----------
    confidences : np.ndarray
        Array of confidence values in [0, 1].

    Returns
    -------
    float
        Mean confidence in [0, 1].
    """
    if confidences.size == 0:
        return 0.0
    return float(np.clip(np.mean(confidences), 0.0, 1.0))


def agreement_rate(
    ternary: np.ndarray,
    baseline: np.ndarray,
) -> float:
    """Fraction of positions where ternary and baseline agree on sign.

    Neutral ternary values (0) are treated as agreeing with any
    baseline value (they abstain rather than disagree).

    Parameters
    ----------
    ternary : np.ndarray
        Ternary decisions {-1, 0, +1}.
    baseline : np.ndarray
        Baseline decisions (sign matters).

    Returns
    -------
    float
        Agreement rate in [0, 1].
    """
    if ternary.size == 0:
        return 1.0
    neutral_mask = ternary == 0
    sign_baseline = np.sign(baseline)
    sign_baseline[sign_baseline == 0] = 1
    agree = neutral_mask | (ternary == sign_baseline)
    return float(np.mean(agree))


def design_score(
    ternary: np.ndarray,
    confidences: np.ndarray,
    baseline: np.ndarray,
    *,
    w_neutral: float = 0.3,
    w_confidence: float = 0.4,
    w_agreement: float = 0.3,
) -> float:
    """Composite design score in [0, 1].

    Weighted combination of neutral usage, confidence efficiency,
    and agreement rate with a baseline decoder.

    Parameters
    ----------
    ternary : np.ndarray
        Ternary states {-1, 0, +1}.
    confidences : np.ndarray
        Node confidences [0, 1].
    baseline : np.ndarray
        Baseline decoder output.
    w_neutral, w_confidence, w_agreement : float
        Weights (must sum to 1).

    Returns
    -------
    float
        Composite score in [0, 1].
    """
    # Fixed ordering: neutral, confidence, agreement
    nu = neutral_usage(ternary)
    ce = confidence_efficiency(confidences)
    ar = agreement_rate(ternary, baseline)
    score = w_neutral * nu + w_confidence * ce + w_agreement * ar
    # Round to 1e-12 to eliminate cross-platform floating-point drift
    score = round(score, 12)
    return float(np.clip(score, 0.0, 1.0))


def compute_all_metrics(
    ternary: np.ndarray,
    confidences: np.ndarray,
    baseline: np.ndarray,
) -> Dict[str, float]:
    """Compute all decoder design metrics.

    Parameters
    ----------
    ternary : np.ndarray
        Ternary states.
    confidences : np.ndarray
        Confidence values.
    baseline : np.ndarray
        Baseline decoder output.

    Returns
    -------
    dict
        All metric values.
    """
    return {
        "neutral_usage": neutral_usage(ternary),
        "confidence_efficiency": confidence_efficiency(confidences),
        "agreement_rate": agreement_rate(ternary, baseline),
        "design_score": design_score(ternary, confidences, baseline),
    }
