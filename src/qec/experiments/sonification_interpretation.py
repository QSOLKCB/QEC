"""
Sonification Interpretation Layer (v72.4.0)

Deterministic interpretation of comparison metrics from
run_sonification_comparison() (v72.3.0).

Transforms raw metrics into normalized scores, a composite score,
and a classification verdict.
"""


def interpret_sonification_comparison(comparison: dict) -> dict:
    """Interpret comparison metrics from run_sonification_comparison().

    Args:
        comparison: Dict produced by run_sonification_comparison().

    Returns:
        Dict with keys: scores, derived, composite_score, verdict.
    """
    # Extract inputs (no mutation of comparison dict)
    baseline_silence = comparison["baseline_silence_fidelity"]
    multidim_silence = comparison["multidim_silence_fidelity"]
    leakage_rate = comparison["leakage_rate"]
    baseline_energy = comparison["baseline_energy"]
    ch0_energy = comparison["multidim_ch0_energy"]
    ch1_energy = comparison["multidim_ch1_energy"]
    channel_corr = comparison["channel_correlation"]
    baseline_var = comparison["baseline_variance"]
    multidim_var = comparison["multidim_variance"]

    # Step 1 — Derived metrics
    if baseline_energy == 0.0:
        energy_ratio = 1.0
    else:
        energy_ratio = (ch0_energy + ch1_energy) / (2.0 * baseline_energy)

    variance_delta = multidim_var - baseline_var
    abs_corr = abs(channel_corr)

    # Step 2 — Normalization to [0, 1]
    silence_score = min(baseline_silence, multidim_silence)
    leakage_score = 1.0 - leakage_rate
    correlation_score = 1.0 - abs_corr

    # Bounded sigmoid-style mapping for variance
    if variance_delta == 0.0:
        variance_score = 0.0
    else:
        variance_score = variance_delta / (1.0 + abs(variance_delta))

    energy_score = min(energy_ratio, 1.0)

    # Clamp all scores to [0, 1]
    silence_score = max(0.0, min(1.0, silence_score))
    leakage_score = max(0.0, min(1.0, leakage_score))
    correlation_score = max(0.0, min(1.0, correlation_score))
    variance_score_clamped = max(0.0, min(1.0, variance_score))
    energy_score = max(0.0, min(1.0, energy_score))

    # Step 3 — Composite score (weighted sum)
    composite = (
        0.35 * silence_score
        + 0.20 * leakage_score
        + 0.15 * correlation_score
        + 0.15 * variance_score_clamped
        + 0.15 * energy_score
    )
    composite = max(0.0, min(1.0, composite))

    # Step 4 — Classification verdict
    if silence_score < 1.0 or leakage_rate > 0.0:
        verdict = "invalid"
    elif correlation_score > 0.5 and variance_score > 0.0:
        verdict = "multidim_improves_structure"
    elif correlation_score < 0.2:
        verdict = "channels_redundant"
    elif variance_score < 0.0:
        verdict = "baseline_more_stable"
    else:
        verdict = "tradeoff"

    return {
        "scores": {
            "silence": silence_score,
            "leakage": leakage_score,
            "correlation": correlation_score,
            "variance": variance_score_clamped,
            "energy": energy_score,
        },
        "derived": {
            "energy_ratio": energy_ratio,
            "variance_delta": variance_delta,
            "abs_correlation": abs_corr,
        },
        "composite_score": composite,
        "verdict": verdict,
    }
