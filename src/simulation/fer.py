"""
Deterministic Monte Carlo frame-error-rate (FER) simulation harness.

All randomness flows through ``np.random.default_rng(seed)``.
Output is a plain-Python dict that is directly JSON-serializable.
"""

from __future__ import annotations

import json
import numpy as np

from ..qec_qldpc_codes import bp_decode, channel_llr, syndrome


# ═══════════════════════════════════════════════════════════════════════
# Wilson Score Confidence Interval Helpers (numpy-only, deterministic)
# ═══════════════════════════════════════════════════════════════════════

def _probit(p):
    """Inverse standard normal CDF (quantile function), numpy-only.

    Uses Abramowitz & Stegun approximation 26.2.23 for the central
    region and a rational tail approximation.  Accuracy ~1e-8, which
    is more than sufficient for confidence-interval computation.

    Args:
        p: Probability in (0, 1).

    Returns:
        z such that Phi(z) = p, where Phi is the standard normal CDF.
    """
    # Coefficients for the rational approximation (A&S 26.2.23).
    a0 = 2.515517
    a1 = 0.802853
    a2 = 0.010328
    b1 = 1.432788
    b2 = 0.189269
    b3 = 0.001308

    if p <= 0.0 or p >= 1.0:
        raise ValueError(f"_probit requires p in (0, 1), got {p}")

    # Use symmetry: if p > 0.5, compute for (1 - p) and negate.
    if p > 0.5:
        return -_probit(1.0 - p)

    # t = sqrt(-2 * ln(p))  for p <= 0.5
    t = np.sqrt(-2.0 * np.log(p))

    # Rational approximation: z ≈ t - (a0 + a1*t + a2*t^2) / (1 + b1*t + b2*t^2 + b3*t^3)
    numerator = a0 + a1 * t + a2 * t * t
    denominator = 1.0 + b1 * t + b2 * t * t + b3 * t * t * t
    z = -(t - numerator / denominator)

    return z


def _wilson_ci(k, n_trials, alpha, gamma):
    """Continuity-corrected Wilson score confidence interval.

    Computes a confidence interval for a binomial proportion p_hat = k / n
    using the Wilson score method with an additive continuity correction
    controlled by *gamma*.

    Args:
        k: Number of successes (e.g. frame errors).
        n_trials: Total number of trials.
        alpha: Significance level (e.g. 0.05 for 95% CI).
        gamma: Continuity correction factor (>= 0).

    Returns:
        (lower, upper, width) tuple of floats.
        lower and upper are clamped to [0, 1].
    """
    # z-score for two-sided alpha: e.g. alpha=0.05 → z ≈ 1.96.
    z = _probit(1.0 - alpha / 2.0)
    z2 = z * z

    p_hat = float(k) / float(n_trials)
    n = float(n_trials)

    # Continuity correction: gamma / (2n), applied symmetrically.
    cc = gamma / (2.0 * n)

    # Wilson score denominator.
    denom = 1.0 + z2 / n

    # Wilson score centre.
    centre = (p_hat + z2 / (2.0 * n)) / denom

    # Wilson score margin (before continuity correction).
    margin = (z / denom) * np.sqrt(
        p_hat * (1.0 - p_hat) / n + z2 / (4.0 * n * n)
    )

    # Apply continuity correction and clamp to [0, 1].
    lower = max(0.0, centre - margin - cc / denom)
    upper = min(1.0, centre + margin + cc / denom)
    width = upper - lower

    return lower, upper, width


def simulate_fer(H, decoder_config, noise_config, trials, seed=None,
                 ci_method=None, alpha=0.05, gamma=1.5,
                 early_stop_epsilon=None):
    """Run Monte Carlo FER simulation over a grid of error probabilities.

    Args:
        H: Binary parity-check matrix, shape (m, n).
        decoder_config: Dict of keyword arguments forwarded to
            :func:`bp_decode`.  Common keys: ``mode``, ``max_iters``,
            ``damping``, ``norm_factor``, ``offset``, ``clip``,
            ``postprocess``.
        noise_config: Dict with keys:

            - ``"p_grid"`` — list/array of physical error probabilities.
            - ``"bias"`` (optional) — bias parameter for :func:`channel_llr`.

        trials: Number of Monte Carlo frames per *p* value.
        seed: Master RNG seed for reproducibility.
        ci_method: Confidence interval method.  ``None`` (default) disables
            CI computation.  ``"wilson"`` computes the continuity-corrected
            Wilson score interval for each FER estimate.
        alpha: Significance level for the CI (default 0.05 → 95% CI).
        gamma: Continuity correction factor for the Wilson interval
            (default 1.5).
        early_stop_epsilon: If set (float > 0), stop trials for a given *p*
            once the CI width falls below this threshold.  Requires
            ``ci_method`` to be set.

    Returns:
        JSON-serializable dict::

            {
                "seed": <int or None>,
                "decoder": <decoder_config>,
                "noise": <noise_config (p_grid as list)>,
                "results": {
                    "p": [...],
                    "FER": [...],
                    "BER": [...],
                    "mean_iters": [...],
                    "actual_trials": [...]  // only when early_stop_epsilon set
                },
                "ci": {                     // only when ci_method is set
                    "method": "wilson",
                    "alpha": ...,
                    "gamma": ...,
                    "lower": [...],
                    "upper": [...],
                    "width": [...]
                }
            }
    """
    if trials < 1:
        raise ValueError(f"trials must be >= 1, got {trials}")

    # ── CI parameter validation ──
    if ci_method is not None and ci_method != "wilson":
        raise ValueError(
            f"ci_method must be None or 'wilson', got '{ci_method}'"
        )
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0.0, 1.0), got {alpha}")
    if gamma <= 0.0:
        raise ValueError(f"gamma must be > 0.0, got {gamma}")
    if early_stop_epsilon is not None:
        if early_stop_epsilon <= 0.0:
            raise ValueError(
                f"early_stop_epsilon must be > 0.0, got {early_stop_epsilon}"
            )
        if ci_method is None:
            raise ValueError(
                "early_stop_epsilon requires ci_method to be set"
            )

    rng = np.random.default_rng(seed)
    _, n = H.shape

    p_grid = list(np.asarray(noise_config["p_grid"], dtype=np.float64))
    bias = noise_config.get("bias", None)

    fer_list: list[float] = []
    ber_list: list[float] = []
    mean_iters_list: list[float] = []
    actual_trials_list: list[int] = []
    frame_errors_list: list[int] = []

    # Shallow copy of decoder_config.
    # Invalid keys are intentionally rejected by bp_decode's kwargs validation.
    dc = dict(decoder_config) if decoder_config else {}

    for p in p_grid:
        frame_errors = 0
        total_bit_errors = 0
        total_iters = 0
        actual_trial_count = 0

        for _ in range(trials):
            actual_trial_count += 1

            e = (rng.random(n) < p).astype(np.uint8)
            s = syndrome(H, e)
            llr = channel_llr(e, p, bias=bias)

            correction, iters = bp_decode(
                H, llr, syndrome_vec=s, **dc
            )

            total_iters += iters
            residual = e ^ correction
            if np.any(residual):
                frame_errors += 1
            total_bit_errors += int(np.sum(residual))

            # ── Early stop: check CI width after each trial ──
            # Requires at least 2 trials so the CI is meaningful.
            # The check is deterministic: same seed + same epsilon
            # produces the same break point.
            if early_stop_epsilon is not None and actual_trial_count >= 2:
                _, _, w = _wilson_ci(
                    frame_errors, actual_trial_count, alpha, gamma
                )
                if w < early_stop_epsilon:
                    break

        fer_list.append(float(frame_errors) / actual_trial_count)
        ber_list.append(float(total_bit_errors) / (actual_trial_count * n))
        mean_iters_list.append(float(total_iters) / actual_trial_count)
        actual_trials_list.append(actual_trial_count)
        frame_errors_list.append(frame_errors)

    # Ensure noise_config is JSON-safe (convert any numpy arrays).
    noise_safe = {
        "p_grid": [float(x) for x in noise_config["p_grid"]],
    }
    if bias is not None:
        noise_safe["bias"] = _json_safe(bias)

    results_dict = {
        "p": [float(x) for x in p_grid],
        "FER": fer_list,
        "BER": ber_list,
        "mean_iters": mean_iters_list,
    }

    # Include actual_trials only when early termination is active,
    # so existing consumers of v2.4.0 output are not surprised.
    if early_stop_epsilon is not None:
        results_dict["actual_trials"] = actual_trials_list

    result = {
        "seed": int(seed) if seed is not None else None,
        "decoder": {k: _json_safe(v) for k, v in dc.items()},
        "noise": noise_safe,
        "results": results_dict,
    }

    # ── Compute Wilson CI for each p-value ──
    if ci_method == "wilson":
        ci_lower: list[float] = []
        ci_upper: list[float] = []
        ci_width: list[float] = []
        for i in range(len(p_grid)):
            # Use stored integer frame error count directly (no float reconstruction).
            lo, hi, w = _wilson_ci(frame_errors_list[i], actual_trials_list[i], alpha, gamma)
            ci_lower.append(float(lo))
            ci_upper.append(float(hi))
            ci_width.append(float(w))

        result["ci"] = {
            "method": "wilson",
            "alpha": float(alpha),
            "gamma": float(gamma),
            "lower": ci_lower,
            "upper": ci_upper,
            "width": ci_width,
        }

    return result


def save_results(path, results_dict):
    """Write simulation results to a JSON file.

    Args:
        path: File path to write.
        results_dict: Dict returned by :func:`simulate_fer`.
    """
    with open(path, "w") as f:
        json.dump(results_dict, f, indent=2)


def _json_safe(obj):
    """Convert numpy scalars/arrays to JSON-serializable Python types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(x) for x in obj]
    return obj
