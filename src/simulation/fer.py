"""
Deterministic Monte Carlo frame-error-rate (FER) simulation harness.

All randomness flows through ``np.random.default_rng(seed)``.
Output is a plain-Python dict that is directly JSON-serializable.
"""

from __future__ import annotations

import json
import numpy as np

from ..qec_qldpc_codes import bp_decode, channel_llr, syndrome


def simulate_fer(H, decoder_config, noise_config, trials, seed=None):
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
                    "mean_iters": [...]
                }
            }
    """
    if trials < 1:
        raise ValueError(f"trials must be >= 1, got {trials}")
    rng = np.random.default_rng(seed)
    _, n = H.shape

    p_grid = list(np.asarray(noise_config["p_grid"], dtype=np.float64))
    bias = noise_config.get("bias", None)

    fer_list: list[float] = []
    ber_list: list[float] = []
    mean_iters_list: list[float] = []

    # Shallow copy of decoder_config.
    # Invalid keys are intentionally rejected by bp_decode's kwargs validation.
    dc = dict(decoder_config) if decoder_config else {}

    for p in p_grid:
        frame_errors = 0
        total_bit_errors = 0
        total_iters = 0

        for _ in range(trials):
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

        fer_list.append(float(frame_errors) / trials)
        ber_list.append(float(total_bit_errors) / (trials * n))
        mean_iters_list.append(float(total_iters) / trials)

    # Ensure noise_config is JSON-safe (convert any numpy arrays).
    noise_safe = {
        "p_grid": [float(x) for x in noise_config["p_grid"]],
    }
    if bias is not None:
        noise_safe["bias"] = _json_safe(bias)

    return {
        "seed": int(seed) if seed is not None else None,
        "decoder": {k: _json_safe(v) for k, v in dc.items()},
        "noise": noise_safe,
        "results": {
            "p": [float(x) for x in p_grid],
            "FER": fer_list,
            "BER": ber_list,
            "mean_iters": mean_iters_list,
        },
    }


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
