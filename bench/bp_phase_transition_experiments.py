"""
v4.7.1 — BP Phase Transition Experiments.

Structured experiments analyzing BP dynamics, metastability, and freeze
behavior across code distances and noise levels.

Uses BSC syndrome-only channel model: the decoder receives uniform LLR
with no per-variable sign information, forcing it to rely entirely on
syndrome constraints.  This produces realistic (non-zero) FER and
non-trivial BP iteration traces.

Analysis-only. Does not modify decoder code.
All artifacts are deterministic and JSON-serializable.

Experiments:
  1. Distance scaling of freeze probability
  2. Noise threshold for freeze onset
  3. DPS inversion diagnostic
  4. Metastability scaling (spin-glass diagnostic)
  + Phase transition plot
  + Determinism verification
"""

from __future__ import annotations

import json
import math
import sys
import os

import numpy as np

# Ensure project root is on path.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from qec_qldpc_codes import bp_decode, syndrome, create_code
from qec.channel.bsc_syndrome import BSCSyndromeChannel
from qec.diagnostics.bp_dynamics import compute_bp_dynamics_metrics
from qec.diagnostics.bp_regime_trace import compute_bp_regime_trace
from qec.diagnostics.bp_freeze_detection import compute_bp_freeze_detection

# BSC syndrome-only channel: uniform LLR, no sign leakage.
_BSC_SYNDROME = BSCSyndromeChannel()

# ── Constants ────────────────────────────────────────────────────────

SEED = 42
TRIALS = 200
MAX_ITERS = 50
BP_MODE = "min_sum"
SCHEDULE = "flooding"


# ── Helpers ──────────────────────────────────────────────────────────


def _pre_generate_instances(H, p, trials, rng):
    """Pre-generate error instances deterministically.

    Uses BSC syndrome-only channel: uniform LLR magnitude with no sign
    leakage, so the decoder must rely entirely on syndrome constraints.
    """
    n = H.shape[1]
    instances = []
    for _ in range(trials):
        e = (rng.random(n) < p).astype(np.uint8)
        s = syndrome(H, e)
        llr = _BSC_SYNDROME.compute_llr(p, n)
        instances.append({"e": e, "s": s, "llr": llr})
    return instances


def _run_single_point(H, instances, max_iters=MAX_ITERS, bp_mode=BP_MODE):
    """Run baseline decoding on instances, collecting all diagnostics.

    Returns dict with FER, freeze metrics, regime trace metrics, and
    per-iteration dynamics analysis.
    """
    frame_errors = 0
    all_freeze_detections = []
    all_regime_traces = []
    all_bp_dynamics = []
    iter_counts = []

    for inst in instances:
        e = inst["e"]
        s = inst["s"]
        llr = inst["llr"].copy()

        # Decode with energy trace and LLR history.
        result = bp_decode(
            H, llr,
            max_iters=max_iters,
            mode=bp_mode,
            schedule=SCHEDULE,
            syndrome_vec=s,
            energy_trace=True,
            llr_history=max_iters,
        )
        correction, iters = result[0], result[1]
        llr_hist = result[2]
        trace = result[-1]
        iter_counts.append(iters)

        # FER: syndrome-consistency semantics.
        s_correction = syndrome(H, correction)
        is_frame_error = not np.array_equal(s_correction, s)
        if is_frame_error:
            frame_errors += 1

        # Build LLR trace list.
        llr_trace_list = (
            [llr_hist[i] for i in range(llr_hist.shape[0])]
            if llr_hist is not None and llr_hist.shape[0] > 0
            else []
        )

        if len(list(trace)) < 2 or len(llr_trace_list) < 2:
            continue

        energy_list = list(trace)

        # BP dynamics.
        bp_dyn = compute_bp_dynamics_metrics(
            llr_trace=llr_trace_list,
            energy_trace=energy_list,
            correction_vectors=None,
        )
        all_bp_dynamics.append(bp_dyn)

        # Regime trace.
        rt = compute_bp_regime_trace(
            llr_trace=llr_trace_list,
            energy_trace=energy_list,
            correction_vectors=None,
        )
        all_regime_traces.append(rt)

        # Freeze detection.
        fd = compute_bp_freeze_detection(
            llr_trace=llr_trace_list,
            energy_trace=energy_list,
        )
        all_freeze_detections.append(fd)

    n_trials = len(instances)
    fer = float(frame_errors) / n_trials if n_trials else 0.0

    # Freeze statistics from bp_freeze_detection.
    freeze_count = sum(1 for fd in all_freeze_detections if fd["freeze_detected"])
    freeze_prob = float(freeze_count) / len(all_freeze_detections) if all_freeze_detections else 0.0

    frozen_iters = [fd["freeze_iteration"] for fd in all_freeze_detections if fd["freeze_detected"]]
    mean_freeze_iter = float(np.mean(frozen_iters)) if frozen_iters else None

    freeze_scores = [fd["freeze_score"] for fd in all_freeze_detections]
    mean_freeze_score = float(np.mean(freeze_scores)) if freeze_scores else 0.0

    # Metastable probability from regime traces (freeze_score > 0.5).
    metastable_count = sum(
        1 for rt in all_regime_traces
        if rt["summary"]["freeze_score"] > 0.5
    )
    metastable_prob = float(metastable_count) / len(all_regime_traces) if all_regime_traces else 0.0

    # Switch rate from regime traces.
    switch_rates = [rt["summary"]["switch_rate"] for rt in all_regime_traces]
    mean_switch_rate = float(np.mean(switch_rates)) if switch_rates else 0.0

    # Event rate.
    event_counts = [rt["summary"]["num_events"] for rt in all_regime_traces]
    mean_event_rate = float(np.mean(event_counts)) if event_counts else 0.0

    # Regime frequencies.
    regime_counts = {}
    total_labels = 0
    for rt in all_regime_traces:
        for regime in rt["regime_trace"]:
            regime_counts[regime] = regime_counts.get(regime, 0) + 1
            total_labels += 1
    regime_freq = {}
    if total_labels > 0:
        for k in sorted(regime_counts.keys()):
            regime_freq[k] = float(regime_counts[k]) / float(total_labels)

    # BP dynamics regime distribution.
    dyn_regime_counts = {}
    for bd in all_bp_dynamics:
        r = bd["regime"]
        dyn_regime_counts[r] = dyn_regime_counts.get(r, 0) + 1

    # Mean iterations.
    mean_iters = float(np.mean(iter_counts)) if iter_counts else 0.0

    return {
        "fer": fer,
        "frame_errors": frame_errors,
        "trials": n_trials,
        "mean_iters": mean_iters,
        "freeze_probability": freeze_prob,
        "mean_freeze_iteration": mean_freeze_iter,
        "mean_freeze_score": mean_freeze_score,
        "metastable_probability": metastable_prob,
        "switch_rate": mean_switch_rate,
        "event_rate": mean_event_rate,
        "regime_frequencies": regime_freq,
        "dynamics_regime_counts": dyn_regime_counts,
        "traces_analyzed": len(all_freeze_detections),
    }


# ── Experiment 1: Distance Scaling of Freeze Probability ────────────


def experiment_1_distance_scaling():
    """Freeze probability vs code distance at fixed noise."""
    print("\n" + "=" * 72)
    print("EXPERIMENT 1: Distance Scaling of Freeze Probability")
    print("=" * 72)

    distances = [5, 7, 9, 11, 13]
    p = 0.007
    results = []

    for d in distances:
        print(f"  Running d={d}, p={p}, trials={TRIALS}...")
        rng = np.random.default_rng(SEED)
        H = create_code(name="rate_0.50", lifting_size=d, seed=SEED).H_X
        instances = _pre_generate_instances(H, p, TRIALS, rng)
        r = _run_single_point(H, instances)
        r["distance"] = d
        r["noise"] = p
        results.append(r)
        print(f"    FER={r['fer']:.4f}  freeze_prob={r['freeze_probability']:.4f}  "
              f"metastable_prob={r['metastable_probability']:.4f}  "
              f"mean_iters={r['mean_iters']:.1f}  "
              f"regimes={r['dynamics_regime_counts']}")

    # Print table.
    print(f"\n{'distance':>8} | {'FER':>8} | {'freeze_prob':>11} | {'mean_freeze_iter':>16} | "
          f"{'metastable_prob':>15} | {'switch_rate':>11} | {'mean_iters':>10}")
    print("-" * 95)
    for r in results:
        mfi = f"{r['mean_freeze_iteration']:.1f}" if r['mean_freeze_iteration'] is not None else "N/A"
        print(f"{r['distance']:>8} | {r['fer']:>8.4f} | {r['freeze_probability']:>11.4f} | {mfi:>16} | "
              f"{r['metastable_probability']:>15.4f} | {r['switch_rate']:>11.4f} | {r['mean_iters']:>10.1f}")

    return results


# ── Experiment 2: Noise Threshold for Freeze Onset ──────────────────


def experiment_2_noise_threshold():
    """Freeze probability vs noise at fixed distance d=9."""
    print("\n" + "=" * 72)
    print("EXPERIMENT 2: Noise Threshold for Freeze Onset (d=9)")
    print("=" * 72)

    d = 9
    p_values = [0.001, 0.003, 0.005, 0.007, 0.01, 0.015]
    results = []

    H = create_code(name="rate_0.50", lifting_size=d, seed=SEED).H_X

    for p in p_values:
        print(f"  Running d={d}, p={p}, trials={TRIALS}...")
        rng = np.random.default_rng(SEED)
        instances = _pre_generate_instances(H, p, TRIALS, rng)
        r = _run_single_point(H, instances)
        r["distance"] = d
        r["noise"] = p
        results.append(r)
        print(f"    FER={r['fer']:.4f}  freeze_prob={r['freeze_probability']:.4f}  "
              f"metastable_prob={r['metastable_probability']:.4f}  "
              f"event_rate={r['event_rate']:.4f}")

    # Print table.
    print(f"\n{'noise':>8} | {'FER':>8} | {'freeze_prob':>11} | {'metastable_prob':>15} | "
          f"{'event_rate':>10} | {'mean_iters':>10}")
    print("-" * 72)
    for r in results:
        print(f"{r['noise']:>8.4f} | {r['fer']:>8.4f} | {r['freeze_probability']:>11.4f} | "
              f"{r['metastable_probability']:>15.4f} | {r['event_rate']:>10.4f} | "
              f"{r['mean_iters']:>10.1f}")

    return results


# ── Experiment 3: DPS Inversion Diagnostic ──────────────────────────


def experiment_3_dps_inversion():
    """Test whether DPS inversion correlates with freeze dynamics."""
    print("\n" + "=" * 72)
    print("EXPERIMENT 3: DPS Inversion Diagnostic")
    print("=" * 72)

    distances = [7, 11]
    p_values = [0.001, 0.003, 0.005, 0.007, 0.01, 0.015]
    results = []

    for d in distances:
        H = create_code(name="rate_0.50", lifting_size=d, seed=SEED).H_X
        for p in p_values:
            print(f"  Running d={d}, p={p}, trials={TRIALS}...")
            rng = np.random.default_rng(SEED)
            instances = _pre_generate_instances(H, p, TRIALS, rng)
            r = _run_single_point(H, instances)
            r["distance"] = d
            r["noise"] = p
            results.append(r)
            print(f"    FER={r['fer']:.4f}  freeze_prob={r['freeze_probability']:.4f}")

    # Print table.
    print(f"\n{'distance':>8} | {'noise':>8} | {'FER':>8} | {'freeze_prob':>11} | "
          f"{'mean_freeze_iter':>16} | {'mean_iters':>10}")
    print("-" * 72)
    for r in results:
        mfi = f"{r['mean_freeze_iteration']:.1f}" if r['mean_freeze_iteration'] is not None else "N/A"
        print(f"{r['distance']:>8} | {r['noise']:>8.4f} | {r['fer']:>8.4f} | "
              f"{r['freeze_probability']:>11.4f} | {mfi:>16} | {r['mean_iters']:>10.1f}")

    # Compute DPS slopes.
    for d in distances:
        d_results = [r for r in results if r["distance"] == d]
        if len(d_results) >= 2:
            fer_vals = {r["noise"]: r["fer"] for r in d_results}
            print(f"\n  d={d} FER profile: {fer_vals}")

    return results


# ── Experiment 4: Metastability Scaling ─────────────────────────────


def experiment_4_metastability_scaling():
    """Metastability vs distance at fixed noise (spin-glass test)."""
    print("\n" + "=" * 72)
    print("EXPERIMENT 4: Metastability Scaling (Spin-Glass Diagnostic)")
    print("=" * 72)

    distances = [5, 7, 9, 11, 13]
    p = 0.007
    results = []

    for d in distances:
        print(f"  Running d={d}, p={p}, trials={TRIALS}...")
        rng = np.random.default_rng(SEED)
        H = create_code(name="rate_0.50", lifting_size=d, seed=SEED).H_X
        instances = _pre_generate_instances(H, p, TRIALS, rng)
        r = _run_single_point(H, instances)
        r["distance"] = d
        r["noise"] = p
        results.append(r)
        print(f"    metastable_prob={r['metastable_probability']:.4f}  "
              f"freeze_prob={r['freeze_probability']:.4f}  "
              f"FER={r['fer']:.4f}")

    # Print table.
    print(f"\n{'distance':>8} | {'FER':>8} | {'metastable_prob':>15} | "
          f"{'freeze_prob':>11} | {'mean_freeze_iter':>16} | {'mean_freeze_score':>16}")
    print("-" * 85)
    for r in results:
        mfi = f"{r['mean_freeze_iteration']:.1f}" if r['mean_freeze_iteration'] is not None else "N/A"
        print(f"{r['distance']:>8} | {r['fer']:>8.4f} | {r['metastable_probability']:>15.4f} | "
              f"{r['freeze_probability']:>11.4f} | {mfi:>16} | {r['mean_freeze_score']:>16.4f}")

    # Fit simple scaling: freeze_prob vs distance.
    d_arr = np.array([r["distance"] for r in results], dtype=np.float64)
    fp_arr = np.array([r["freeze_probability"] for r in results], dtype=np.float64)
    fer_arr = np.array([r["fer"] for r in results], dtype=np.float64)

    if len(d_arr) >= 2:
        A = np.vstack([d_arr, np.ones(len(d_arr))]).T
        coef_fp, _, _, _ = np.linalg.lstsq(A, fp_arr, rcond=None)
        coef_fer, _, _, _ = np.linalg.lstsq(A, fer_arr, rcond=None)
        print(f"\n  Linear fit (freeze_prob): slope={coef_fp[0]:.6f}  intercept={coef_fp[1]:.6f}")
        print(f"  Linear fit (FER):         slope={coef_fer[0]:.6f}  intercept={coef_fer[1]:.6f}")
        scaling = {
            "freeze_slope": float(coef_fp[0]),
            "freeze_intercept": float(coef_fp[1]),
            "fer_slope": float(coef_fer[0]),
            "fer_intercept": float(coef_fer[1]),
        }
        return results, scaling

    return results, None


# ── Phase Transition Plot ───────────────────────────────────────────


def generate_phase_transition_plot(all_data):
    """Generate BP phase transition plot with two panels:
    1. Freeze probability vs noise per distance
    2. FER vs noise per distance
    """
    print("\n" + "=" * 72)
    print("GENERATING PHASE TRANSITION PLOT")
    print("=" * 72)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  WARNING: matplotlib not available. Skipping plot generation.")
        return False

    # Build curves from all_data.
    freeze_curves = {}  # distance -> {noise: freeze_prob}
    fer_curves = {}     # distance -> {noise: FER}
    score_curves = {}   # distance -> {noise: mean_freeze_score}
    meta_curves = {}    # distance -> {noise: metastable_prob}
    for entry in all_data:
        d = entry["distance"]
        p = entry["noise"]
        for curves, key in [(freeze_curves, "freeze_probability"),
                             (fer_curves, "fer"),
                             (score_curves, "mean_freeze_score"),
                             (meta_curves, "metastable_probability")]:
            if d not in curves:
                curves[d] = {}
            curves[d][p] = entry[key]

    markers = ['o', 's', '^', 'D', 'v']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=150)

    # Panel 1: FER vs noise.
    ax = axes[0, 0]
    for i, d in enumerate(sorted(fer_curves.keys())):
        noise_vals = sorted(fer_curves[d].keys())
        vals = [fer_curves[d][p] for p in noise_vals]
        ax.plot(noise_vals, vals,
                marker=markers[i % len(markers)],
                color=colors[i % len(colors)],
                linewidth=2, markersize=7, label=f"d={d}")
    ax.set_xlabel("Noise Level (p)")
    ax.set_ylabel("Frame Error Rate")
    ax.set_title("FER vs Noise (BSC Syndrome Channel)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0, top=1.05)

    # Panel 2: Freeze probability vs noise.
    ax = axes[0, 1]
    for i, d in enumerate(sorted(freeze_curves.keys())):
        noise_vals = sorted(freeze_curves[d].keys())
        vals = [freeze_curves[d][p] for p in noise_vals]
        ax.plot(noise_vals, vals,
                marker=markers[i % len(markers)],
                color=colors[i % len(colors)],
                linewidth=2, markersize=7, label=f"d={d}")
    ax.set_xlabel("Noise Level (p)")
    ax.set_ylabel("Freeze Probability")
    ax.set_title("BP Freeze Probability vs Noise")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=-0.05, top=1.05)

    # Panel 3: Metastable probability vs noise.
    ax = axes[1, 0]
    for i, d in enumerate(sorted(meta_curves.keys())):
        noise_vals = sorted(meta_curves[d].keys())
        vals = [meta_curves[d][p] for p in noise_vals]
        ax.plot(noise_vals, vals,
                marker=markers[i % len(markers)],
                color=colors[i % len(colors)],
                linewidth=2, markersize=7, label=f"d={d}")
    ax.set_xlabel("Noise Level (p)")
    ax.set_ylabel("Metastable Probability")
    ax.set_title("Regime-Trace Metastability vs Noise")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=-0.05, top=1.05)

    # Panel 4: Mean freeze score vs noise.
    ax = axes[1, 1]
    for i, d in enumerate(sorted(score_curves.keys())):
        noise_vals = sorted(score_curves[d].keys())
        vals = [score_curves[d][p] for p in noise_vals]
        ax.plot(noise_vals, vals,
                marker=markers[i % len(markers)],
                color=colors[i % len(colors)],
                linewidth=2, markersize=7, label=f"d={d}")
    ax.set_xlabel("Noise Level (p)")
    ax.set_ylabel("Mean Freeze Score")
    ax.set_title("BP Freeze Score vs Noise")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)

    plt.suptitle("v4.7.1 BP Phase Transition Analysis", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig("bp_phase_transition_plot.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: bp_phase_transition_plot.png")
    return True


# ── Determinism Verification ────────────────────────────────────────


def verify_determinism():
    """Run experiment 1 (d=7 only) twice and compare all outputs."""
    print("\n" + "=" * 72)
    print("DETERMINISM VERIFICATION")
    print("=" * 72)

    d = 7
    p = 0.007
    runs = []

    for run_idx in range(2):
        rng = np.random.default_rng(SEED)
        H = create_code(name="rate_0.50", lifting_size=d, seed=SEED).H_X
        instances = _pre_generate_instances(H, p, TRIALS, rng)
        r = _run_single_point(H, instances)
        runs.append(r)
        print(f"  Run {run_idx + 1}: FER={r['fer']:.6f}  "
              f"freeze_prob={r['freeze_probability']:.6f}  "
              f"mean_freeze_score={r['mean_freeze_score']:.6f}  "
              f"metastable_prob={r['metastable_probability']:.6f}")

    # Compare all scalar fields.
    match = True
    compare_keys = ["fer", "frame_errors", "freeze_probability",
                    "mean_freeze_score", "metastable_probability",
                    "switch_rate", "event_rate", "mean_iters",
                    "traces_analyzed"]
    for key in compare_keys:
        if runs[0][key] != runs[1][key]:
            print(f"  MISMATCH on {key}: {runs[0][key]} vs {runs[1][key]}")
            match = False

    # Compare regime distributions.
    if runs[0]["dynamics_regime_counts"] != runs[1]["dynamics_regime_counts"]:
        print(f"  MISMATCH on dynamics_regime_counts")
        match = False
    if runs[0]["regime_frequencies"] != runs[1]["regime_frequencies"]:
        print(f"  MISMATCH on regime_frequencies")
        match = False

    if match:
        print("  DETERMINISM CHECK: PASSED — all outputs identical across both runs")
    else:
        print("  DETERMINISM CHECK: FAILED — outputs differ")

    return match


# ── Full Noise Sweep Per Distance ───────────────────────────────────


def full_noise_sweep():
    """Run noise sweep for all distances (needed for phase plot)."""
    print("\n" + "=" * 72)
    print("FULL NOISE SWEEP (for phase transition plot)")
    print("=" * 72)

    distances = [5, 7, 9, 11, 13]
    p_values = [0.001, 0.003, 0.005, 0.007, 0.01, 0.015]
    all_data = []

    for d in distances:
        H = create_code(name="rate_0.50", lifting_size=d, seed=SEED).H_X
        for p in p_values:
            print(f"  Running d={d}, p={p}...")
            rng = np.random.default_rng(SEED)
            instances = _pre_generate_instances(H, p, TRIALS, rng)
            r = _run_single_point(H, instances)
            r["distance"] = d
            r["noise"] = p
            all_data.append(r)
            print(f"    FER={r['fer']:.4f}  freeze_prob={r['freeze_probability']:.4f}  "
                  f"mean_iters={r['mean_iters']:.1f}")

    return all_data


# ── Main ────────────────────────────────────────────────────────────


def main():
    """Run all v4.7.1 BP phase transition experiments."""
    print("v4.7.1 — BP Phase Transition Experiments")
    print(f"Seed: {SEED}")
    print(f"Trials: {TRIALS}")
    print(f"Max iters: {MAX_ITERS}")
    print(f"BP mode: {BP_MODE}")
    print(f"Channel: BSC syndrome-only (no sign leakage)")

    # Experiment 1.
    exp1 = experiment_1_distance_scaling()

    # Experiment 2.
    exp2 = experiment_2_noise_threshold()

    # Experiment 3.
    exp3 = experiment_3_dps_inversion()

    # Experiment 4.
    exp4, scaling_fit = experiment_4_metastability_scaling()

    # Full noise sweep for phase plot.
    all_data = full_noise_sweep()

    # Phase transition plot.
    plot_ok = generate_phase_transition_plot(all_data)

    # Determinism verification.
    det_ok = verify_determinism()

    # Save JSON data.
    output = {
        "version": "v4.7.1",
        "config": {
            "seed": SEED,
            "trials": TRIALS,
            "max_iters": MAX_ITERS,
            "bp_mode": BP_MODE,
            "schedule": SCHEDULE,
            "channel": "bsc_syndrome",
        },
        "experiment_1_distance_scaling": _serialize_results(exp1),
        "experiment_2_noise_threshold": _serialize_results(exp2),
        "experiment_3_dps_inversion": _serialize_results(exp3),
        "experiment_4_metastability_scaling": {
            "results": _serialize_results(exp4),
            "scaling_fit": scaling_fit,
        },
        "full_noise_sweep": _serialize_results(all_data),
        "determinism_check": {
            "passed": det_ok,
        },
    }

    with open("bp_phase_transition_data.json", "w") as f:
        json.dump(output, f, indent=2, sort_keys=False)
    print(f"\nSaved: bp_phase_transition_data.json")

    print("\n" + "=" * 72)
    print("ALL EXPERIMENTS COMPLETE")
    print("=" * 72)

    return output


def _serialize_results(results):
    """Make results JSON-serializable."""
    out = []
    for r in results:
        entry = {}
        for k, v in r.items():
            if isinstance(v, (np.integer, np.int64)):
                entry[k] = int(v)
            elif isinstance(v, (np.floating, np.float64)):
                entry[k] = float(v)
            elif isinstance(v, np.ndarray):
                entry[k] = v.tolist()
            else:
                entry[k] = v
        out.append(entry)
    return out


if __name__ == "__main__":
    main()
