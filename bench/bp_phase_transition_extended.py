"""
v4.7.1 — Extended Noise Sweep (BP Phase Transition Probe).

Extends the initial experiments to higher noise levels:
  p in {0.01, 0.02, 0.03, 0.05, 0.08}

The initial sweep (p <= 0.015) showed freeze_probability = 0 everywhere.
This extended sweep probes whether higher noise activates BP metastability
and freeze dynamics.

Uses the same deterministic configuration, BSC syndrome-only channel,
and diagnostic stack as the initial experiments.

Analysis-only. Does not modify decoder code.
"""

from __future__ import annotations

import hashlib
import json
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

# ── Constants (identical to initial sweep) ───────────────────────────

SEED = 42
TRIALS = 200
MAX_ITERS = 50
BP_MODE = "min_sum"
SCHEDULE = "flooding"

# Extended noise levels.
DISTANCES = [5, 7, 9, 11, 13]
P_VALUES_EXTENDED = [0.01, 0.02, 0.03, 0.05, 0.08]


# ── Helpers (reused from initial sweep) ──────────────────────────────


def _pre_generate_instances(H, p, trials, rng):
    """Pre-generate error instances deterministically."""
    n = H.shape[1]
    instances = []
    for _ in range(trials):
        e = (rng.random(n) < p).astype(np.uint8)
        s = syndrome(H, e)
        llr = _BSC_SYNDROME.compute_llr(p, n)
        instances.append({"e": e, "s": s, "llr": llr})
    return instances


def _run_single_point(H, instances, max_iters=MAX_ITERS, bp_mode=BP_MODE):
    """Run baseline decoding on instances, collecting all diagnostics."""
    frame_errors = 0
    all_freeze_detections = []
    all_regime_traces = []
    all_bp_dynamics = []
    iter_counts = []

    for inst in instances:
        e = inst["e"]
        s = inst["s"]
        llr = inst["llr"].copy()

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

        s_correction = syndrome(H, correction)
        is_frame_error = not np.array_equal(s_correction, s)
        if is_frame_error:
            frame_errors += 1

        llr_trace_list = (
            [llr_hist[i] for i in range(llr_hist.shape[0])]
            if llr_hist is not None and llr_hist.shape[0] > 0
            else []
        )

        if len(list(trace)) < 2 or len(llr_trace_list) < 2:
            continue

        energy_list = list(trace)

        bp_dyn = compute_bp_dynamics_metrics(
            llr_trace=llr_trace_list,
            energy_trace=energy_list,
            correction_vectors=None,
        )
        all_bp_dynamics.append(bp_dyn)

        rt = compute_bp_regime_trace(
            llr_trace=llr_trace_list,
            energy_trace=energy_list,
            correction_vectors=None,
        )
        all_regime_traces.append(rt)

        fd = compute_bp_freeze_detection(
            llr_trace=llr_trace_list,
            energy_trace=energy_list,
        )
        all_freeze_detections.append(fd)

    n_trials = len(instances)
    fer = float(frame_errors) / n_trials if n_trials else 0.0

    freeze_count = sum(1 for fd in all_freeze_detections if fd["freeze_detected"])
    freeze_prob = float(freeze_count) / len(all_freeze_detections) if all_freeze_detections else 0.0

    frozen_iters = [fd["freeze_iteration"] for fd in all_freeze_detections if fd["freeze_detected"]]
    mean_freeze_iter = float(np.mean(frozen_iters)) if frozen_iters else None

    freeze_scores = [fd["freeze_score"] for fd in all_freeze_detections]
    mean_freeze_score = float(np.mean(freeze_scores)) if freeze_scores else 0.0

    metastable_count = sum(
        1 for rt in all_regime_traces
        if rt["summary"]["freeze_score"] > 0.5
    )
    metastable_prob = float(metastable_count) / len(all_regime_traces) if all_regime_traces else 0.0

    switch_rates = [rt["summary"]["switch_rate"] for rt in all_regime_traces]
    mean_switch_rate = float(np.mean(switch_rates)) if switch_rates else 0.0

    event_counts = [rt["summary"]["num_events"] for rt in all_regime_traces]
    mean_event_rate = float(np.mean(event_counts)) if event_counts else 0.0

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

    dyn_regime_counts = {}
    for bd in all_bp_dynamics:
        r = bd["regime"]
        dyn_regime_counts[r] = dyn_regime_counts.get(r, 0) + 1

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


# ── Extended Noise Sweep ─────────────────────────────────────────────


def extended_noise_sweep():
    """Run the full (distance x noise) grid at extended noise levels."""
    print("\n" + "=" * 72)
    print("EXTENDED NOISE SWEEP")
    print(f"  distances: {DISTANCES}")
    print(f"  p_values:  {P_VALUES_EXTENDED}")
    print(f"  trials:    {TRIALS}")
    print(f"  seed:      {SEED}")
    print("=" * 72)

    all_data = []

    for d in DISTANCES:
        H = create_code(name="rate_0.50", lifting_size=d, seed=SEED).H_X
        print(f"\n  Distance d={d}  (H: {H.shape[0]}x{H.shape[1]})")
        for p in P_VALUES_EXTENDED:
            print(f"    p={p:.3f} ...", end="", flush=True)
            rng = np.random.default_rng(SEED)
            instances = _pre_generate_instances(H, p, TRIALS, rng)
            r = _run_single_point(H, instances)
            r["distance"] = d
            r["noise"] = p
            all_data.append(r)
            print(f"  FER={r['fer']:.4f}  freeze_prob={r['freeze_probability']:.4f}  "
                  f"metastable_prob={r['metastable_probability']:.4f}  "
                  f"mean_freeze_score={r['mean_freeze_score']:.4f}  "
                  f"switch_rate={r['switch_rate']:.4f}  "
                  f"mean_iters={r['mean_iters']:.1f}  "
                  f"regimes={r['dynamics_regime_counts']}")

    # Print summary table.
    print("\n" + "=" * 120)
    print("EXTENDED SWEEP RESULTS")
    print("=" * 120)
    print(f"{'distance':>8} | {'noise':>8} | {'FER':>8} | {'freeze_prob':>11} | "
          f"{'mean_freeze_score':>17} | {'metastable_prob':>15} | {'switch_rate':>11} | "
          f"{'event_rate':>10} | {'mean_iters':>10}")
    print("-" * 120)
    for r in all_data:
        mfi = f"{r['mean_freeze_iteration']:.1f}" if r['mean_freeze_iteration'] is not None else "N/A"
        print(f"{r['distance']:>8} | {r['noise']:>8.3f} | {r['fer']:>8.4f} | "
              f"{r['freeze_probability']:>11.4f} | "
              f"{r['mean_freeze_score']:>17.4f} | "
              f"{r['metastable_probability']:>15.4f} | "
              f"{r['switch_rate']:>11.4f} | "
              f"{r['event_rate']:>10.4f} | "
              f"{r['mean_iters']:>10.1f}")

    return all_data


# ── Phase Transition Plot ────────────────────────────────────────────


def generate_extended_plot(all_data):
    """Generate extended BP phase transition plot."""
    print("\n" + "=" * 72)
    print("GENERATING EXTENDED PHASE TRANSITION PLOT")
    print("=" * 72)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Build curves.
    freeze_curves = {}
    fer_curves = {}
    score_curves = {}
    meta_curves = {}
    switch_curves = {}
    iter_curves = {}
    for entry in all_data:
        d = entry["distance"]
        p = entry["noise"]
        for curves, key in [
            (freeze_curves, "freeze_probability"),
            (fer_curves, "fer"),
            (score_curves, "mean_freeze_score"),
            (meta_curves, "metastable_probability"),
            (switch_curves, "switch_rate"),
            (iter_curves, "mean_iters"),
        ]:
            if d not in curves:
                curves[d] = {}
            curves[d][p] = entry[key]

    markers = ['o', 's', '^', 'D', 'v']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    fig, axes = plt.subplots(2, 3, figsize=(18, 10), dpi=150)

    def _plot_panel(ax, curves, ylabel, title, ylim=None):
        for i, d in enumerate(sorted(curves.keys())):
            noise_vals = sorted(curves[d].keys())
            vals = [curves[d][p] for p in noise_vals]
            ax.plot(noise_vals, vals,
                    marker=markers[i % len(markers)],
                    color=colors[i % len(colors)],
                    linewidth=2, markersize=7, label=f"d={d}")
        ax.set_xlabel("Noise Level (p)", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(left=0)
        if ylim is not None:
            ax.set_ylim(*ylim)

    _plot_panel(axes[0, 0], fer_curves,
                "Frame Error Rate", "FER vs Noise (BSC Syndrome Channel)",
                ylim=(0, 1.05))
    _plot_panel(axes[0, 1], freeze_curves,
                "Freeze Probability", "BP Freeze Probability vs Noise",
                ylim=(-0.05, 1.05))
    _plot_panel(axes[0, 2], meta_curves,
                "Metastable Probability", "Regime-Trace Metastability vs Noise",
                ylim=(-0.05, 1.05))
    _plot_panel(axes[1, 0], score_curves,
                "Mean Freeze Score", "BP Freeze Score vs Noise")
    _plot_panel(axes[1, 1], switch_curves,
                "Switch Rate", "Regime Switch Rate vs Noise")
    _plot_panel(axes[1, 2], iter_curves,
                "Mean Iterations", "Mean BP Iterations vs Noise")

    plt.suptitle("v4.7.1 Extended BP Phase Transition Analysis\n"
                 "p in {0.01, 0.02, 0.03, 0.05, 0.08}",
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig("bp_phase_transition_plot_extended.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: bp_phase_transition_plot_extended.png")
    return True


# ── Determinism Verification ─────────────────────────────────────────


def verify_determinism_extended():
    """Run one (d, p) point twice, compare outputs, compare JSON hashes."""
    print("\n" + "=" * 72)
    print("DETERMINISM VERIFICATION (extended sweep)")
    print("=" * 72)

    d = 9
    p = 0.05
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
              f"metastable_prob={r['metastable_probability']:.6f}  "
              f"switch_rate={r['switch_rate']:.6f}")

    match = True
    compare_keys = ["fer", "frame_errors", "freeze_probability",
                    "mean_freeze_score", "metastable_probability",
                    "switch_rate", "event_rate", "mean_iters",
                    "traces_analyzed"]
    for key in compare_keys:
        if runs[0][key] != runs[1][key]:
            print(f"  MISMATCH on {key}: {runs[0][key]} vs {runs[1][key]}")
            match = False

    if runs[0]["dynamics_regime_counts"] != runs[1]["dynamics_regime_counts"]:
        print("  MISMATCH on dynamics_regime_counts")
        match = False
    if runs[0]["regime_frequencies"] != runs[1]["regime_frequencies"]:
        print("  MISMATCH on regime_frequencies")
        match = False

    # JSON hash comparison.
    json_1 = json.dumps(_serialize_results([runs[0]]), sort_keys=True)
    json_2 = json.dumps(_serialize_results([runs[1]]), sort_keys=True)
    hash_1 = hashlib.sha256(json_1.encode()).hexdigest()
    hash_2 = hashlib.sha256(json_2.encode()).hexdigest()
    print(f"  JSON hash run 1: {hash_1}")
    print(f"  JSON hash run 2: {hash_2}")
    if hash_1 != hash_2:
        print("  MISMATCH on JSON hash")
        match = False

    if match:
        print("  DETERMINISM CHECK: PASSED — all outputs and JSON hashes identical")
    else:
        print("  DETERMINISM CHECK: FAILED — outputs differ")

    return match, hash_1


# ── Main ─────────────────────────────────────────────────────────────


def main():
    """Run extended noise sweep and produce all artifacts."""
    print("v4.7.1 — Extended Noise Sweep (BP Phase Transition Probe)")
    print(f"Seed: {SEED}")
    print(f"Trials: {TRIALS}")
    print(f"Max iters: {MAX_ITERS}")
    print(f"BP mode: {BP_MODE}")
    print(f"Channel: BSC syndrome-only (no sign leakage)")
    print(f"Extended noise: {P_VALUES_EXTENDED}")

    # Run the full sweep.
    all_data = extended_noise_sweep()

    # Generate plot.
    generate_extended_plot(all_data)

    # Determinism verification.
    det_ok, det_hash = verify_determinism_extended()

    # Save JSON.
    output = {
        "version": "v4.7.1-extended",
        "config": {
            "seed": SEED,
            "trials": TRIALS,
            "max_iters": MAX_ITERS,
            "bp_mode": BP_MODE,
            "schedule": SCHEDULE,
            "channel": "bsc_syndrome",
            "distances": DISTANCES,
            "p_values": P_VALUES_EXTENDED,
        },
        "extended_noise_sweep": _serialize_results(all_data),
        "determinism_check": {
            "passed": det_ok,
            "json_hash": det_hash,
            "config": {"distance": 9, "noise": 0.05},
        },
    }

    with open("bp_phase_transition_data_extended.json", "w") as f:
        json.dump(output, f, indent=2, sort_keys=False)
    print(f"\nSaved: bp_phase_transition_data_extended.json")

    print("\n" + "=" * 72)
    print("EXTENDED SWEEP COMPLETE")
    print("=" * 72)

    return output


if __name__ == "__main__":
    main()
