"""CPU-only, phenomenological X-fault surface-memory correlation experiment.

Fault histories are propagated through a real Stim Clifford circuit by GF(2)
linearity, not by assigning synthetic detector or logical labels. See the
accompanying protocol for the deliberately narrow physical model.
"""
from __future__ import annotations

import hashlib
import json
import math
import platform
from pathlib import Path

import numpy as np

from . import imports

MODEL_VERSION = "surface-memory-x-markov-v1"


def _hash(value):
    if not isinstance(value, bytes):
        value = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(value).hexdigest()


def build_memory_circuit(distance, rounds, probabilities):
    """Inject X_ERROR on data immediately before each syndrome round's first H.

    probabilities has shape (rounds, distance**2), in final-M target order.
    Gates, ancillas, resets and measurements are otherwise perfect.
    """
    imports.require_stim("surface memory correlation benchmark")
    base = imports.stim.Circuit.generated(
        "surface_code:rotated_memory_z", distance=distance, rounds=rounds,
    ).flattened()
    data = [t.value for op in base if op.name == "M" for t in op.targets_copy()]
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.shape != (rounds, len(data)):
        raise ValueError("probabilities must have shape (rounds, distance**2)")
    if not np.all(np.isfinite(probabilities)) or np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("probabilities must be finite and within [0, 1]")
    result = imports.stim.Circuit()
    round_index = 0
    start = True
    for op in base:
        if op.name == "H" and start:
            for qubit, p in zip(data, probabilities[round_index]):
                if p:
                    result.append("X_ERROR", [qubit], float(p))
            round_index += 1
            start = False
        result.append(op)
        if op.name == "MR":
            start = True
    if round_index != rounds or len(data) != distance**2:
        raise RuntimeError("unsupported generated circuit structure")
    return result, data


def physical_rates(distance, p, placement, homogeneous=False):
    """Equal-mean synthetic hot stripe versus dispersed diagonal in data grid."""
    if placement not in ("stripe", "diagonal"):
        raise ValueError("placement must be stripe or diagonal")
    if homogeneous:
        return np.full(distance**2, p)
    # Final-M ordering is row-major for Stim's rotated-memory generator.
    # This ordering is checked against circuit coordinates in run_experiment.
    hot = np.zeros((distance, distance), dtype=bool)
    if placement == "stripe":
        hot[0, :] = True
    else:
        np.fill_diagonal(hot, True)
    return np.where(hot.ravel(), p * (distance + 1) / 2, p / 2)


def sample_faults(rates, rounds, shots, rho, seed):
    """Stationary binary Markov faults; P(1|x)=(1-rho)*p+rho*x.

    Every site has marginal p at every round, including the first; different
    sites and shots are independent. rho=0 is precisely the IID control.
    """
    rng = np.random.default_rng(seed)
    faults = np.empty((shots, rounds, len(rates)), dtype=np.uint8)
    faults[:, 0] = rng.random((shots, len(rates))) < rates
    for t in range(1, rounds):
        probability = (1 - rho) * rates + rho * faults[:, t - 1]
        faults[:, t] = rng.random((shots, len(rates))) < probability
    return faults


def fault_response(distance, rounds):
    """Exact detector/observable responses to each deterministic physical fault."""
    ideal, data = build_memory_circuit(distance, rounds, np.zeros((rounds, distance**2)))
    boundaries = []
    start = True
    for index, op in enumerate(ideal):
        if op.name == "H" and start:
            boundaries.append(index)
            start = False
        if op.name == "MR":
            start = True
    response = []
    # Circuit slicing/concatenation runs in Stim rather than rebuilding every
    # instruction in Python for each of O(rounds*distance**2) fault locations.
    for boundary in boundaries:
        for qubit in data:
            circuit = ideal[:boundary]
            circuit.append("X_ERROR", [qubit], 1.)
            circuit += ideal[boundary:]
            response.append(circuit.compile_detector_sampler(seed=0).sample(1, append_observables=True)[0])
    return np.asarray(response, dtype=np.uint8)


def propagate_faults(faults, response):
    # uint8 overflow is harmless modulo two.
    return (faults.reshape(len(faults), -1) @ response) % 2


def wilson_interval(failures, shots):
    z = 1.959963984540054
    phat = failures / shots
    den = 1 + z*z/shots
    center = (phat + z*z/(2*shots)) / den
    half = z * math.sqrt(phat*(1-phat)/shots + z*z/(4*shots*shots)) / den
    return [max(0., center-half), min(1., center+half)]


def detector_correlations(events, circuit):
    """Per-pair adjacent-time covariance for identical stabilizer coordinates.

    Nonstationary boundary detector pairs are reported explicitly, not pooled
    into a Pearson coefficient; zero variance is not called zero correlation.
    """
    coords = circuit.get_detector_coordinates()
    lookup = {tuple(xyz): i for i, xyz in coords.items()}
    pairs = []
    for i, xyz in coords.items():
        j = lookup.get(tuple(xyz[:2] + [xyz[2] + 1]))
        if j is None:
            continue
        x, y = events[:, i].astype(float), events[:, j].astype(float)
        covariance = float(np.mean(x*y) - np.mean(x)*np.mean(y))
        variance = float(np.var(x)*np.var(y))
        pairs.append({"detectors": [i, j], "coordinates": [xyz, coords[j]],
                      "covariance": covariance,
                      "pearson": covariance / math.sqrt(variance) if variance > 0 else None})
    return pairs


def run_experiment(*, distances=(3, 5), rounds=(3, 5, 9), p=0.01,
                   shots=2000, rho=0.7, seed=12345):
    """Sweep placements/noise with one fixed homogeneous IID decoder per d,r.

    This has its own experimental schema; it does not reinterpret historical
    interop records as circuit-level correlation evidence.
    """
    if not imports.HAS_STIM or not imports.HAS_PYMATCHING:
        return {"status": "skipped", "reason": "stim and pymatching are required"}
    if not distances or any(d not in (3, 5) for d in distances):
        raise ValueError("distances must contain only 3 or 5")
    if not rounds or any(not isinstance(r, int) or isinstance(r, bool) or not 1 <= r <= 150 for r in rounds):
        raise ValueError("rounds must be integers in [1, 150]")
    if not isinstance(shots, int) or isinstance(shots, bool) or not 1 <= shots <= 20000:
        raise ValueError("shots must be an integer in [1, 20000]")
    if not math.isfinite(p) or not 0 <= p <= 0.1 or not math.isfinite(rho) or not 0 <= rho < 1:
        raise ValueError("require 0 <= p <= 0.1 and 0 <= rho < 1")
    if not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    config = dict(distances=sorted(set(distances)), rounds=sorted(set(rounds)),
                  p=p, shots=shots, rho=rho, seed=seed)
    # Conservative dense GF(2) propagation preflight, per sweep case.
    # Bounds include live uint8 arrays, decoder inputs and float statistics.
    for d in config["distances"]:
        for r in config["rounds"]:
            faults_count = r*d*d
            outputs_count = r*(d*d - 1) + 1
            estimated_bytes = 8*(faults_count*outputs_count + shots*(faults_count + outputs_count))
            multiply_adds = shots*faults_count*outputs_count
            if estimated_bytes > 256*1024*1024 or multiply_adds > 2_000_000_000:
                raise ValueError(
                    f"dense CPU resource budget exceeded for d={d}, rounds={r}, shots={shots}: "
                    f"estimated {estimated_bytes} bytes and {multiply_adds} multiply-adds; "
                    "reduce shots (150 rounds at d=5 supports up to 128 shots)"
                )
    records = []
    for d in config["distances"]:
        for r in config["rounds"]:
            ideal, data = build_memory_circuit(d, r, np.zeros((r, d*d)))
            xy = ideal.get_final_qubit_coordinates()
            expected = [[x, y] for y in range(1, 2*d, 2) for x in range(1, 2*d, 2)]
            if [xy[q] for q in data] != expected:
                raise RuntimeError("Stim data ordering changed")
            # At p=0 use a declared nonzero graph-building rate, never a noise floor.
            decoder_p = p if p else 0.01
            decoder_circuit, _ = build_memory_circuit(d, r, np.full((r, d*d), decoder_p))
            dem = decoder_circuit.detector_error_model(decompose_errors=True)
            matcher = imports.pymatching.Matching.from_detector_error_model(dem)
            response = fault_response(d, r)
            for homogeneous in (False, True):
                for placement in ("stripe", "diagonal"):
                    rates = physical_rates(d, p, placement, homogeneous)
                    for model, persistence in (("iid", 0.), ("temporal_markov", rho)):
                        # Common random numbers across placements; controls agree bitwise.
                        sub_seed = int(_hash([seed, d, r, model, homogeneous])[:16], 16)
                        faults = sample_faults(rates, r, shots, persistence, sub_seed)
                        outcomes = propagate_faults(faults, response)
                        detectors, observables = outcomes[:, :-1], outcomes[:, -1:]
                        predictions = matcher.decode_batch(detectors)
                        failures = int(np.count_nonzero(np.any(predictions != observables, axis=1)))
                        record = dict(distance=d, rounds=r, placement=placement,
                                      homogeneous_control=homogeneous, model=model, rho=persistence,
                                      seed=sub_seed, physical_x_rates=rates.tolist(), shots=shots,
                                      failures=failures, logical_error_rate=failures/shots,
                                      logical_error_wilson95=wilson_interval(failures, shots),
                                      empirical_fault_rate=float(faults.mean()),
                                      empirical_site_rates=faults.mean(axis=(0, 1)).tolist(),
                                      decoder_p=decoder_p, decoder_dem_sha256=_hash(str(dem).encode()),
                                      ideal_circuit_sha256=_hash(str(ideal).encode()),
                                      fault_response_sha256=_hash(response.tobytes()),
                                      fault_history_sha256=_hash(faults.tobytes()),
                                      outcomes_sha256=_hash(outcomes.tobytes()),
                                      detector_rates=detectors.mean(axis=0).tolist(),
                                      adjacent_detector_correlations=detector_correlations(detectors, ideal))
                        records.append(record)
    source = {str(Path(__file__).name): _hash(Path(__file__).read_bytes()),
              "runners.py": _hash(Path(__file__).with_name("runners.py").read_bytes())}
    result = dict(schema="qec.surface_memory_correlation.v1", status="completed",
                  model_version=MODEL_VERSION, config=config, source_sha256=source,
                  dependencies=dict(python=platform.python_version(), numpy=np.__version__,
                                    stim=imports.STIM_VERSION, pymatching=imports.PYMATCHING_VERSION),
                  decoder="PyMatching independent minimum-weight matching; fixed homogeneous IID X DEM",
                  records=records)
    result["receipt_sha256"] = _hash(result)
    return result


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shots", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=12345)
    args = parser.parse_args()
    result = run_experiment(shots=args.shots, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    if result["status"] == "skipped":
        parser.exit(2, result["reason"] + "\n")


if __name__ == "__main__":
    main()
