# CPU surface-memory correlation protocol (experimental v1)

This is the first, CPU-only Stim/PyMatching stage. It is a small-lab methods
experiment inspired by the correlated-QEC discussion around arXiv:2610.23840,
not a reproduction of IBM hardware, the paper's device noise, or a 104-site
NVIDIA simulation. The identifier is context, not imported calibration data.
No quantitative agreement with that work is claimed. An effect in this model
only supports a conclusion about this model.

## Physics and scope

- Stim `surface_code:rotated_memory_z`, distances 3 and 5; default round sweep
  3, 5, 9. Logical failure means PyMatching's predicted observable differs
  from the propagated logical-Z observable, not a residual physical error.
- Perfect Clifford gates, resets, ancillas and measurements. Physical data
  X faults occur immediately before the first ancilla H in each syndrome
  extraction round. This is a deliberately restricted phenomenological
  bit-flip channel, not a circuit-level depolarizing or coherent model.
- Site histories are independent across shots and data qubits. Within each
  history, `P(X_t=1 | X_(t-1)=x) = (1-rho)*p_i + rho*x`, with an initial
  Bernoulli(p_i) draw. Hence all time marginals are exactly p_i and lag-one
  binary correlation is rho. IID uses rho=0; the temporal lane defaults to .7.
  These are *fault events*, so adjacent X events compose by Pauli parity,
  not a persistent physical error state copied onto a data qubit each round.
- The existing `qec/simulation/correlated_noise_simulator.py` remains unchanged.
  Its temporal activation formula does not preserve the specified marginal
  rate, so reusing it here would confound correlation with noise strength.
  The stationary transition above is a dedicated adaptation of its Markov
  concept, coupled to explicit physical faults rather than synthetic labels.

## Placements and negative controls

Final-M target order is checked against Stim's row-major data coordinates.
Two synthetic spatial placements have identical rate multisets and mean p:

1. `stripe`: the d data sites in the first row are hot.
2. `diagonal`: the d data sites on the main diagonal are hot.

Hot sites have p_i=p*(d+1)/2; other sites have p_i=p/2. Moving the hot sites
therefore changes the physical error parameters at specific qubits. This is
spatial *inhomogeneity*, not simultaneous spatially correlated fault events.
It is not a measured calibration map or a hardware placement optimization.
The allowed mean p is [0,.1]. Homogeneous controls set every p_i=p: both
placement labels must then yield bit-for-bit identical fault/outcome hashes.

## Fixed decoder and exact propagation

For each distance/round pair a single PyMatching graph is built from an IID
homogeneous X-fault Stim detector error model at mean p. This same decoder
and weights are used for both placements, both temporal models and both
homogeneous controls. There is no correlation-aware decoder or retuning per
placement. At p=0 only graph construction uses .01; the generated physical
histories still contain zero faults. The graph's text SHA-256 is recorded.

Each single deterministic X_ERROR(1) circuit is simulated in Stim to obtain
its detector and logical-observable response. GF(2) superposition propagates
arbitrary sampled histories through this exact Clifford response matrix.
Tests compare this shortcut to direct deterministic Stim circuits for whole
histories at both distances. This avoids one circuit compilation per shot;
it is not a tensor-network approximation. The response approach does not
extend to coherent/non-Clifford noise without new validation.

## Measurements and uncertainty

Each record reports shots, logical failures, per-shot logical failure rate,
95% Wilson binomial interval, empirical site fault rates, detector rates and
per-pair covariance/Pearson correlation for identical stabilizers at adjacent
times. Boundary pairs are retained and identified by coordinates. Pearson
is null for zero variance. Detector-pair estimates are descriptive: no
multiple-testing correction or confidence intervals are claimed. Shots are
independent; rounds within a shot are not independent samples. Logical rates
are per complete memory experiment, not per-round rates or threshold fits.

Common random numbers are used across placements, making comparisons paired;
individual Wilson intervals do not constitute a paired significance test.
Zero observed failures is an upper-bound statement, not proof of zero error.
The checked-in tests/fixtures/surface_memory_correlation/cpu_v1_seed12345.json modest-shot receipt is a reproducibility smoke experiment,
not evidence for a threshold, optimal placement, scaling advantage or a
small lab matching the large simulation's conclusions.

## Run and replay

Use Python 3.12 (the CI lane) for the pinned environment (core QEC supports Python >=3.10). Install optional
packages with `python -m pip install -e '.[dev]' -r requirements/memory-py312plus.txt`.
From the repository root:

    python -m bench.interop.memory_correlation --shots 2000 --seed 12345 --output artifacts/surface_memory_correlation/local.json
    python -m pytest -q tests/test_surface_memory_correlation.py tests/test_interop_determinism.py tests/test_interop_import_hygiene.py tests/test_interop_schema_v312.py

The runner API is `bench.interop.runners.run_surface_memory_correlation`;
`distances`, `rounds`, `p`, `shots`, `rho`, and `seed` are keyword parameters.
The dense reference implementation supports rounds 1–150 and shots 1–20,000, subject to a per-case preflight
of 256 MiB conservative estimated working allocation and two billion dense
multiply-adds. For distance 5 and 150 rounds use at most 128 shots per case;
long-round results therefore have correspondingly broad statistical uncertainty.
Larger shot budgets need a separately validated chunked/sparse implementation. Existing runners and historical fixture schemas are unchanged. This lane has
a separate `qec.surface_memory_correlation.v1` schema. Absent optional
backends produce an explicit skipped API record; CLI exits 2, never a
success-looking empty benchmark. Physics tests skip without the backends;
the dedicated CI lane installs pinned backends so they actually run.

Receipts contain full configuration, model identity, exact dependency
versions, source-file hashes, ideal-circuit/decoder/response hashes, sampling
seeds, fault-history/outcome hashes and a canonical receipt hash. Sampling
uses NumPy's default PCG64 generator. Source/dependency changes may change
results and receipt identity; do not expect bitwise replay across versions.
Source hashes bind executable benchmark files; the receipt does not attest
an entire repository or imply a signed trusted execution environment.

## Next stage (separate work, not implemented)

Only after validating this baseline, design a separate NVIDIA simulation
lane for coherent/non-Clifford effects. Establish overlap cases against
Stim first, specify precision and approximation error budgets, then measure
wall time, memory and multi-GPU scaling on the actual 8–16 GPU budget.
Tensor-network truncation or reduced instances may make experiments cheaper,
but whether they preserve scientific conclusions must itself be tested.
No GPU deployment, rental, purchase or claim of full-scale reproducibility
is included in this PR.
