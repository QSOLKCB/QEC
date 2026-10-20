# NVIDIA coherent-memory validation scaffold (experimental v1)

This stage adds a deliberately small, exact coherent/non-Clifford validation
problem with an optional NVIDIA cuStateVec backend. It complements, rather
than replaces, the [CPU Stim/PyMatching surface-memory
protocol](surface_memory_correlation.md). It does not reproduce a 104-site
simulation, a hardware calibration, or the scientific conclusions of a
large-device experiment. GPU execution is a separate acceptance stage;
passing CPU tests does not establish that NVIDIA execution works.

The implementation is fixed at three qubits. It is neither a scalable
surface-code simulator nor continuous error-corrected quantum memory.
There is no intermediate syndrome extraction, measurement, reset, recovery,
decoder feedback, or tensor-network truncation. One ideal final syndrome
measurement and conditional recovery defines the logical output channel.
The existing `qec_steane` capability or placeholder is not used as a
simulation oracle. Historical
fixtures and the earlier surface-memory schema remain unchanged.

## 1. Repetition-storage experiment

Use the repetition encoding `|0L> = |000>`, `|1L> = |111>` and test two
initial states: `|0L>` and `|+iL> = (|000> + i|111>) / sqrt(2)`.
A round applies the same `Rx(theta) = exp(-i theta X / 2)` to each of the three data qubits, with
`theta` in radians. Repeat that unitary round `r` times without observing
the register. At the end, jointly project the commuting `Z0 Z1` and
`Z1 Z2` stabilizers. Record the syndrome as `(b0 XOR b1) + 2*(b1 XOR b2)`.
Syndromes 0, 1, 2, 3 receive corrections I, X0, X2, X1 respectively.
Map the corrected `|000>` and `|111>` amplitudes back to a two-dimensional
logical state. Each syndrome retains an unnormalized 2-by-2 logical density
block; its trace is the syndrome probability. Sum the four blocks to trace
out the syndrome while preserving coherence *within* each block.

This is a quantum instrument, not a classical majority relabeling. For
`|0L>` its logical-one probability equals the familiar final-majority error,
but the off-diagonal density elements and the `|+iL>` input expose coherent
logical rotations that populations alone miss. This code corrects ideal
single-X faults; it does not protect against arbitrary noise or implement
a noisy repeated-recovery circuit.

For `c = cos(r*theta/2)`, `s = sin(r*theta/2)`, and logical Pauli X, an
independent analytic description has syndrome Kraus operators
`A0 = c^3 I + (-i*s)^3 X` and
`A1 = c^2*(-i*s) I + c*(-i*s)^2 X`. The three nonzero syndromes each
have operator A1. The corresponding block is `A rho A-dagger`. Comparing
all four full complex blocks checks both their probabilities and coherence.

The matched comparison is the *one-round* Pauli twirl: independently apply
X to each data qubit with probability `p = sin(theta / 2)^2` in each round,
then compose those channels for `r` rounds. Each round and site has an
independent Pauli draw. Applying X twice cancels, so the final parity
probability is not `r * p`. The analytic predictions are:

    p_coherent(r) = sin(r * theta / 2)^2
    p_twirled(r)  = (1 - cos(theta)^r) / 2
    majority_error(p) = 3 * p^2 - 2 * p^3

For `|0L>`, the two channels agree on one-round computational-basis
populations, but their logical density matrices already differ in coherent
off-diagonal terms. The `|+iL>` input makes the channel distinction visible
without assuming population agreement. Multiple rounds also generally
separate the `|0L>` final-majority probabilities. Twirling the *whole accumulated*
rotation instead would erase this distinction and is not the comparison
implemented here. Zero angle, one round, Clifford-compatible angles, and
multi-round non-Clifford angles provide distinct controls. Oscillations
and coherent revivals belong to this exact unitary model, not evidence
that physical error correction improves with storage time.

The primary observables are syndrome probabilities and the full recovered
logical density matrices, including a matched twirled logical density.
The final-majority formula is an independent `|0L>` control. There is no
Monte Carlo shot budget or binomial confidence interval in this coherent
lane. A difference from the twirl is
a model difference; numerical tolerances are not scientific error bars.

## 2. Phase-sensitive GHZ control

The second circuit starts in `|000>`, prepares GHZ with `H(0)`, `CX(0,1)`,
`CX(0,2)`, applies `Rz(theta)` to qubit 0 for `r` rounds, then reverses the
preparation with `CX(0,2)`, `CX(0,1)`, `H(0)`. A final qubit-0 outcome of
one has probability `sin(r * theta / 2)^2`.

Before decoding, GHZ computational-basis populations alone hide the
relative phase. The inverse preparation converts that phase into a
measurable probability, catching errors that a population-only control
could miss. A one-round Z-twirled channel has probability
`sin(theta / 2)^2`; its composed phase-parity probability is again
`(1 - cos(theta)^r) / 2`. This control validates phase propagation and
entangling-gate conventions. It is an interference probe, not a
fault-tolerant GHZ-memory claim or a second decoder benchmark.

## 3. Backends and acceptance boundaries

- `cpu` evolves the explicit state using NumPy dense matrices. The
  independent numerical reference uses `complex128`; analytic predictions
  are an additional oracle rather than a replacement for state evolution.
- `cuquantum` stores the state in CuPy device memory and applies gates with
  `cuquantum.bindings.custatevec.apply_matrix`. CuPy is the allocation and
  transfer layer, not a CPU substitute labelled as NVIDIA execution.
- The CPU reference is always `complex128`; a CPU `complex64` request is
  rejected. The GPU supports both `complex64` and `complex128`. Its absolute
  validation budgets are `2e-4` and `1e-10`, respectively, covering norm,
  global-phase-aligned state error against CPU, and full syndrome-block or
  GHZ probability error against the analytic control. These are small-case
  acceptance limits, not validated large-circuit error budgets.
  Do not require bitwise CPU/GPU agreement, or infer
  phase correctness merely from matching probabilities.
- Missing packages, no usable CUDA device, initialization failures, and
  failed numerical checks must remain explicit non-success outcomes. A
  requested NVIDIA run must never silently fall back to CPU. An unavailable
  GPU is not a GPU test pass.
- CPU physics and optional Stim cross-checks can run without CuPy,
  cuQuantum, or a GPU. Stim overlap checks concern supported Clifford or
  Pauli-channel cases; Stim is not an oracle for arbitrary Rx/Rz evolution.

The public NVIDIA bindings accept device state pointers, gate matrices,
target/control indices, precision, and an explicit extra-workspace pointer.
The implementation uses that API rather than a shell command that merely
reports an NVIDIA device. API references:
[cuStateVec apply_matrix](https://docs.nvidia.com/cuda/cuquantum/26.09.0/python/bindings/generated/cuquantum.bindings.custatevec.apply_matrix.html)
and [low-level bindings](https://docs.nvidia.com/cuda/cuquantum/26.09.0/python/bindings.html).

## 4. Bounded execution and receipts

Accepted configurations contain exactly three qubits, rounds from 1 to
150 inclusive, at most 16 requested round values, and a finite angle
within `[-pi, pi]`. There is no CLI
qubit-count or multi-GPU scaling switch. The state has eight complex
amplitudes: 64 bytes at `complex64`, or 128 bytes at `complex128`.
Those are state payload sizes, **not** total process or GPU memory.
CUDA contexts, loaded libraries, gate buffers, allocation pools and
cuStateVec workspaces can dominate this tiny problem.

Receipts bind the configuration, executable source hashes, evolved state
hashes, and canonical receipt hash. Numerical outputs, validation errors,
backend and dependency identity, runtime/driver observations, timing, state
payload, cuStateVec workspace and device-memory observations must be read
together. Unsupported observations are unavailable, not fabricated zeros.
Sampled free-memory differences are not a measured process peak, and a
workspace request is not a bound on every CUDA allocation.

Timing includes the scope stated by the receipt and must synchronize GPU
work before reporting completion. First-use setup, JIT compilation,
allocation and host/device transfers can dominate; one three-qubit run
cannot support a speedup or throughput claim. Record warm-up policy and
repeat measurements separately before making performance comparisons.

SHA-256 binds bytes and supports integrity checks. It does not authenticate
the machine, prove that a GPU executed the work, sign the result, or attest
the complete repository. State and receipt hashes can change across
precision, source, library and hardware versions. Runtime measurements
are observations, not deterministic physics inputs.

## 5. Run the CPU stage

From a checked-out repository root in an isolated Python environment:

    python -m pip install -e '.[dev]'
    python -m bench.interop.nvidia_coherent_memory --backend cpu --precision complex128 --rounds 1 3 9 --theta .125 --output artifacts/nvidia-coherent-memory/cpu-complex128.json
    python -m pytest -q tests/test_nvidia_coherent_memory.py tests/test_nvidia_evidence_bundle.py

The dedicated CI lane also installs `requirements-memory.txt`, so the
Stim overlap tests execute rather than skip. They compare complete
Clifford states up to global phase and sum exact weighted deterministic
Pauli histories for the twirled logical channel. The independent CPU
tests use dense stabilizer projectors rather than the implementation's
syndrome-index loop. A passing CPU-only run is expected to leave
`gpu_verified` false. CLI outcomes are 0 for completion, 2 for an
unavailable backend, and 1 for a failed backend or numerical check.

Keep the JSON receipts even when a requested backend is unavailable or a
check fails. Separate CPU results from later GPU evidence; never rename a
CPU receipt to imply NVIDIA execution. Use new output paths for distinct
runs and preserve previously published historical fixtures.

## 6. Optional NVIDIA environment

The proposed CUDA-13 recipe in `requirements-nvidia.txt` pins the top-level
releases `cuquantum-python-cu13==26.9.0` and
`cupy-cuda13x[ctk]==14.2.0`. Their release entries were verified on
2026-10-20: [NVIDIA's cuQuantum Python
release](https://pypi.org/project/cuquantum-python-cu13/26.9.0/) and
[CuPy's CUDA-13 release](https://pypi.org/project/cupy-cuda13x/14.2.0/).
The original source has user-supplied local GPU receipts; that evidence does
not validate a changed source revision or every compatible interpreter.
This is not a full transitive lockfile.
Save the resolved package inventory with actual execution evidence.

NVIDIA documents Linux, Python 3.11 or newer, and CUDA-12/13 families for
this release. The CUDA-13 driver floor listed by NVIDIA is 580.65.06 on
Linux. The original CPU dependency profile requires Python 3.12 or newer;
inspect the actual machine rather than assuming compatibility from its GPU
name. Native Windows execution is not established by this recipe.
Use consistently CUDA-13 binaries; do not mix CUDA-12 and CUDA-13 wheels.
See NVIDIA's [installation and requirements
guide](https://docs.nvidia.com/cuda/cuquantum/26.09.0/getting-started/index.html).

CuPy's `[ctk]` extra obtains CUDA component wheels, avoiding a requirement
for a system-wide toolkit in that packaging path. It does not install the
NVIDIA driver. Keep only one CuPy variant in the environment; do not add
`cupy` or `cupy-cuda12x` beside `cupy-cuda13x`. See the official [CuPy
installation guide](https://docs.cupy.dev/en/v14.2.0/install.html).

After inspecting the local OS, Python, driver and CUDA visibility, use a
separate environment for the optional packages:

    python -m pip install -e '.[dev]' -r requirements-memory.txt -r requirements-nvidia.txt
    python -m pip check
    python -m bench.interop.nvidia_coherent_memory --backend cuquantum --precision complex128 --rounds 1 3 9 --theta .125 --output artifacts/nvidia-coherent-memory/gpu-complex128.json
    python -m bench.interop.nvidia_coherent_memory --backend cuquantum --precision complex64 --rounds 1 3 9 --theta .125 --output artifacts/nvidia-coherent-memory/gpu-complex64.json

For Python 3.11, explicitly substitute `requirements-memory-py311.txt`:

    python -m pip install -e '.[dev]' -r requirements-memory-py311.txt -r requirements-nvidia.txt
    python -m pip check

This compatibility profile pins NumPy 2.4.6 and SciPy 1.17.1, preserving
Stim 1.15.0, PyMatching 2.3.1, and both original NVIDIA pins. The original
`requirements-memory.txt` pins remain unchanged for Python 3.12+.
Official release metadata lists Python >=3.11 for
[NumPy 2.4.6](https://pypi.org/project/numpy/2.4.6/) and
[SciPy 1.17.1](https://pypi.org/project/scipy/1.17.1/), whereas the original
NumPy 2.5.4 / SciPy 1.18.1 profile requires >=3.12. Available wheels and a
successful install are dependency compatibility, not proof of GPU execution.
The dedicated CPU workflow tests both profiles on their stated interpreters.

The local evidence wrapper records a resolved name/version package
inventory. The CPU requirements ensure that Stim overlap tests do not
silently skip in this acceptance environment. Resolve installation or compatibility failures
explicitly; do not downgrade the protocol, substitute CPU results, or
change drivers/system security settings just to obtain a green receipt.

## 7. Staged local acceptance and future scale

1. Review and merge the implementation PR, then verify required CI is green
   on the merged commit. CPU/Stim evidence is the first gate, not a claim
   of successful NVIDIA execution.
2. Only after that gate, run the bounded protocol through Codex on the
   intended RTX 5060 Ti 16 GB machine. Verify the actual device identity
   and memory rather than copying the intended hardware label into results.
3. Compare both precision modes, the repetition and GHZ controls, analytic
   predictions, and dense-reference errors. Preserve failures and dependency
   information alongside successful receipts. GPU acceptance remains
   pending until these checks really execute on that machine.
4. Export the source-bound evidence, verification output and integrity
   manifest as a separate archive for review. After the merged-and-green
   gate, the bounded wrapper runs CPU once, GPU in both precisions, and
   the dedicated CPU tests and mandatory real-GPU tests while preserving logs
   and commit/cleanliness data:

       python scripts/run_nvidia_pilot.py --output-dir /absolute/path/outside-checkout/new-pilot-run
       python scripts/run_nvidia_pilot.py --verify /absolute/path/outside-checkout/new-pilot-run/evidence.zip

   Use a fresh output directory outside the checkout. The wrapper produces
   `evidence.zip` and `evidence.zip.sha256`; it must not replace an older
   run. Inspect each command's exit status, skipped tests and GPU receipts,
   not just the presence of a ZIP. The verifier checks member hashes,
   receipt hashes and source binding. Integrity verification alone does
   not turn an unavailable backend, dirty worktree or failed test into
   successful GPU acceptance. A ZIP digest establishes the archive's
   integrity, not its scientific validity or authenticity.

   A later reproducibility check can use the recorded source revision and
   a fresh outside-checkout output directory:

       python scripts/run_nvidia_pilot.py --replay /absolute/path/to/prior/evidence.zip --output-dir /absolute/path/outside-checkout/replay-run

   Replay verifies the earlier bundle and runs the trusted local checkout,
   never code extracted from an archive. Read its fresh CPU/GPU results and
   phase-aligned state comparison, not just the earlier receipt hashes.
   It requires the exact recorded module source and previously successful
   CPU/GPU receipts, then compares all eight amplitudes within twice the
   relevant precision budget. The comparison is saved as `replay.json`
   beside the new archive, not inside it. In contrast, `--verify` only
   checks integrity and reported numeric/schema guards; it executes no
   simulation and cannot replace numerical replay.
   This command also executes GPU work and belongs after the same approval
   and merged-and-green gate as the original local pilot.

No GPU rental, hardware purchase, cloud deployment, or multi-GPU execution
is part of this stage. An 8-16 H200 study is a future *measured* protocol,
not a promised capacity estimate: specify an actual circuit family,
observable, device/interconnect topology, precision and approximation
budget; establish small-instance exact overlap; measure peak memory,
communication, synchronized wall time, repeated-run variance and scaling
efficiency. Any tensor-network truncation needs its own convergence tests.
Three-qubit correctness is necessary for this adapter, but does not
establish correctness, feasibility, or scientific agreement at 104 sites.


## 8. Distinct-gate preparation and historical evidence

The GPU adapter caches host matrix buffers by gate name and angle within a
single execution. Workspace queries are reused only for the same matrix and
control count on the same handle and precision. Every original gate remains
in its original order with its original target/control indices. This changes
preparation overhead only, not the physical model, recovery, numeric tolerance,
or synchronized gate timing scope. Buffers stay alive until final stream
synchronization. No global or cross-device cache is retained.

CPU tests inspect order, buffer identity, dtype, angle separation, control-count
query separation and per-execution ownership. `tests/test_nvidia_gpu_acceptance.py`
executes both precisions, zero/non-Clifford/Clifford/boundary angles and 1, 3,
and 150 rounds on actual hardware. It skips unavailable GPUs in ordinary CPU
CI; the evidence wrapper sets `QEC_REQUIRE_GPU=1`, making unavailability fail
instead of skip. CPU mocks are explicitly test doubles, not GPU evidence.

The historical `cpu_v1.json` is preserved byte-for-byte, bound to the inert
`source_v1.py` snapshot; tests hash but never import or execute that snapshot.
`cpu_v2.json` binds the current implementation. Historical bundles can still
be verified as data, but replay requires an exact source match with the trusted
local checkout. An older GPU receipt is never relabelled as acceptance of the
new cache implementation. The uploaded candidate was a separate implementation;
its local timing observations do not establish performance of this integration.
Fresh merged-source GPU acceptance and replay remain a separate local gate.

Supporting review history: uploaded archive `qec-review-20261020-165641.zip`
(SHA-256 `77dae10116e4c57192657ef11f7f457bda67bf95e9e67110f1fa625cacfc12e7`)
contains successful original-source and Python 3.11 compatibility receipts,
along with separate candidate results. Its original module matches merged
revision `ea906e3be26390d4303e9d6a56f094819b8f587d`. The review checked receipt,
configuration, state and source hashes; these are integrity checks, not
independent hardware attestation. The original archived run used Python
3.13.14 and the compatibility run used 3.11.15. Neither is a GPU receipt for
the current module. Private paths and host logs are not copied into fixtures.

Here `cpu_v2.json` means the second source-bound fixture, not a changed
numerical schema: both fixtures use `qec.nvidia_coherent_memory.v1` because
the scientific receipt contract is unchanged. Source hashes, rather than the
schema label alone, distinguish historical and current execution.

`python scripts/run_nvidia_pilot.py --preflight` uses only the standard library
to report interpreter support and module presence, including when NumPy is
missing. It selects installation guidance but does not install dependencies,
enforce version pins, load CUDA, or certify binary/driver compatibility. Follow
the explicit profile installation and `pip check`; execution failures remain
failures even when the modules were discoverable.
