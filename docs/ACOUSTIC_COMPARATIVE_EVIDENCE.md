# v177.5 bounded comparative evidence and lab export

## Scope and status

This opt-in experiment compares **classical synthetic integer PCM**, measured-symbol
recovery, a caller-declared independent reference gate, and the unchanged exact
finite-Pauli decoders. It is implemented, unreleased evidence. It does not evolve a
quantum state, establish physical quantum protection, measure microphones, qualify
hardware, or prove the origin of a measurement. Hashes bind bytes and declared
fixtures, not physical truth. The shared `maximum_claim_level` is a ceiling, never
an assertion that every observation reached that level.

The battery has 19 named controls for each of two alphabets, 38 trials maximum.
One single-site Pauli fixture is used per alphabet. This is not an exhaustive
Pauli battery, a physical noise distribution, a threshold estimate, or evidence of
population-level improvement. No confidence interval or significance claim is
inferred from these selected deterministic controls.

## Methods and fairness

Every trial evaluates:

1. `unfiltered`: matched-filter receiver, first repetition only, no FEC
2. `cancellation_only`: declared reference subtraction, first repetition only, no FEC
3. `fec_only`: unfiltered samples, matched-filter majority repetition FEC
4. `combined`: declared reference subtraction and majority repetition FEC
5. `robust_median_period`: unfiltered samples, median-period projection and majority FEC
6. `direct_syndrome_reference`: privileged measured symbols directly into the unchanged exact decoder
7. `exact_clean_reference`: privileged clean PCM into the receiver and independent gate

The first five receive the same underlying original PCM, observed PCM, seed,
three transmitted repetitions, nominal harmonic frequency budget, transmitted
energy, and whole-frame sample-duration budget. The no-FEC methods still transmit
all repetitions and only inspect the first; unused repetitions are not an energy
or latency discount. The robust method is FEC-equipped and has no cancellation.

Nominal bandwidth denotes the highest declared harmonic; quantized PCM is not
strictly bandlimited. Sample energy and mean sample power are integer/rational
quantities. Whole-frame duration excludes processing and reference acquisition
and is not a hardware latency claim.

Cancellation receives an extra reference waveform generated independently from a
known additive noise generator/seed, with declared permission, delay, gain, and
polarity. This is a **privileged synthetic reference**, not an inferred clean
residual or a demonstration that an arbitrary second voice can cancel noise.
Nonadditive controls use an explicit zero reference. All verification gates
consume four extra independently declared measured symbols. The clean control
additionally receives the full original waveform. The direct reference bypasses
the channel entirely. The evidence records these extra resources and unmeasured
reference-acquisition cost; it never claims equal total resources or treats the
direct baseline as a functioning acoustic receiver.

## Controls and limits

All current noise-module families are exercised: broadband, colored, correlated,
synthetic speech-like interference, impulses, clipping, timing slip, missing
samples, phase drift, nonlinear distortion, echo and aliasing. Additional controls:

- Clean: cancellation has no benefit
- Aligned broadband reference: exact classical subtraction helps this fixture
- Wrong-phase reference: subtraction becomes addition and worsens PCM error
- Unrelated-seed reference: independent does not mean useful or aligned
- No tone: every channel receiver rejects; privileged references still work
- Coherent valid phase substitution across all repetitions: every channel method
  can accept the wrong symbols, and the independent reference veto prevents decoding
- First-repetition coherent substitution: no-FEC accepts wrong symbols, while
  majority repetition FEC recovers this bounded fixture
- A single amplified carrier period: exercises robust projection and the residual
  guard, which can legitimately prevent any final robust-estimator advantage

Synthetic control transforms operate on actual samples and are retained in the
source description. They are separate from the general noise-family contract.
Hidden fixture truth is available only to the fixture generator and the
post-decoding evaluator, never to the receiver or cancellation algorithm.

## Metrics and denominators

Each method retains predicted symbols, raw receiver acceptance, raw accepted-wrong
frames **before** the independent reference veto, independent verification,
decoder attempts/acceptance, and exact-model logical outcomes. A receiver accept
is not verified recovery. Verified recovery is not automatically decoder success.
Exact logical success requires a decoder-accepted correction and stabilizer
membership of the residual; zero syndrome alone is insufficient.

Counts are integers. Rates store numerator, denominator and `defined`. A zero
denominator is `0/0, defined=false`, never manufactured zero BER or SER.

- SER is wrong non-erased symbols / non-erased symbols
- Symbol erasure rate is erased symbols / all four transmitted symbols
- Error-or-erasure rate is (wrong + erased) / all four transmitted symbols
- BER is differing bits / decoded bits in explicit two-bit natural binary labels
- Qutrit BER uses `00,01,10` with `11` unused; this is a classical label metric,
  not a native binary physical channel or a quantum error probability
- Missing-length frames erase all four symbols; partial symbol decisions survive
  frame rejection for conditional metrics
- False-accept rates show both all applicable receiver frames and accepted frames
- The direct-syndrome baseline marks receiver metrics inapplicable and uses an
  undefined `0/0` receiver rate in both per-trial and aggregate summaries. Its
  recovery-verification metric is also inapplicable: it consumes an authoritative
  reference without recovering a channel observation, while decoder/model outcomes
  remain applicable

Waveform squared error uses aligned equal-length samples only and is undefined
for length mismatch. Neither favorable waveform error nor a transport receipt
promotes an artifact into verified acoustic/decoder evidence.

## Replay contract

`build_trial_source`, `build_battery_source`, `run_comparative_trial`, and
`run_comparative_battery` expose the finite deterministic input/output contract.
Each complete source contains channel configuration, actual original PCM,
measured reference, noise plans/seeds, actual independent reference PCM,
cancellation permission/configuration, control transform, and model-only error.

`validate_comparative_trial` requires external source and measured-reference
SHA-256 pins. `validate_comparative_battery` requires an external complete-battery
source pin (which binds every individual measured reference). These must be
retained outside the artifact being checked. The validator reconstructs the
catalogue source, recomputes every sample transformation, receiver observation,
reference gate, decoder decision, model outcome, resource count and metric, then
compares the entire canonical receipt. It does not merely rehash embedded claims.
Unknown fields, malformed types, rehashed altered outputs, reordered methods,
changed fixtures and fully consistent substitutions under stale pins fail closed.

Battery receipts retain complete inputs and output hashes for full receiver/gate
observations rather than duplicate their nested content for every method. Replay
recomputes those full observations. The separate lab demonstration includes the
full selected observations and gate receipts. Native transport receipts are not
embedded in each comparative trial.

## Ordinary WAV/JSON lab export

Run from a stable checkout with the repository's test-capable Python environment:

```sh
PYTHONPATH=src python scripts/generate_acoustic_trial_evidence.py --output /tmp/qec-acoustic-evidence
```

The destination must be empty. Default output includes:

- `battery.json`, `battery_source.json`, `battery_validation.json`
- Complete individual `trials/NN_qD_CASE.json` and `_source.json` bundles
- `demonstration.json`, its source and full replay validation
- `lab/original.wav`, `lab/observed.wav`, `lab/filtered.wav`: actual mono PCM16 WAV
- Six separate `lab/{original,observed,filtered,candidate,rejected,verified}.json` files
- Six actual complete stage JSON payloads transported through the unchanged native
  Packet model, with full native replay receipts in `transport/`
- A small separate Digital binding-record specimen, explicitly **not** a claim to
  transport the full candidate observation or PCM
- `environment.json` and a file-hash `manifest.json`

The demonstration is one explicitly selected qutrit broadband trial, independent
of which cases are selected for the battery. It demonstrates the original,
observed and filtered PCM, a real candidate, a real rejected raw observation,
and a fully reference-gated verified receipt. Every transported stage contains
its actual canonical JSON bytes, including actual PCM bytes for waveform roles.
The Digital specimen transports only its stated compact binding record.

`--case`, `--alphabet`, `--seed` and `--transport` allow bounded selections;
default transport is Packet. Selecting Digital for all six full stages is
bounded but materially larger because its native evidence records per-byte
frames. The entire export is capped at 32 MiB; an oversized full-stage Digital
selection fails explicitly. Keep those runs separate from normal comparative
battery exports.

Generation now claims a destination exclusively before writing artifacts; a
second generator targeting the same empty destination fails without writing.
On POSIX, output-parent traversal and descendant creation are anchored to
retained no-follow directory descriptors, and exclusive artifact creation uses
`O_EXCL`. On Windows, the writer creates descendants relative to retained
directory handles using native `NtCreateFile` and a delete-on-close exclusive
claim; missing native capabilities fail closed. The writer constructs manifest
file digests from bytes written through these handles, not mutable `rglob`
pathnames. It then reopens a guarded inventory to verify the complete written
snapshot. Source compilation is compared with loaded Python function code so
a file modified after import cannot be silently attested as the executed code.

POSIX inventories also retain each discovered regular file's size, inode,
device, and timestamps and require subsequent stat/open reads to match that
same identity. These are identity-guarded snapshots, **not** a global atomic
filesystem transaction: a sufficiently privileged concurrent process can
rename directories after they have been opened or mutate exports after return.
Run in a private, access-controlled working directory when adversarial
same-user processes are in scope; keep external digest pins independent.

Filesystem replay rejects special files, symlinks/reparse points, unlisted files
and detected entry replacement or mutation. Directory guards remain open while
bounded file snapshots are collected; downstream replay uses those retained
bytes. This is identity-guarded snapshotting, not a globally atomic filesystem
snapshot. POSIX requires descriptor-based directory enumeration, relative opens,
and no-follow/directory/nonblocking flags. Windows requires native
`GetFileInformationByHandleEx` support for 128-bit `FileIdInfo` and
`FileIdExtdDirectoryInfo`, plus `OpenFileById` for descendant directories and
payloads. A 64-bit identifier-open fallback is allowed only after
`ERROR_INVALID_PARAMETER` with zero high bits, and the complete 128-bit ID
and volume are still verified. SMB and unsupported primitives, filesystems or
metadata queries fail closed. There is no descendant pathname fallback for
directory enumeration, payload metadata or payload reads; the root manifest
uses the same guarded access.
Windows path-stat creation time is not compared with descriptor metadata-change
time. Shared-meaning cross-API identity fields and full same-API mutation checks
remain enforced.

This is SPECTRAL-compatible only in the ordinary-file sense: PCM16 WAV and JSON
can be imported or inspected by suitable external tools. No SPECTRAL repository,
UI, project format or hardware integration has been implemented or tested.

`validate_export` requires independently retained manifest, battery-source,
demonstration-source and measured-reference pins. It checks exact file bytes,
canonical JSON, original/observed/filtered WAV samples, role-to-source linkage,
all comparative results, and every native transport receipt. It also checks the
recorded code-file hashes against the executing checkout. Do not substitute
embedded pins when validating an untrusted export.

## Provenance and performance

The generator records the actual current Git commit, actual clean/dirty state,
actual SHA-256 and size of acoustic modules, the complete decoder and sonification source trees, native
Digital/Packet source trees, canonical serialization, generator script, and
dependency manifests. Installed dependency and Python/platform versions are
recorded. Git commit and source files are rechecked after generation; concurrent
source changes fail generation rather than silently misattribute the evidence.
A dirty working tree is reported honestly, never presented as a clean release.

Runtime nanoseconds and peak Python `tracemalloc` bytes are measured separately
for generation, comparative replay, file export and native demonstration. Peak
Python allocation is not process RSS. These host-dependent observations are
excluded from canonical scientific receipt identities, and are not performance
proof, physical latency or hardware QA. The manifest binds files and environment
records, but its host-dependent identity is distinct from the canonical battery
identity. Full-suite and CI status remain separate checks against an exact commit.

## Coverage and deferred work

| Area | Implemented boundary | Deferred / not claimed |
|---|---|---|
| Channel | Fixed integer multitone phase alphabet; channel contract supports 500/1000 Hz carrier and explicit amplitude | General AM/FM coding; continuous carrier estimation |
| Noise | Twelve declared synthetic sample-transform families and bounded negative controls | Measured microphones, physical noise distributions, field acoustics |
| Cancellation | Explicit independently supplied reference and fixed subtraction | Learned/adaptive filters, arbitrary voice separation, target fitting |
| Receiver | Matched-filter and robust median-period projections, residual checks, repetition FEC | General synchronization and hardware receiver calibration |
| QEC bridge | Independent fixture veto and unchanged exact finite-Pauli decoders | Quantum state evolution and physical quantum protection |
| Transport | Actual six-role Packet bytes plus separate small Digital specimen | Real network deployment or physical latency |
| Lab | Ordinary PCM16 WAV/JSON export and fully replayable sources | Actual external SPECTRAL UI/repository integration |

Focused tests cover determinism, all named controls, fairness costs, false accepts,
BER/SER/erasure denominators, malformed and fully rehashed attacks, bounded inputs,
actual WAV samples, full native byte replay, manifest binding and export tampering.
The repository-wide test gate is additionally required before publication.
