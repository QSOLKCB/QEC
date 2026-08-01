# QEC v170.2.0 — NEXUS Geometry Adapter and Triality Receipt Bridge

## Purpose

QEC v170.2.0 adds a strict external adapter for the QSOLKCB/NEXUS native
VORTEX-N evidence runner. The adapter does **not** copy the NEXUS geometry into
QEC, replace any QEC decoder, or claim that NEXUS proves quantum or physical
behaviour.

The division of responsibility is explicit:

- **NEXUS** computes deterministic geometry, traces, triality metadata,
  sonification controls and native receipt chains;
- **QEC** pins source identities, constrains allowed capabilities, verifies the
  emitted CSV, binds executable bytes through a build attestation, and creates a
  canonical parent receipt.

## Presented v4.0.0 result set

QEC presents **NEXUS v4.0.0 as the current result set**. It is merged on the
NEXUS `main` branch, pinned to an exact commit and used as the authoritative
execution profile for the bridge. The pending Zenodo record is publication
metadata only; it does not make the merged v4 code or its results provisional.

The v4 result surface presented by QEC includes:

- preserved continuous gate → exact centre → mouth geometry;
- inbound `-1`, nexus `0` and outbound `+1` transfer trits;
- deterministic three-lane triality partitioning;
- scalar and ordered parallel invariant verification;
- uniform-floor and optional Fibonacci representative ordering;
- ternary sonification-event streams;
- per-lane, all-lane and chained SHA-256 receipts;
- D1 untwisted and D2 twisted two-mouth visual contracts.

The CI evidence bundle records the actual v4 CSV rows, executable digest,
build-attestation digest and QEC parent-receipt digest generated from the pinned
commit. Numeric results and hashes are presented from that run rather than
predeclared in documentation.

## Pinned source profiles

### Archived published v3 baseline

```text
version: 3.0.0
commit: e078b135322dc12a2565b9c512fc4ba75193dea7
doi: 10.5281/zenodo.21745329
capabilities: verify, trace
```

The DOI is the frozen published source identity for the historical NEXUS v3
baseline. It remains available for historical comparison and does not replace
the current v4 result set.

### Current v4.0.0 result profile

```text
version: 4.0.0
commit: 1e93a509a28144d70a17fa76b330ae042db7beab
doi status: pending
capabilities: verify, verify-parallel, trace, fibonacci, ternary, receipt
```

The v4 profile is pinned to the merged `main` commit. No Zenodo DOI is invented
or inferred. Its DOI field remains `null` with `doi_status: pending` until the
record is published and deliberately added in a later QEC release.

## Four proof layers

A QEC NEXUS run produces four distinct identity layers:

1. **Source identity** — pinned repository, version, commit and optional DOI.
2. **Build attestation** — hash-bound declaration linking the source profile to
   the observed executable SHA-256 and toolchain description.
3. **Raw evidence** — byte-identical NEXUS CSV output.
4. **QEC execution receipt** — canonical parent receipt containing the request,
   source identity, build-attestation hash, output hash and independently
   recomputed invariants.

A build attestation does not claim cross-environment reproducible binary bytes.
It records the bytes actually executed and the source checkout declared by the
controlled build step.

## Capability gating

QEC will not silently use a v4 command under the v3 profile. For example,
`ternary`, `receipt`, `fibonacci` and `verify-parallel` fail before subprocess
execution when the source profile is `v3.0.0`.

This prevents a receipt from combining the historical v3 DOI identity with the
current v4 capability surface.

## Build attestation

After building the pinned NEXUS checkout:

```bash
qec-nexus-bridge attest-build \
  --profile v4.0.0 \
  --binary external/NEXUS/target/release/nexus \
  --toolchain "rustc 1.82 / cargo --release --locked" \
  --output artifacts/nexus-build-attestation.json
```

The resulting `qec.nexus-build-attestation.v1` object contains:

- the complete pinned source profile;
- the executable name and full SHA-256;
- the declared build toolchain;
- an explicit `reproducible_binary_claim: false` boundary;
- its own canonical SHA-256 identity.

Every execution command requires this attestation and independently rehashes the
binary before launching NEXUS.

## Verification run

```bash
qec-nexus-bridge verify \
  --profile v4.0.0 \
  --binary external/NEXUS/target/release/nexus \
  --attestation artifacts/nexus-build-attestation.json \
  --output artifacts/nexus-verify
```

QEC independently checks:

- requested logical, rendered and particle cardinalities match the output;
- the centre error is exactly zero;
- the antipodal error is exactly zero;
- orientation changes sign exactly at the centre;
- the floor-sampling gap residual is bounded by one;
- all numerical fields are finite and canonically parseable.

## Ternary sonification run

```bash
qec-nexus-bridge ternary \
  --profile v4.0.0 \
  --binary external/NEXUS/target/release/nexus \
  --attestation artifacts/nexus-build-attestation.json \
  --channel 17 \
  --steps 32 \
  --base-frequency-hz 432 \
  --output artifacts/nexus-ternary
```

QEC recomputes and validates:

- contiguous ordered steps;
- progress derived from `step / steps`;
- inbound / nexus / outbound classification;
- triality lane derived from `rendered_index mod 3`;
- exact-origin centre rows;
- positive frequency;
- amplitude in `[0, 1]`;
- pan in `[-1, 1]`;
- full-amplitude, centred nexus event.

The emitted controls are sonification mappings. They are not conservation laws,
physical observables or evidence for qutrit physics.

## Fibonacci representative ordering

The `fibonacci` operation verifies the optional NEXUS phi ordering separately
from the canonical uniform floor sampler. QEC checks that the reported
`fib_fraction` agrees with the declared logical index and logical cardinality.

Fibonacci ordering is a representative-selection policy, not a replacement for
the baseline model and not evidence of a golden-ratio physical substrate.

## Native triality receipts

```bash
qec-nexus-bridge receipt \
  --profile v4.0.0 \
  --binary external/NEXUS/target/release/nexus \
  --attestation artifacts/nexus-build-attestation.json \
  --samples 10000 \
  --output artifacts/nexus-receipts
```

The adapter requires the exact ordered rows:

```text
lane-0
lane-1
lane-2
all-lanes
chain
```

Every digest must be a complete lowercase SHA-256. Truncated hashes and
reordered receipt rows fail closed.

## Output bundle

Each execution directory contains:

```text
nexus_output.csv
request.json
source_identity.json
build_attestation.json
execution_receipt.json
```

The parent receipt schema is:

```text
qec.nexus-execution-receipt.v1
```

It binds:

- QEC version `170.2.0`;
- pinned NEXUS source identity;
- canonical invocation contract;
- observed binary SHA-256;
- build-attestation SHA-256;
- raw CSV SHA-256;
- row count;
- independently verified invariants;
- explicit scientific and decoder boundaries.

Validate a stored parent receipt with:

```bash
qec-nexus-bridge validate \
  --receipt artifacts/nexus-verify/execution_receipt.json
```

## Security and execution boundary

The adapter:

- invokes an explicit executable path without `shell=True`;
- uses a bounded timeout;
- requires UTF-8 CSV on stdout;
- rejects successful executions that emit unexpected stderr;
- sets deterministic locale and timezone variables;
- never mutates `src/qec/decoder/`;
- never treats external output as trusted merely because NEXUS emitted it.

## Scientific boundary

QEC v170.2.0 claims only deterministic adapter execution, byte identity,
source/profile binding and declared invariant verification.

It does **not** claim:

- that VORTEX-N is a quantum decoder;
- that triality is GF(3) geometry;
- that ternary labels establish qutrit physics;
- that a visual or sonified centre is a physical transition;
- that receipts prove physical truth;
- quantum advantage, QEC break-even or a hardware threshold.
