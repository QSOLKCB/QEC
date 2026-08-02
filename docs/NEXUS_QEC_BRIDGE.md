# QEC v170.2.1 — NEXUS Bridge and qBraid Replication Evidence

## Purpose

QEC v170.2.0 introduced a strict external adapter for the QSOLKCB/NEXUS native
VORTEX-N runner. QEC v170.2.1 preserves that execution contract and adds a
separate validation layer for the published NEXUS v4.0.1 qBraid replication
bundle.

The division of responsibility remains explicit:

- **NEXUS** computes deterministic geometry, traces, triality metadata,
  sonification controls, visual contracts and native receipt chains;
- **qBraid replication evidence** records an independently executed environment,
  build, correctness, thread and benchmark result set;
- **QEC** pins identities, verifies artifacts, preserves contradictions and
  creates canonical receipts without promoting implementation results into
  unsupported physical claims.

The adapter does **not** copy NEXUS geometry into QEC, replace a QEC decoder or
claim that VORTEX-N proves quantum behaviour.

## Version separation

QEC v170.2.1 distinguishes three version identities:

```text
QEC package release: 170.2.1
NEXUS execution receipt contract: 170.2.0
NEXUS qBraid replication receipt: 170.2.1
```

The execution contract is unchanged from v170.2.0, so existing execution
receipts and build attestations retain their original version identity. The new
v170.2.1 identity applies to the replication receipt introduced by this release.

## Pinned executable source profiles

### NEXUS v3.0.0 historical baseline

```text
commit: e078b135322dc12a2565b9c512fc4ba75193dea7
doi: 10.5281/zenodo.21745329
capabilities: verify, trace
```

### NEXUS v4.0.0 executable source profile

```text
commit: 1e93a509a28144d70a17fa76b330ae042db7beab
doi: 10.5281/zenodo.21748514
capabilities: verify, verify-parallel, trace, fibonacci, ternary, receipt
```

This remains the frozen source profile used by the execution bridge. Capability
gating prevents v4-only operations from being run under the v3 identity.

## Published NEXUS v4.0.1 result identity

The new publication is:

> Verschoor, D. L. & Slade, T. (2026). *VORTEX-N: A Deterministic Variable-N
> Centre-Transfer Receiver with Native Verification, Ternary Triality
> Execution, and Bounded Logical Sampling* (Version v4.0.1) [Computer
> software]. Zenodo. DOI: `10.5281/zenodo.21751929`.

The qBraid archive records package version `4.0.0` and tested commit
`1e93a509a28144d70a17fa76b330ae042db7beab`. QEC therefore records v4.0.1 as a
**replication-evidence publication** bound to the frozen v4.0.0 executable
source profile. It does not invent a source change that the archive does not
contain.

## Execution bridge proof layers

A direct QEC-to-NEXUS run retains four proof layers:

1. **Source identity** — pinned repository, version, commit and DOI.
2. **Build attestation** — declared source checkout, toolchain and observed
   executable SHA-256.
3. **Raw evidence** — byte-identical NEXUS CSV.
4. **QEC execution receipt** — canonical request, source, attestation, CSV and
   independently recomputed invariants.

The schemas remain:

```text
qec.nexus-build-attestation.v1
qec.nexus-execution-receipt.v1
```

## Replication evidence proof layers

The v4.0.1 qBraid validator adds a separate receipt path:

1. **Publication identity** — Zenodo v4.0.1 DOI.
2. **Archive identity** — external SHA-256 sidecar and observed archive bytes.
3. **Internal manifest** — every listed file independently rehashed.
4. **Raw result checks** — build statuses, source manifests, CSV equivalence,
   thread samples, deterministic outputs, visual markers and benchmark summary.
5. **Anomaly record** — contradictions are preserved rather than erased.
6. **Canonical replication receipt** — a SHA-256-bound QEC classification.

The replication schema is:

```text
qec.nexus-qbraid-replication-receipt.v1
```

## qBraid archive identity

```text
filename: nexus-v4-qbraid-rerun2-results-1e93a509.zip
sha256: 659e493a1b80b391db99b79dd6ee4e7a9b23c1821ff11eadbc3c5c36b10660d8
manifest sha256: 12bcacc9ddf0ef578528cd796f887c4235686c430bf9a1c53f09273c2424d0fc
manifested files verified: 130
archive member files: 132
```

The validator accepts only one safe top-level ZIP directory and rejects path
traversal, symbolic links, encryption, duplicate members, oversized members,
oversized archives and unexpected unmanifested files.

The two declared files outside the internal manifest are:

```text
SHA256SUMS.txt
00-provenance/manifest-verification.txt
```

## Verified qBraid implementation results

QEC independently verifies:

- source-before and source-after manifests are byte-identical;
- the source-manifest diff is empty;
- format, Clippy, release build and primary tests record exit status zero;
- Rust 1.82.0 installation, build and tests record exit status zero;
- scalar and parallel invariant outputs agree for 1, 2, 4 and 7 workers;
- the observed seven-worker process exits successfully;
- process sampling records eight threads: one main plus seven workers;
- repeated ternary outputs are byte-identical;
- repeated native receipts are byte-identical;
- changed receipt inputs produce changed receipt identities;
- a non-finite ternary frequency is rejected;
- D1 and D2 SVG contract markers are present.

## Environment-specific multicore result

```text
platform: qBraid
os: Ubuntu 24.04.4 LTS
kernel: 6.8.0-1059-azure
cpu: AMD EPYC 7763 64-Core Processor
online logical CPUs: 16
effective worker capacity: 7
observed process threads: 8
```

Five observations per mode produced these medians:

```text
scalar: 69.694063 ns/eval
7-worker parallel batch: 18.310215 ns/eval
speedup: 3.8062940822923164x
efficiency: 0.5437562974703309
```

QEC classifies multicore execution as supported by the combination of observed
threads and measured speedup. The performance figures remain specific to this
qBraid environment and workload.

## Preserved anomaly

The archive contains a non-empty file:

```text
01-build/primary/FAILED.txt
```

It states:

```text
required command failed: format-check
```

That marker contradicts the zero exit status recorded for format-check and the
bundle's `VALID / no anomalies` narrative. QEC does not reinterpret the valid
raw measurements as failures, but it also does not hide the contradictory file.

Canonical classification:

```text
usable_verified_evidence: true
blanket_valid_claim_accepted: false
whole_bundle_status: verified_with_declared_anomaly
```

The full analysis is recorded in:

```text
docs/replications/NEXUS_V4_0_1_QBRAID.md
docs/replications/nexus_v4_0_1_qbraid_receipt.json
```

Canonical receipt SHA-256:

```text
6137a128f73a950f0da12f54df10090005d23dfb5e771897e7af70fd55468dcd
```

## Validate the published replication receipt

```bash
qec-nexus-validate-replication receipt \
  --receipt docs/replications/nexus_v4_0_1_qbraid_receipt.json
```

## Rebuild the receipt from the archive

```bash
qec-nexus-validate-replication archive \
  --archive nexus-v4-qbraid-rerun2-results-1e93a509.zip \
  --expected-sha256 659e493a1b80b391db99b79dd6ee4e7a9b23c1821ff11eadbc3c5c36b10660d8 \
  --output nexus_v4_0_1_qbraid_receipt.json
```

## Direct NEXUS execution

Build attestation:

```bash
qec-nexus-bridge attest-build \
  --profile v4.0.0 \
  --binary external/NEXUS/target/release/nexus \
  --toolchain "$(rustc --version) / $(cargo --version)" \
  --output artifacts/nexus-build-attestation.json
```

Scalar verification:

```bash
qec-nexus-bridge verify \
  --profile v4.0.0 \
  --binary external/NEXUS/target/release/nexus \
  --attestation artifacts/nexus-build-attestation.json \
  --output artifacts/nexus-verify
```

Ternary sonification evidence:

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

Native receipt chain:

```bash
qec-nexus-bridge receipt \
  --profile v4.0.0 \
  --binary external/NEXUS/target/release/nexus \
  --attestation artifacts/nexus-build-attestation.json \
  --samples 10000 \
  --output artifacts/nexus-receipts
```

## Security boundary

The direct execution adapter:

- uses an explicit executable path without `shell=True`;
- executes a private copy of the exact binary bytes validated by the attestation;
- applies bounded executable, stdout and stderr sizes;
- applies a bounded timeout;
- requires UTF-8 CSV;
- rejects unexpected stderr from successful commands;
- sets deterministic locale and timezone variables;
- independently recomputes geometry and evidence invariants.

The replication validator:

- never extracts archive members to disk;
- validates paths and ZIP metadata before reading members;
- bounds member count and uncompressed size;
- rejects duplicate JSON keys;
- verifies the external archive hash and internal manifest;
- derives its receipt from raw files rather than trusting `RESULTS.json` alone.

## Scientific boundary

QEC v170.2.1 claims deterministic execution identity, archive identity,
listed-artifact integrity and declared implementation observations only.

It does **not** claim:

- that VORTEX-N is a quantum decoder;
- that triality is GF(3) geometry;
- that ternary labels establish qutrit physics;
- that Fibonacci ordering identifies a physical substrate;
- that sonification controls are physical observables;
- universal multicore performance;
- that receipts prove physical truth;
- quantum advantage, QEC break-even or a hardware threshold.
