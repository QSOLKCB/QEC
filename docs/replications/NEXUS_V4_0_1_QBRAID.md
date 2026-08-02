# NEXUS v4.0.1 — qBraid Multicore Replication Receipt

## Publication identity

```text
publication version: 4.0.1
doi: 10.5281/zenodo.21751929
classification: replication evidence release
```

Citation:

> Verschoor, D. L. & Slade, T. (2026). *VORTEX-N: A Deterministic Variable-N
> Centre-Transfer Receiver with Native Verification, Ternary Triality
> Execution, and Bounded Logical Sampling* (Version v4.0.1) [Computer
> software]. Zenodo. DOI: `10.5281/zenodo.21751929`.

## Frozen source identity

The v4.0.1 publication presents an independent qBraid result set for the frozen
NEXUS v4 implementation. The archive itself records:

```text
package: nexus 4.0.0
source profile: v4.0.0
tested commit: 1e93a509a28144d70a17fa76b330ae042db7beab
control commit: a71e9f73aa000eb5ffb2138471c5d8f49a106ed7
```

QEC therefore keeps the executable source profile at `v4.0.0` and records
`v4.0.1` as the replication-publication identity. This avoids claiming a source
change that the evidence bundle does not contain.

## Archive identity

```text
filename: nexus-v4-qbraid-rerun2-results-1e93a509.zip
sha256: 659e493a1b80b391db99b79dd6ee4e7a9b23c1821ff11eadbc3c5c36b10660d8
internal manifest sha256: 12bcacc9ddf0ef578528cd796f887c4235686c430bf9a1c53f09273c2424d0fc
manifested files verified: 130
archive member files: 132
```

The two permitted files outside the internal manifest are:

```text
SHA256SUMS.txt
00-provenance/manifest-verification.txt
```

The external sidecar hash supplied with the archive matches the observed archive
SHA-256 exactly.

## Verified evidence

QEC independently validates the following archive properties:

- every listed internal SHA-256;
- one safe top-level ZIP directory with no traversal paths, encrypted members or
  symbolic links;
- identical source-before and source-after manifests;
- an empty source-manifest diff;
- zero exit status for format, Clippy, release build and primary tests;
- zero exit status for Rust 1.82.0 installation, build and tests;
- scalar and parallel invariant equivalence for 1, 2, 4 and 7 workers;
- a successful observed seven-worker execution;
- exactly eight observed process threads: one main thread and seven workers;
- deterministic repeated ternary output;
- deterministic native receipts with changed identities when inputs change;
- rejection of non-finite ternary sonification frequency;
- D1 and D2 visual-contract markers.

## Environment-specific performance result

The qBraid run reports five observations per mode on:

```text
platform: qBraid
os: Ubuntu 24.04.4 LTS
kernel: 6.8.0-1059-azure
cpu: AMD EPYC 7763 64-Core Processor
online logical CPUs: 16
effective worker capacity: 7
```

Measured median values:

```text
scalar: 69.694063 ns/eval
7-worker parallel batch: 18.310215 ns/eval
speedup: 3.8062940822923164x
efficiency: 0.5437562974703309
```

The performance values are environment-specific observations. They are not a
universal throughput guarantee.

## Declared anomaly

The archive contains this non-empty file:

```text
01-build/primary/FAILED.txt
```

Its content is:

```text
required command failed: format-check
```

However, the corresponding recorded status files all contain zero, including:

```text
format-check.exit-status: 0
clippy.exit-status: 0
build-release.exit-status: 0
tests.exit-status: 0
```

The raw format-check log is empty, while `RESULTS.json` and `REPORT.md` state
that all build checks passed and that no anomaly was detected.

QEC preserves both facts. It accepts the independently verified measurements and
implementation results, but rejects the unqualified whole-bundle claim of
`VALID / no anomalies`.

Canonical classification:

```text
usable_verified_evidence: true
blanket_valid_claim_accepted: false
whole_bundle_status: verified_with_declared_anomaly
```

This is not a claim that the source, build, correctness, thread or benchmark
results failed. It is a receipt-level statement that the archive contains a
contradictory stale marker and therefore cannot receive an anomaly-free blanket
classification.

## Canonical QEC receipt

```text
docs/replications/nexus_v4_0_1_qbraid_receipt.json
```

Receipt identity:

```text
6137a128f73a950f0da12f54df10090005d23dfb5e771897e7af70fd55468dcd
```

Validate the committed receipt:

```bash
qec-nexus-validate-replication receipt \
  --receipt docs/replications/nexus_v4_0_1_qbraid_receipt.json
```

Rebuild the receipt from the original archive:

```bash
qec-nexus-validate-replication archive \
  --archive nexus-v4-qbraid-rerun2-results-1e93a509.zip \
  --expected-sha256 659e493a1b80b391db99b79dd6ee4e7a9b23c1821ff11eadbc3c5c36b10660d8 \
  --output nexus_v4_0_1_qbraid_receipt.json
```

## Claim boundary

This receipt supports claims about archive identity, listed-artifact integrity,
source immutability, build/test exit statuses, deterministic implementation
outputs, observed process threads and environment-specific benchmark results.

It does not prove universal performance, physical truth, qutrit physics,
GF(3) geometry, quantum advantage, QEC break-even or a hardware threshold.
