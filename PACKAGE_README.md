# QEC 172.0.0 — Crossbar Matrix Core Development Package

This package description applies to the **172.0.0 development/package
candidate** that begins the v172.x Crossbar coordinate-switching phase.

The authoritative published stable release is **v171.5.0**. Package-version
metadata and published-tag status remain intentionally separate so this
unreleased v172.0 candidate is not presented as an already-published stable tag.

## v172.0 Crossbar Matrix Core

The candidate introduces:

- immutable horizontal and vertical link records;
- contiguous canonical axis ordinals;
- exact initial link-state vocabulary;
- deterministic row-major matrix closure;
- canonical identity for every horizontal/vertical intersection;
- `qec.crossbar-matrix-manifest.v1`;
- replay-not-trust matrix validation;
- the `qec-crossbar` CLI;
- dedicated Crossbar CI and regression coverage.

Primary identity:

```text
crossbar_matrix_receipt_hash
```

v172.0 deliberately contains no marker/common-control authority, route search,
reservation, connection commit, continuity proof or cross-era equivalence.
Those remain assigned to later v172.x releases.

This software makes deterministic classical software-model and artifact-identity
claims only. It does not establish physical Crossbar fidelity, carrier-grade
reliability, decoder correctness, quantum hardware behavior, physical truth or
quantum advantage.
