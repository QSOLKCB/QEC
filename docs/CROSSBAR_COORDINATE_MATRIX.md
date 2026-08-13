<!-- SPDX-License-Identifier: MPL-2.0 -->
# QEC v172.0 — Crossbar Matrix Core

## Scope

v172.0 introduces the first Crossbar switching primitive in QEC: an immutable,
bounded horizontal/vertical coordinate matrix with canonical link-state records
and a canonical identity for every intersection.

This release intentionally stops before common control.

```text
declared matrix identity
→ canonical horizontal links
→ canonical vertical links
→ deterministic row-major coordinate closure
→ per-intersection identity
→ matrix manifest
→ replay validation
```

The v172.x architectural rule remains:

```text
canonical request
→ marker input register
→ deterministic path search
→ coordinate closure plan
→ path continuity verification
→ connection receipt
→ marker release
```

Only the matrix substrate required by that later pipeline is implemented here.

## Canonical artifacts

`qec-crossbar matrix` emits:

```text
crossbar_matrix_manifest.json
crossbar_matrix_validation.json
```

The canonical matrix schema is:

```text
qec.crossbar-matrix-manifest.v1
```

The primary v172.0 identity is exposed as:

```text
crossbar_matrix_receipt_hash
```

and is the SHA-256 identity of the validated canonical matrix manifest.

Each horizontal/vertical coordinate also has a derived identity under:

```text
qec.crossbar-intersection-id.v1
```

The intersection identity binds the matrix id, both link ids and both canonical
ordinals. Moving a coordinate or changing the matrix identity therefore changes
the coordinate identity.

## Canonical ordering and bounds

Horizontal and vertical links are separate immutable records.

For each axis:

- ordinals begin at zero;
- ordinals are contiguous;
- tuple order must exactly match ordinal order;
- link ids are unique;
- link state must belong to the exact declared vocabulary;
- the matrix must contain at least one link on each axis.

Crossbar link ids are globally unique across both axes.

The complete intersection inventory is generated in deterministic row-major
order:

```text
H0×V0, H0×V1, ..., H1×V0, H1×V1, ...
```

No dictionary or set traversal participates in canonical ordering.

The v172.0 implementation bounds each axis to at most 4096 links and the
complete matrix to at most 65536 intersections.

## Link-state vocabulary

The exact initial-state vocabulary is:

```text
busy
idle
quarantined
unavailable
```

These are declared initial-state facts in the immutable matrix manifest.
Reservation, release and contention transitions are not implemented in v172.0;
they belong to later v172.x milestones.

Changing any declared link state changes the matrix hash.

## Replay-not-trust validation

`validate_matrix_manifest()`:

1. verifies the outer canonical SHA-256;
2. verifies the exact schema and contract version;
3. reconstructs horizontal and vertical link records;
4. enforces contiguous canonical ordinals and unique identities;
5. reconstructs every expected coordinate;
6. recomputes every intersection identity;
7. requires the supplied intersection list to match complete row-major closure;
8. verifies the machine-readable claim boundary;
9. emits `qec.crossbar-matrix-validation.v1`.

A partial or altered matrix cannot become valid merely by recomputing its outer
SHA-256.

## CLI

Generate a deterministic matrix:

```bash
qec-crossbar matrix \
  --matrix-id crossbar-demo \
  --horizontal-count 4 \
  --vertical-count 4 \
  --link-state H001=busy \
  --output-dir artifacts/crossbar
```

Validate the manifest:

```bash
qec-crossbar validate \
  --manifest artifacts/crossbar/crossbar_matrix_manifest.json
```

## v172.0 claim boundary

v172.0 claims deterministic classical software identity for an immutable
Crossbar-style coordinate matrix and its declared initial link states.

It does **not** yet claim or implement:

- marker/common-control authority;
- route search;
- look-ahead path selection;
- reservation or release;
- contention resolution;
- connection commit;
- end-to-end continuity proof;
- Strowger/Panel/Crossbar outcome equivalence;
- physical Crossbar-switch fidelity;
- carrier-grade reliability;
- quantum hardware behaviour;
- decoder correctness;
- quantum advantage.

The marker/common-control contract begins at v172.1.
