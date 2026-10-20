# QEC contributor instructions

## Scope and workflow

These instructions apply repository-wide. Use a focused branch and a draft pull
request for reviewed changes; do not merge or release without authorization.
Keep commits small and single-purpose. Report checks actually run, failures,
optional skips and unavailable hardware separately. Never turn a skipped or CPU
check into a claim that GPU execution succeeded.

The original v150 constitution is preserved unchanged in
[archive/docs/CLAUDE.md](archive/docs/CLAUDE.md) as historical context, not the
current workflow. This file is the current contributor entry point. No automatic
CLAUDE.md-to-AGENTS.md fallback is assumed. The independent `.claude/` bootstrap
configuration remains in place.

## Repository map

- `src/qec/`: package implementation; `src/bench/`: research harnesses.
- `tests/`: regression checks and immutable historical fixtures.
- `scripts/`: runnable tools; `scripts/maintenance/`: repository administration.
- `docs/`: guides, architecture, roadmap and release records.
- `requirements/`: optional, test and legacy environment profiles.
- `archive/`: preserved experiments, media, opaque corpora and historical tools.
- `viz/` and `tui/`: browser labs and Rust operator interface.

Keep root limited to package/build/test/legal entry points. `PACKAGE_README.md`
is packaging input; `package.json` owns the existing JavaScript checks.
`qec_data_prepared.csv` remains at root because its bytes and historical lineage
are verified by benchmark code. Do not rewrite it as generated output.

## Architecture and protected boundaries

Read [architecture](docs/architecture/ARCHITECTURE.md),
[invariants](docs/architecture/INVARIANTS.md),
[the invariant registry](docs/architecture/INV.md), and the
[audit checklist](docs/contributing/AUDIT_CHECKLIST.md) for affected work.
Current workflow and dependency instructions here supersede conflicting historical
version-specific instructions in archived or migrated reference documents.

Layer order is decoder → channel → diagnostics → analysis → experiments →
benchmark harness → simulation. Lower layers must not import higher layers;
decoder must not import analysis. Avoid circular dependencies and supervisory
leakage into protected decoder behavior.

Do not alter legacy three-message/BP decoding, scheduling or adaptive logic as
incidental maintenance. The narrow owner-approved qutrit syndrome-table
construction exception is described in
[architecture](docs/architecture/ARCHITECTURE.md#narrow-qutrit-construction-exception-2026-10-15-pr-608).
It does not relax unrelated decoder, receipt or authorization boundaries.
Explain risk and obtain the required authorization before expanding that scope.

## Deterministic contracts

Same input must preserve ordering, canonical form, hashes, bytes and proof.
Identity-bearing tuples must use `canonical_hash_identity`; identity must be
sorted, unique and canonical, never guessed or reconstructed. Preserve explicit
ordering, stable serialization, deterministic reductions and immutable state.
Do not introduce implicit ordering, uncontrolled randomness or wall-clock
inputs into canonical results. Explicit seeded simulation and measured hardware
timings must retain their declared evidence boundaries.

Hash canonical JSON using sorted keys, compact separators and UTF-8; exclude
self-referential hash fields. Preserve exact recomputation and existing
`ValueError("INVALID_INPUT")` rejection contracts. Message identity is
`message_hash`; retain canonical payloads, duplicate rejection and ordering by
`(message_hash, sender_id, receiver_id)`. Convergence and derived flags must be
finite, deterministic and verifiable. Conflict classification remains symmetric,
canonical and limited to IDENTICAL, EQUIVALENT, DOMINATED and INCONSISTENT.

Receipts are immutable, canonical, hash-verifiable proof artifacts. Never change
historical fixtures, source metadata, hashes or recorded paths just to fit a new
layout. New results need new artifacts and accurate source identity. Preserve
v166.x authorization/promotion semantics and all later declared claim boundaries.

## Installation and validation

Core dependencies and supported extras are defined in `pyproject.toml`:

```sh
python -m pip install -e '.[dev]'
python -m pytest -q
npm test
npm run check
```

Optional environment profiles and their Python/CUDA boundaries are described in
[requirements/README.md](requirements/README.md). Preserve their pins during
layout work. They are top-level environment pins, not complete transitive locks.
Prefer stdlib and minimal dependencies; do not duplicate package dependency
metadata or install every optional backend by default.

Run targeted replay, invariant, boundary and hash-stability tests for affected
modules. Full `python -m pytest -q` is required for identity, hashing, canonical
ordering/deduplication, receipts/proof artifacts, protocol, convergence, conflict
classification or governance changes. For repository-wide moves, run the full
suite plus package/install checks and affected path/link/CLI/CI checks. Fix
regressions caused by the change; disclose pre-existing or environment failures
and do not claim completion with required checks unrun.

NVIDIA bundle integrity verification is distinct from successful GPU science.
See [the NVIDIA guide](docs/nvidia_coherent_memory.md). Historical bundles retain
their original paths and recorded source. Historical acoustic exports must be
verified with their recorded source checkout, not silently remapped.

## Archive preservation

Use [the archive index](archive/README.md) and
[move manifest](archive/root-cleanup-manifest.json) to locate moved material.
Preserve historical bytes, filenames, Git blob identities and executable modes.
Archives under `archive/corpora/opaque-zips/` are opaque data: do not extract,
execute, repack or install their contents during repository maintenance.
Historical tools are retained for provenance, not automatically supported entry
points. New generated outputs belong outside archived evidence and must not
overwrite it. Do not mistake sonification/analogy for physical quantum evidence.
