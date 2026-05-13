# DOCUMENTATION

## What QEC is

QEC is a deterministic proof system for quantum error correction and
invariant-driven computation.

It transforms structured input into canonical JSON artifacts, binds them to
stable SHA-256 hashes, and verifies replay by recomputing those hashes.

## What QEC is not

QEC is not a probabilistic optimizer, open-ended agent runtime, or hidden-state
policy learner.

QEC does not make external systems deterministic.
It makes the proof boundary deterministic.

## Core law

```text
same input
→ same ordering
→ same canonical form
→ same hash
→ same bytes
→ same proof
```

## Canonical JSON and SHA-256 hashing

QEC canonicalizes artifacts before hashing:

- sorted keys
- compact JSON separators
- UTF-8 encoding
- self-hash exclusion for receipt hash recomputation

This guarantees replay-safe equality checks across environments.

## Receipt / artifact pattern

QEC artifacts are deterministic receipts that record:

- normalized input representation
- deterministic derivations
- explicit failure conditions
- canonical hash identity

Artifacts are immutable once emitted.

## Validator pattern

Validators recompute canonical hashes from artifact payloads and confirm that
stored hashes match recomputed values.

Mismatch indicates tampering, corruption, or invalid construction.

## Self-hash exclusion

Any field storing an artifact hash is excluded from the payload used to compute
that hash, preventing circular identity definitions.

## Replay-safe validation

Replay safety means independently rebuilding canonical payloads from the same
input yields byte-identical hashes and verdicts.

## Major implemented arcs

- **v151**: ingestion boundaries, RES/RAG grounding, deterministic replay chain.
- **v152**: reversible layers and compression-equivalence receipts.
- **v153**: lattice topology, router paths, and readout projections.
- **v154**: multi-scale invariance receipts and aggregation semantics.
- **v155**: entropy/decay signatures and replay-bound decay proofs.
- **v156**: GameWorld interaction boundaries and deterministic interaction
  reporting.
- **v157**: perturbation/stress contracts and replay verdict binding.
- **v158**: substrate constraint receipts.
- **v159**: bounded recursive proof-loop artifacts.
- **v160**: reality-loop composition receipts.
- **v161**: global validation/truth/replay receipts through v161.2.

## Installation

See [`INSTALL.md`](INSTALL.md) for full setup.

Quick command:

```bash
python -m pip install -e ".[dev,science]"
```

## Running tests

```bash
pytest -q
pytest -q -ra
```

## Running demos

SPHAERA proof demo:

```bash
python scripts/sphaera_proof_demo.py
```

Rust TUI control surface:

- install and usage: [`USAGE.md`](USAGE.md)

## Working with proof artifacts

Recommended workflow:

1. Build artifact from deterministic input.
2. Persist canonical JSON.
3. Record artifact hash.
4. Revalidate by recomputing hash from canonical payload.
5. Treat mismatch as invalid input/proof.

## Reading failure modes

Typical failure semantics include invalid input, hash mismatch, canonicalization
violations, and replay mismatch.

Treat all such failures as proof-boundary protection, not recoverable noise.

## Contributing without breaking determinism

- Preserve stable ordering.
- Avoid random/time-dependent behavior.
- Keep canonical serialization stable.
- Maintain float64 and explicit typing discipline.
- Avoid hidden dependencies in validators.
- Normalize external scientific dependencies through authoritative upstream
  sources with explicit deterministic review; QEC intentionally avoids
  unrestricted PyPI dependency resolution for optional dependency gating.
- Add tests for deterministic replay when changing behavior.

For historical context and release planning lineage, see [`ROADMAP.md`](ROADMAP.md).
