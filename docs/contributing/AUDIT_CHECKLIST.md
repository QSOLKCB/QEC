# QSOL QEC — Pre-Merge Audit Checklist

This checklist defines the standard pre-merge audit procedure for the
QSOL QEC framework.  It protects four architectural invariants:

- **Determinism** — identical inputs produce byte-identical outputs.
- **Decoder integrity** — the BP decoder core is never modified by
  experiments, diagnostics, or control layers.
- **Experimental reproducibility** — all experiments are fully
  reproducible under fixed configuration.
- **Architectural modularity** — strict layering between decoder core,
  diagnostics, experiments, channels, and benchmarking.

Every pull request that touches `src/qec/` or `bench/` must pass this
checklist before merge.

---

## 1. Decoder Safety

- [ ] No unintended modifications inside `src/qec/decoder/`
- [ ] Decoder algorithms remain deterministic
- [ ] No monkey-patching or runtime modification of decoder functions
- [ ] Decoder API remains backward compatible
- [ ] Reference decoder behavior unchanged when experiments are disabled
- [ ] No new imports from decoder core in experiment or diagnostic modules
- [ ] Self-contained experimental BP implementations do not share state
      with the production decoder

## 2. Determinism

- [ ] No unseeded randomness (`random`, `numpy.random`, `hash()`)
- [ ] Deterministic ordering for all node lists (`sorted()` used)
- [ ] Dictionary iteration order not relied upon for output ordering
- [ ] Floating-point outputs follow 12-decimal rounding convention
- [ ] JSON artifacts are stable across repeated runs
- [ ] No global mutable state
- [ ] Sub-seed derivation uses SHA-256 or equivalent (if applicable)

## 3. Experimental Isolation

- [ ] Experiments live in `src/qec/experiments/`
- [ ] Baseline decoder output unchanged when experiments are enabled
- [ ] Experiments wrap the decoder; they do not modify it
- [ ] Features can be fully disabled via CLI flags
- [ ] Experiment modules do not import from `src.qec.decoder`
- [ ] Input arrays are copied before mutation

## 4. CLI Integration

- [ ] New flags are optional with `action="store_true"`
- [ ] Default behavior is unchanged when new flags are not set
- [ ] Dependencies between flags are explicit (implication chains documented)
- [ ] CLI help text is present and accurate
- [ ] Enabling a new flag does not disable or alter other experiments
- [ ] Flag wiring in `main()` matches `run_evaluation()` parameters

## 5. Numerical Stability

- [ ] Values clamped where required (e.g., damping to `[0.5, 0.9]`)
- [ ] Division-by-zero protected with early returns or guards
- [ ] Probability values remain in `[0.0, 1.0]`
- [ ] Normalization uses `value / max_value` (not `value / sum`)
- [ ] `np.arctanh` inputs clipped to avoid infinity
- [ ] `np.tanh` inputs clipped to avoid overflow
- [ ] No silent NaN or Inf propagation

## 6. Data Artifact Integrity

- [ ] All outputs are JSON-serializable (no numpy scalars leak)
- [ ] Output schema is documented in docstrings
- [ ] New metrics are stored under new keys — no overwriting of
      existing metrics
- [ ] Aggregation logic is deterministic (sum/count, not unordered
      reductions)
- [ ] `round(..., 12)` applied to all floating-point outputs
- [ ] Pass-through fields match upstream data structures

## 7. Code Simplicity

- [ ] No unnecessary external dependencies (stdlib + numpy only)
- [ ] Modular implementation with small, single-purpose functions
- [ ] Clear separation between experiments and core algorithms
- [ ] No refactoring of unrelated code
- [ ] No style-only changes to existing files
- [ ] Minimal diff — every changed line serves the feature

## 8. Testing Coverage

- [ ] Unit tests added for all new public functions
- [ ] Determinism tests included (run twice, compare JSON output)
- [ ] Edge cases tested (empty inputs, zero values, boundary conditions)
- [ ] JSON serialization roundtrip tested
- [ ] Pipeline integration test (full stack from diagnostics to output)
- [ ] Decoder safety test (verify no decoder imports in module source)
- [ ] All tests pass (`pytest` exit code 0)

## 9. Performance Impact

- [ ] Baseline decoding speed unchanged when new features are disabled
- [ ] Extra computations are behind opt-in flags only
- [ ] No heavy operations in decoder hot path
- [ ] No algorithmic complexity regressions in existing code paths

## 10. Release Documentation

- [ ] CHANGELOG.md updated with version entry
- [ ] CLI documentation reflects new flags
- [ ] Version number incremented according to versioning policy
- [ ] Release report prepared (if applicable)

## 11. Fork Synchronization Safety

`QSOLKCB/QEC` is the canonical development line. The upstream
`multimodalas/fusion-qec` fork is retained for historical lineage only.
Divergence from the upstream fork is expected and intentional; the
canonical `v137.x` history does not share a common ancestor with the
upstream fork.

- [ ] This PR does **not** merge any commits from `multimodalas/fusion-qec`
- [ ] This PR does **not** rebase `main` onto any upstream branch
- [ ] The GitHub "Sync fork" button was **not** used
- [ ] No automation in this PR performs upstream sync, fork merge, or
      upstream rebase against `QSOLKCB/QEC:main`
- [ ] Any ref repair on protected branches (`main`, release tags) uses
      `git push --force-with-lease=<ref>:<expected-sha>` with an explicit
      expected SHA — never plain `--force`
- [ ] Before any protected-ref rewrite, a recovery pointer is pushed to
      `origin` capturing the pre-rewrite tip (see
      `PROJECT_STATE.md` § Disaster Recovery for the canonical procedure)

### Disaster Recovery Reference

A prior accidental fork-sync rolled `origin/main` back from the canonical
`v137.x` tip (`fbdde2c`, PR #257 era) to a 307-commit historical fork
snapshot (`0c0f4e3`, "Merge pull request #25 from QSOLKCB/main") with no
common ancestor. Recovery was performed via a leased force-push:

```
git push origin fbdde2c:refs/heads/main \
    --force-with-lease=refs/heads/main:0c0f4e3
```

Safety pointers preserved on origin before the restore:

- `recovery/origin-main-rolledback-0c0f4e3` (pre-restore remote tip)
- `recovery/local-main-pre-restore-fbdde2c` (canonical tip snapshot)

Full restoration record lives in `PROJECT_STATE.md`.

---

## Audit Outcome

- [ ] All checklist items verified
- [ ] All tests passing
- [ ] No determinism violations
- [ ] No decoder core modifications
- [ ] Safe to merge
