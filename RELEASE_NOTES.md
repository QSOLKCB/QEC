## [42.0.0] — Persistent Spectral Landscape Learning

- Added persistent spectral landscape memory across experiments.
- Added deterministic clustering of explored spectral regions.
- Added JSON persistence helpers for landscape save/load.
- Added novelty scoring helper for exploration guidance.
- Added landscape coverage metric and discovery artifact export fields.

## [38.0.0] — Non-Backtracking Eigenmode Mutation

- Added non-backtracking matrix utilities.
- Added leading eigenmode detection for Tanner graphs.
- Added eigenmode-guided mutation scoring.
- Added optional mutation operator guidance using spectral instability directions.
- Added integration with spectral trajectory recording (`nb_eigenvalue` per step).

## v2.8.0 — Deterministic Scheduling & State-Aware Enhancements

Belief Propagation decoder enhancements for QLDPC codes.

### `improved_norm` / `improved_offset` modes

- Extended min-sum variants with dual scaling parameters `alpha1` and `alpha2`.
- Deterministic, invariant-preserving check-node update modifications.
- Fully backward-compatible with existing min-sum modes.

### `hybrid_residual` schedule

- Deterministic even/odd check-node partitioning.
- Within each layer, checks are ordered by descending residual.
- Optional `hybrid_residual_threshold` prioritizes high-residual checks within each layer.
- No randomness; stable tie-breaking by ascending check index.

### Deterministic ensemble decoding (`ensemble_k`)

- Runs K independent BP passes with deterministic, zero-mean alternating LLR perturbations.
- Member 0 uses exact baseline LLR.
- Selection criteria:
  - Converged solutions preferred.
  - Lowest syndrome weight.
  - Deterministic member index tie-break.
- No RNG usage; fully reproducible.

### State-aware residual weighting (`state_aware_residual`)

- Residual ordering can be modulated by per-check state weights:
  - `weight = s_by_state[label] * |cos(phi_by_state[label])|`
- Applied multiplicatively to raw residuals.
- Strict validation of state labels (non-negative, in-range, length `m`).
- Disabled by default — baseline behavior unchanged when off.

### Test Boundary Stabilization

- Added `pytest.ini` to scope test discovery to `tests/`.
- Full regression: 339 passed, 7 skipped, 0 failed.
- Determinism verified across repeated runs.

## v0.3 – Qiskit Backend Integration

- **NEW: Dual backend support** - Choose between QuTiP or Qiskit for quantum simulations
- `SteaneCodeQiskit` class implementing [[7,1,3]] code with Qiskit framework
- Runtime backend switching via `!backend` command in IRC bot
- Factory function `create_steane_code(backend)` for flexible instantiation
- Environment variable `QEC_BACKEND` for default backend selection
- Full test coverage for both backends (13 new tests)
- Updated documentation and examples showing backend comparison
- Backward compatible - QuTiP remains the default backend

## v0.2 – Unified QEC Toolkit Global Demo Release

- Added robust, modular, and globally accessible QEC demo notebook: `notebooks/qec_demo_global.ipynb`
- Harmonized backend selection and simulation interfaces for Qiskit/QuTiP
- Enhanced LMIC/Colab/qBraid compatibility
- Improved documentation for reproducibility, user guidance, and global outreach
- Unified batch simulation/benchmarking for surface and color codes
- Static and real-time syndrome visualization (MIDI/cube)
- Export capability for DAW/producer workflows
- LMIC/global accessibility and ethics statement

All files and features ready for research, education, and music/producer integration.