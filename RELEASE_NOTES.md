# RELEASE NOTES

## 43.0.0
Autonomous Discovery Scheduling - Added deterministic spectral landscape gap detection for under-explored regions.
- Added deterministic experiment target generation from discovered landscape gaps.
- Added opt-in autonomous experiment scheduler integrated into discovery engine.
- Added deterministic FIFO experiment queue for scheduling workflows.
- Added scheduling metrics utilities and scheduler artifact export fields.

## 42.0.0
Persistent Spectral Landscape Learning - Added persistent spectral landscape memory across experiments.
- Added deterministic clustering of explored spectral regions.
- Added JSON persistence helpers for landscape save/load.
- Added novelty scoring helper for exploration guidance.
- Added landscape coverage metric and discovery artifact export fields.

## 38.0.0
Non-Backtracking Eigenmode Mutation - Added non-backtracking matrix utilities.
- Added leading eigenmode detection for Tanner graphs.
- Added eigenmode-guided mutation scoring.
- Added optional mutation operator guidance using spectral instability directions.
- Added integration with spectral trajectory recording (`nb_eigenvalue` per step).

## v2.8.0
Deterministic Scheduling & State-Aware Enhancements Belief Propagation decoder enhancements for QLDPC codes.

### `improved_norm` / `improved_offset` modes

- Extended min-sum variants with dual scaling parameters `alpha1` and `alpha2`.
- Deterministic, invariant-preserving check-node update modifications.
- Fully backward-compatible with existing min-sum modes.

### `hybrid_residual` schedule

- Deterministic even/odd check-node partitioning.
- Within each layer, checks are ordered by descendin

## v0.3 – Qiskit Backend Integration
- **NEW: Dual backend support** - Choose between QuTiP or Qiskit for quantum simulations
- `SteaneCodeQiskit` class implementing [[7,1,3]] code with Qiskit framework
- Runtime backend switching via `!backend` command in IRC bot
- Factory function `create_steane_code(backend)` for flexible instantiation
- Environment variable `QEC_BACKEND` for default backend selection
- Full test coverage for both backends (13 new tests)
- Updated documentation and examples showing backend comparison
- Backward 

## v0.2 – Unified QEC Toolkit Global Demo Release
- Added robust, modular, and globally accessible QEC demo notebook: `notebooks/qec_demo_global.ipynb`
- Harmonized backend selection and simulation interfaces for Qiskit/QuTiP
- Enhanced LMIC/Colab/qBraid compatibility
- Improved documentation for reproducibility, user guidance, and global outreach
- Unified batch simulation/benchmarking for surface and color codes
- Static and real-time syndrome visualization (MIDI/cube)
- Export capability for DAW/producer workflows
- LMIC/global accessibility
