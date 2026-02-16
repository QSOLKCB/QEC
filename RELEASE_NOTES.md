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