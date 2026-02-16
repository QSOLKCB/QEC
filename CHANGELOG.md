# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

## [0.3.0] - 2025-10-11
### Added
- **Qiskit backend support** - Added `SteaneCodeQiskit` class for Qiskit-based simulations
- Factory function `create_steane_code(backend)` for flexible backend selection
- Runtime backend switching in IRC bot via `!backend` command
- Environment variable `QEC_BACKEND` for configuring default backend
- 13 new comprehensive tests for both QuTiP and Qiskit backends
- `demo_backend_comparison()` function in `qec_steane.py` 
- `examples/backend_comparison_demo.py` demonstrating dual backend usage
- Dependencies: `qiskit>=0.45.0` and `qiskit-aer>=0.13.0`

### Changed
- Updated `SteaneCode` class documentation to clarify QuTiP implementation
- Enhanced `IntegratedQECBot.__init__()` to accept `backend` parameter
- Modified `cmd_threshold()` to display current backend
- Updated README.md, RELEASE_NOTES.md, and IRC_BOT_GUIDE.md with backend info
- Enhanced IMPLEMENTATION_SUMMARY.md with Qiskit details

### Fixed
- Import handling for Qiskit optional dependency (graceful fallback)
- Type hints compatibility when Qiskit not installed

## [0.2.0] - 2024-MM-DD
### Added
- AI-powered IRC bot with QEC simulations
- MIDI export capabilities
- LLM integration for conversational AI
- Comprehensive test suite

## [0.1.0] - YYYY-MM-DD
### Added
- Initial public release placeholder


[Unreleased]: https://github.com/multimodalas/fusion-qec-sim/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/multimodalas/fusion-qec-sim/releases/tag/v0.3.0
[0.2.0]: https://github.com/multimodalas/fusion-qec-sim/releases/tag/v0.2.0
[0.1.0]: https://github.com/multimodalas/fusion-qec-sim/releases/tag/v0.1.0