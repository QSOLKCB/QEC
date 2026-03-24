# Archived Legacy Tests (v3.0.2 Interop Era)

These tests originate from the pre-v98 architecture and are **not compatible**
with the current v98 system.

## Why archived

These tests depend on subsystems, import paths, or external packages that no
longer align with the current architecture:

- `bench.adapters.bp` / `bench.config` / `bench.schema` (v3 adapter layer)
- `qec_qldpc_codes` relative-import wiring (pre-package restructure)
- `cryptography` signing paths removed from default dependencies
- `matplotlib`-dependent analysis pipelines
- Assertion targets calibrated against v3.0.2 decoder outputs

## Status

- **Preserved for historical and reference purposes only.**
- **Not collected by pytest** (`archive/` is excluded via `testpaths` and
  `norecursedirs` in `pytest.ini`).
- No modifications have been made to the test logic itself.

## Reactivation

If any of these tests need to be brought back into the active suite, they must
first be updated to match the current v98 import structure and dependency set.
