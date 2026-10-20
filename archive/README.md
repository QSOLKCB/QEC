# Preserved research and historical material

This directory keeps historical material available without placing it among the
runtime entry points. Nothing was deleted from the historical experiment corpus.
The root-cleanup manifest records each original path, destination, original Git
blob, byte count, SHA-256 and mode, and identifies live files whose references
were updated during relocation.

- `sonification/audio/`, `video/`, `images/`: original experimental media.
- `corpora/opaque-zips/`: original ZIP corpus, including non-game snapshots.
  These are opaque data, not installation packages. Do not extract or execute.
- `docs/`: superseded planning/state reports and the original CLAUDE constitution.
- `experiments/bp-v4.7.1/`: report and associated data/plots together.
- `reports/`: historical diagrams, figures, benchmark report and result snapshots.
- `legacy-tools/`: historical source/patches, retained unchanged for provenance.
- `v3_tests/`: pre-existing archived tests, unchanged by this cleanup.

Historical text is preserved byte-for-byte, including obsolete commands and
links. Consult `root-cleanup-manifest.json` for old-to-new paths rather than
treating an archived command as a current instruction. Archived scripts may
assume an old working directory and are not current supported launchers.

Current documentation starts at [docs/README.md](../docs/README.md) and contributor
instructions at [AGENTS.md](../AGENTS.md). The verified v3 CSV baseline deliberately
remains at `qec_data_prepared.csv` in the root: it is immutable provenance, not
unused output. Historical receipt paths and benchmark metadata are unchanged.
