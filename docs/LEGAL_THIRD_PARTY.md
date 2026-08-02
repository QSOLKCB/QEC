# Third-Party Tool Legal Matrix

This QEC-authored document is governed by the Mozilla Public License 2.0
(`MPL-2.0`) under the repository-wide licensing policy. Third-party tools,
dependencies and referenced upstream materials retain their own licences and
notices; their inclusion here does not relicense them.

This document defines the legal and licensing status of all third-party tools
referenced by the QEC benchmarking interop layer (`bench/interop/`).

## Categories

### Category A — Permissive Open-Source Tools (Bench-Only)

These tools are **optionally** used in `bench/interop/` for reference baseline
generation. They are **never** imported by core QEC modules.

| Tool       | Version Tested | SPDX License ID             | Upstream URL                           | Usage                                      |
|------------|---------------|-----------------------------|----------------------------------------|--------------------------------------------|
| Stim       | ≥1.12         | Apache-2.0                  | https://github.com/quantumlib/Stim     | Circuit/DEM generation for reference baselines |
| PyMatching | ≥2.0          | Apache-2.0 (verified 2026-02-25) | https://github.com/oscarhiggott/PyMatching | MWPM decoding for reference baselines      |

### Category B — Clean-Room Implementations (Literature-Based)

All QEC-native decoders are implemented from published literature. No
proprietary code, APIs, or reverse-engineered logic is used.

| Component             | Basis                                      | Notes                              |
|-----------------------|--------------------------------------------|------------------------------------|
| BP (min-sum/SP)       | Standard belief-propagation (textbook)     | Deterministic, seeded              |
| OSD family            | Panteleev & Kalachev (2021)                | Clean-room from paper              |
| Decimation            | Standard iterative decimation              | Literature-based                   |
| CSS construction      | Calderbank-Shor-Steane construction        | Textbook                           |

### Category C — Proprietary/Restricted Systems (NOT Implementable)

The following systems are **not** implemented, wrapped, or reverse-engineered
in this project. They are listed solely for documentation completeness.

| System              | Vendor      | Status                                  |
|---------------------|-------------|-----------------------------------------|
| Riverlane LCD       | Riverlane   | Proprietary; not implemented or cloned  |
| Google Matching     | Google      | Internal; not publicly available        |
| IBM Qiskit-QEC      | IBM         | Not used as a dependency                |

**Note:** Version tested reflects minimum validated version; newer compatible
versions may be used.

## Version Audit Procedure

1. Before each release, verify that upstream licenses for Category A tools
   have not changed by checking the `LICENSE` file in each upstream repository.
2. Confirm that Category A tools remain in `bench/interop/` only and are
   never imported by core modules under `src/`.
3. Run the import hygiene test (`test_interop_import_hygiene.py`) to verify
   no leakage.
4. Preserve every upstream copyright, attribution, patent and licence notice.

## Contributor Checklist

- [ ] New QEC-authored files are contributed under `MPL-2.0`.
- [ ] New third-party tool? Add it to the appropriate category above.
- [ ] Verify SPDX license ID matches upstream `LICENSE` file.
- [ ] Confirm the tool is used only in `bench/interop/` (Category A).
- [ ] Confirm no proprietary algorithms are reverse-engineered (Category C).
- [ ] Preserve all third-party notices without replacing them with MPL 2.0.
- [ ] Run `pytest tests/test_interop_import_hygiene.py` before merging.
