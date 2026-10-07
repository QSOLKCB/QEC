# Backend-owned command capabilities

`python -m qec.capabilities` (or `qec-capabilities`) exports `qec-capabilities/1` from the installed QEC package. It imports only static command declarations and scalar parser helpers, without importing scientific CLI/report/decoder modules. Consumers do not inspect argparse internals.

The first export includes:

| Action | Module | Effect | JSON result schema |
|---|---|---|---|
| `qec.ququart.benchmark` | `qec.benchmark.ququart_battery.cli` | writes artifacts | `qec.ququart-fer-battery.v170.1.1` |
| `qec.ququart.validate` | `qec.benchmark.ququart_battery.validate_cli` | may write a receipt | `qec.ququart-report-claim-validation.v1` |
| `qec.qutrit.benchmark` | `qec.benchmark.qutrit_battery.cli` | writes artifacts | `qec.qutrit-decoder-benchmark.v1` |

`qec.command_specs` owns `CommandSpec` and `ScalarOption` declarations for those commands. Each existing CLI's public `parser()` function builds its argparse parser from that same declaration. Exported defaults follow argparse string-default conversion; Path defaults and choices are strings in JSON. Required options, scalar choices and meaningful existing backend bounds are retained. A metadata change to a supported scalar option updates its parser and descriptor together.

The envelope contains `protocol`, `implementation_modules`, and `actions`. Each action contains an ID, fixed module entry point, display text, `fields`, an effect and an `output` contract. Field types are string, integer, number and boolean; each field specifies a long option flag. Optional values without defaults are omitted by clients. Boolean store values use literal `true`/`false`. Unsupported types produce an explicit error rather than a partial descriptor.

Path fields declare `path_role` (`read-file`, `write-file`, `read-directory`, `write-directory`) and `path_base: cwd`. Clients retain literal values and execute with the selected working directory. These are semantic hints, not path existence, permission or sandbox guarantees. Backend validation remains authoritative.

An output declares `format: json`, its schema and a view hint. Artifact manifests declare their output-directory parameter. Validation receipts declare `success_field: passed`; a zero-exit result without `passed: true` is not a successful validation. Validation conservatively advertises a possible write because its optional `--output` writes a receipt.

The exporter does not discover arbitrary modules, grant authority, read artifact files or execute a benchmark. `implementation_modules` allows a consumer to bind metadata to the installed provider/declaration/helper files. Consumers should also record interpreter/package identity and command source hashes. These are informative snapshots, not authenticated or complete environment fingerprints.

Protocol versions are exact compatibility boundaries. Unsupported input/output structures need a deliberate consumer update. Workbench P2 requires this exporter; its old pinned P1 environment can use an explicitly selected legacy argparse adapter.

Validation:

```sh
python -m pytest -q tests/test_command_capabilities.py tests/test_ququart_battery_report.py tests/test_qutrit_benchmark_report.py
python -m qec.capabilities
```

These tests exercise metadata/parsers and real existing report assembly. Small test workloads establish integration rather than a new scientific claim. The companion WORKBENCH acceptance additionally exercises direct CLI, CLI and Chromium forms against the installed package.
