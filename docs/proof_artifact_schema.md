# QEC Proof Artifact Schema

All proof artifacts follow these rules:

- frozen / immutable dataclass where applicable
- explicit `to_dict()`
- canonical JSON bytes
- stable SHA-256 hash
- self-referential hash fields excluded
- recomputed hash must match stored hash
- all proof artifacts must satisfy: `recomputed_hash == stored_hash`
- invalid input raises `ValueError("INVALID_INPUT")`

## Current Examples

- `DistributedConvergenceReceipt`
- `ExtractionReceipt`

## v151.0 ExtractionReceipt Fields

- version
- raw_bytes_hash
- extraction_config_hash
- input_hash
- config_hash
- extraction_hash
- query_fields
- determinism_status
- stable_hash
