# Dependency profiles

Run installation commands from the repository root in an isolated environment.
The normal package and developer/science install remains:

```sh
python -m pip install -e '.[dev,science]'
```

`pyproject.toml` remains the package metadata and default dependency source.
These separate profiles preserve the previous requirement files byte-for-byte;
this directory move changes no version constraints or dependency policy.

| Profile | Purpose | Previous path |
| --- | --- | --- |
| `legacy-full.txt` | Historical broad dependency list; not the recommended default and not a lockfile. | `requirements.txt` |
| `dev.txt` | Lower-bound developer dependencies used by `scripts/test_quick.sh`. | `dev-requirements.txt` |
| `test.txt` | Lower-bound test/science dependencies, not exact pins. | `requirements-test.txt` |
| `memory-py312plus.txt` | Optional CPU surface-memory physics lane for Python 3.12+; the NVIDIA wrapper supports Python 3.12–3.14 with this profile. | `requirements-memory.txt` |
| `memory-py311.txt` | Explicit Python 3.11 compatibility profile for the optional physics lanes. | `requirements-memory-py311.txt` |
| `nvidia-cu13.txt` | Optional CUDA-13 validation packages, combined with the matching memory profile. | `requirements-nvidia.txt` |

The memory and NVIDIA profiles contain top-level pins, not complete transitive
lockfiles. Their names do not guarantee a compatible GPU, driver, operating
system, or reproducible full environment. Record the resolved packages with
execution evidence. The preserved comment in `memory-py311.txt` refers to
`requirements-memory.txt`, now `memory-py312plus.txt`.

For the CPU physics lane on Python 3.12+:

```sh
python -m pip install -e '.[dev]' -r requirements/memory-py312plus.txt
```

On Python 3.11, substitute `requirements/memory-py311.txt`. For optional GPU
execution, follow the platform and driver checks in the
[NVIDIA guide](../docs/nvidia_coherent_memory.md), then add
`-r requirements/nvidia-cu13.txt`. The
[surface-memory guide](../docs/surface_memory_correlation.md) describes the
separate CPU physics lane.

## Historical evidence

Do not rewrite historical receipts, manifests, archives, or source snapshots
to replace their original dependency paths. Newly generated NVIDIA bundles and
acoustic export manifests record the new paths.

The NVIDIA wrapper verifies bundled member bytes under their recorded names,
so bundles with either the old root paths or the new directory paths remain
verifiable. Replay still requires the exact recorded scientific source and all
existing clean-run, dependency, and CPU/GPU acceptance checks. Archived code is
never executed by verification.

Acoustic exports are deliberately tied to an exact source-file manifest,
including the exporter itself. Verify or replay a historical acoustic export
from its recorded source checkout with the original paths and external pins;
the current exporter does not silently remap or accept a changed source
manifest.
