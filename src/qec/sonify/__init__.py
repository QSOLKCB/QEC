"""v167.0 deterministic symbolic sonification runtime skeleton."""

from .canonical import *
from .events import *
from .mapping import *

version = "v167.0"
SONIFY_RUNTIME_KIND = "SymbolicSonificationRuntimeSkeleton"


def symbolic_sonification_runtime_skeleton_hash() -> str:
    return canonical_sha256({
        "release": version,
        "runtime_kind": SONIFY_RUNTIME_KIND,
        "modules": [
            "src/qec/sonify/__init__.py",
            "src/qec/sonify/events.py",
            "src/qec/sonify/mapping.py",
            "src/qec/sonify/canonical.py",
        ],
        "schema_fixture": "tests/fixtures/sonify/event_schema_v167.json",
        "forbidden_scopes": [
            "no MIDI export",
            "no mapping packs",
            "no audio rendering",
            "no LLM calls",
            "no network calls",
            "no decoder imports",
            "no medical / biological / physics / cosmology / QEC advantage claims",
        ],
    })
