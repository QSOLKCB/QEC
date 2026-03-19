"""
v10.0.0 — Persistent Discovery Archive.

SQLite-backed storage for discovered LDPC/QLDPC codes with
content-addressable indexing, genealogy tracking, and population
seeding support.

Layer 3 — Archive.
Does not import or modify the decoder (Layer 1).
Fully deterministic: no randomness, no global state, no input mutation.
"""

from qec.archive.storage import DiscoveryArchive
from qec.archive.seeding import seed_population_from_archive

__all__ = [
    "DiscoveryArchive",
    "seed_population_from_archive",
]
