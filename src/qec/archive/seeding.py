"""
v10.0.0 — Archive Population Seeding.

Utility for seeding discovery populations from an existing archive.

Layer 3 — Archive.
Does not import or modify the decoder (Layer 1).
Fully deterministic: no randomness, no global state, no input mutation.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from qec.archive.storage import DiscoveryArchive


def seed_population_from_archive(
    archive_path: str,
    size: int = 10,
    strategy: str = "elite",
) -> list[np.ndarray]:
    """Seed a discovery population from a persistent archive.

    Parameters
    ----------
    archive_path : str
        Path to the SQLite archive database.
    size : int
        Number of seed matrices to retrieve.
    strategy : str
        Selection strategy: "elite" or "diverse".

    Returns
    -------
    list[np.ndarray]
        List of parity-check matrices suitable for population
        initialization.
    """
    archive = DiscoveryArchive(db_path=archive_path)
    try:
        matrices = archive.seed_population(size=size, strategy=strategy)
    finally:
        archive.close()
    return matrices
