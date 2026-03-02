"""
Redundant Parity Check (RPC) augmentation — v3.8.0.

Deterministic, opt-in augmentation of the parity-check matrix by
appending redundant rows formed from XOR of existing row pairs.

The feasible set is unchanged: every valid codeword of the original
code satisfies the augmented system, and vice versa.

Mathematical definition
-----------------------
Given H in {0,1}^{m x n} and syndrome s in {0,1}^m, for selected
row index pairs (i, j) with i < j (lexicographic order):

    r     = H[i] XOR H[j]
    s_r   = s[i] XOR s[j]

Append r to H, s_r to s, subject to:
    - r is not the zero vector
    - r does not duplicate any existing row in H_aug
    - w_min <= weight(r) <= w_max
    - stop after max_rows rows accepted

Enumeration is lexicographic in (i, j) with i < j.
No randomness.  No hashing with nondeterministic ordering.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RPCConfig:
    """Configuration for RPC augmentation.

    Attributes
    ----------
    enabled : bool
        Whether RPC augmentation is active.  When False,
        ``build_rpc_augmented_system`` returns (H, s) unchanged.
    max_rows : int
        Maximum number of redundant rows to append.
    w_min : int
        Minimum Hamming weight of an accepted redundant row.
    w_max : int
        Maximum Hamming weight of an accepted redundant row.
    """
    enabled: bool = False
    max_rows: int = 10
    w_min: int = 2
    w_max: int = 50

    @classmethod
    def from_dict(cls, d: dict) -> "RPCConfig":
        """Construct from a plain dict (e.g. parsed from JSON config)."""
        return cls(
            enabled=d.get("enabled", False),
            max_rows=d.get("max_rows", 10),
            w_min=d.get("w_min", 2),
            w_max=d.get("w_max", 50),
        )


@dataclass(frozen=True)
class StructuralConfig:
    """Top-level structural geometry configuration (v3.8.0).

    Attributes
    ----------
    rpc : RPCConfig
        RPC augmentation sub-config.
    """
    rpc: RPCConfig = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.rpc is None:
            object.__setattr__(self, "rpc", RPCConfig())

    @classmethod
    def from_dict(cls, d: dict | None) -> "StructuralConfig":
        """Construct from a plain dict, or return defaults if None."""
        if d is None:
            return cls()
        rpc_dict = d.get("rpc")
        rpc = RPCConfig.from_dict(rpc_dict) if rpc_dict else RPCConfig()
        return cls(rpc=rpc)


def build_rpc_augmented_system(
    H: np.ndarray,
    s: np.ndarray,
    config: StructuralConfig | RPCConfig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build an RPC-augmented parity-check system.

    Parameters
    ----------
    H : np.ndarray
        Binary parity-check matrix, shape (m, n), dtype uint8.
    s : np.ndarray
        Binary syndrome vector, length m, dtype uint8.
    config : StructuralConfig | RPCConfig | None
        Structural configuration.  If None or RPC is disabled,
        returns (H, s) unchanged.

    Returns
    -------
    H_aug : np.ndarray
        Augmented parity-check matrix, shape (m + k, n).
    s_aug : np.ndarray
        Augmented syndrome vector, length m + k.

    Notes
    -----
    The original H and s are never modified in place.
    Enumeration of row pairs is lexicographic: (i, j) with i < j,
    iterating i from 0..m-1, j from i+1..m-1.
    """
    # Resolve config
    if config is None:
        return H, s
    if isinstance(config, StructuralConfig):
        rpc = config.rpc
    elif isinstance(config, RPCConfig):
        rpc = config
    else:
        return H, s

    if not rpc.enabled:
        return H, s

    H = np.asarray(H, dtype=np.uint8)
    s = np.asarray(s, dtype=np.uint8)
    m, n = H.shape

    max_rows = rpc.max_rows
    w_min = rpc.w_min
    w_max = rpc.w_max

    # Collect accepted redundant rows and syndrome bits.
    new_rows: list[np.ndarray] = []
    new_syns: list[int] = []

    # Build a set of existing row signatures for duplicate detection.
    # Use tuple representation for O(1) lookup.
    existing_rows: set[tuple[int, ...]] = set()
    for i in range(m):
        existing_rows.add(tuple(int(x) for x in H[i]))

    # Lexicographic enumeration: i < j
    for i in range(m):
        if len(new_rows) >= max_rows:
            break
        for j in range(i + 1, m):
            if len(new_rows) >= max_rows:
                break

            r = H[i] ^ H[j]
            w = int(np.sum(r))

            # Skip zero vector
            if w == 0:
                continue

            # Weight filter
            if w < w_min or w > w_max:
                continue

            # Duplicate check (against original + already-added rows)
            r_tuple = tuple(int(x) for x in r)
            if r_tuple in existing_rows:
                continue

            # Accept this row
            new_rows.append(r)
            new_syns.append(int(s[i]) ^ int(s[j]))
            existing_rows.add(r_tuple)

    if not new_rows:
        return H, s

    H_aug = np.vstack([H] + [r.reshape(1, -1) for r in new_rows])
    s_aug = np.concatenate([s, np.array(new_syns, dtype=np.uint8)])

    return H_aug, s_aug
