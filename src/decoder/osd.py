"""
Order-0 Ordered Statistics Decoding (OSD-0).

Post-processing step for belief-propagation decoders.  When BP fails
to converge, OSD-0 uses the soft reliability information (LLRs) to
select an information set via Gaussian elimination and solve for the
most likely error pattern.
"""

from __future__ import annotations

import numpy as np

from .gf2 import gf2_row_echelon


def osd0(H, llr, hard_decision, syndrome_vec=None):
    """Order-0 Ordered Statistics Decoding.

    Steps:
        1. Sort columns of *H* by ``|llr|`` ascending (least reliable first)
           using a stable sort.
        2. Row-reduce the permuted *H* over GF(2).  Pivots land on the
           least-reliable independent columns.
        3. The non-pivot (most-reliable) columns form the information set
           whose bits are kept from *hard_decision*.
        4. Back-substitute to solve for the pivot (redundant) bits.
        5. Un-permute and return the corrected word.

    The *never-degrade* guarantee ensures that if the returned word does
    not satisfy the syndrome, the original *hard_decision* is returned
    unchanged.

    Args:
        H: Binary parity-check matrix, shape (m, n).
        llr: Per-variable log-likelihood ratios, length n.
        hard_decision: Binary hard-decision vector from BP, length n.
        syndrome_vec: Target syndrome, length m.  Defaults to all-zeros.

    Returns:
        Corrected binary vector, length n, dtype uint8.
    """
    H = np.asarray(H)
    llr = np.asarray(llr, dtype=np.float64)
    hard_decision = np.asarray(hard_decision, dtype=np.uint8)
    m, n = H.shape

    if syndrome_vec is None:
        syndrome_vec = np.zeros(m, dtype=np.uint8)
    syndrome_vec = np.asarray(syndrome_vec, dtype=np.uint8)

    # Step 1 — Sort columns by reliability (ascending: least reliable first).
    # Pivots will be selected among the leading (least-reliable) columns;
    # the trailing (most-reliable) non-pivot columns become the info set.
    reliability_order = np.argsort(np.abs(llr), kind="stable")

    # Step 2 — Permute H columns and augment with syndrome.
    H_perm = H[:, reliability_order].astype(np.uint8)
    augmented = np.hstack([H_perm, syndrome_vec.reshape(-1, 1).astype(np.uint8)])

    # Row-reduce; pivots only searched within first n columns (not syndrome).
    R_aug, pivot_cols = gf2_row_echelon(augmented, n_pivot_cols=n)
    rank = len(pivot_cols)

    if rank == 0:
        return hard_decision.copy()

    R = R_aug[:, :n]
    s_transformed = R_aug[:, n]

    # Step 3 — Identify info set (non-pivot = most reliable independent).
    pivot_set = set(pivot_cols)
    info_cols = [c for c in range(n) if c not in pivot_set]

    # Step 4 — Set info bits from hard decision (permuted space).
    hard_perm = hard_decision[reliability_order]
    result_perm = np.zeros(n, dtype=np.uint8)
    for c in info_cols:
        result_perm[c] = hard_perm[c]

    # Step 5 — Back-substitute (top-down: R is upper-triangular in pivots).
    for i in range(rank - 1, -1, -1):
        pc = pivot_cols[i]
        rhs = s_transformed[i]
        for j in range(n):
            if j != pc:
                rhs ^= np.uint8(R[i, j]) & result_perm[j]
        result_perm[pc] = np.uint8(rhs) & np.uint8(1)

    # Step 6 — Un-permute.
    result = np.zeros(n, dtype=np.uint8)
    result[reliability_order] = result_perm

    # Never-degrade guard.
    osd_syn = (
        (H.astype(np.int32) @ result.astype(np.int32)) % 2
    ).astype(np.uint8)
    if not np.array_equal(osd_syn, syndrome_vec):
        return hard_decision.copy()

    return result
