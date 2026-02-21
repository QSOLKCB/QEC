"""
Ordered Statistics Decoding (OSD-0 and OSD-1).

Post-processing steps for belief-propagation decoders.  When BP fails
to converge, OSD uses the soft reliability information (LLRs) to
select an information set via Gaussian elimination and solve for the
most likely error pattern.
"""

from __future__ import annotations

import numpy as np

from .gf2 import gf2_row_echelon


# ═══════════════════════════════════════════════════════════════════════
# Shared OSD-0 core
# ═══════════════════════════════════════════════════════════════════════

def _osd0_core(H, llr, hard_decision, syndrome_vec):
    """Sort, row-reduce, and back-substitute (OSD-0 core).

    This is the shared computation used by both :func:`osd0` and
    :func:`osd1`.  It does NOT apply the never-degrade guard;
    callers are responsible for that.

    Args:
        H: Binary parity-check matrix, shape (m, n).
        llr: Per-variable log-likelihood ratios, length n.
        hard_decision: Binary hard-decision vector from BP, length n.
        syndrome_vec: Target syndrome, length m.

    Returns:
        (result, reliability_order, pivot_cols, rank, valid):
            result — OSD-0 solution in original (un-permuted) space.
            reliability_order — column sort permutation (ascending |llr|).
            pivot_cols — pivot columns in permuted space.
            rank — GF(2) rank of the permuted H.
            valid — True if result satisfies the syndrome.
    """
    m, n = H.shape

    # Step 1 — Sort columns by reliability (ascending: least reliable first).
    # Pivots will land on the least-reliable independent columns;
    # the trailing (most-reliable) non-pivot columns become the info set.
    reliability_order = np.argsort(np.abs(llr), kind="stable")

    # Step 2 — Permute H columns and augment with syndrome.
    H_perm = H[:, reliability_order].astype(np.uint8)
    augmented = np.hstack([H_perm, syndrome_vec.reshape(-1, 1).astype(np.uint8)])

    # Row-reduce; pivots only searched within first n columns (not syndrome).
    R_aug, pivot_cols = gf2_row_echelon(augmented, n_pivot_cols=n)
    rank = len(pivot_cols)

    if rank == 0:
        # No pivots: cannot solve; return hard_decision as the "solution".
        return hard_decision.copy(), reliability_order, pivot_cols, rank, False

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

    # Check syndrome validity (callers may still apply never-degrade).
    osd_syn = (
        (H.astype(np.int32) @ result.astype(np.int32)) % 2
    ).astype(np.uint8)
    valid = np.array_equal(osd_syn, syndrome_vec)

    return result, reliability_order, pivot_cols, rank, valid


# ═══════════════════════════════════════════════════════════════════════
# OSD-0
# ═══════════════════════════════════════════════════════════════════════

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

    result, _, _, _, valid = _osd0_core(H, llr, hard_decision, syndrome_vec)

    # Never-degrade guard.
    if not valid:
        return hard_decision.copy()

    return result


# ═══════════════════════════════════════════════════════════════════════
# OSD-1
# ═══════════════════════════════════════════════════════════════════════

def osd1(H, llr, hard_decision, syndrome_vec=None):
    """Order-1 Ordered Statistics Decoding.

    Extends OSD-0 by testing a single-bit flip on the least-reliable
    pivot column after the OSD-0 solution is obtained.  The candidate
    with the lowest Hamming weight that satisfies the syndrome is
    selected.

    The *never-degrade* guarantee is preserved: if neither OSD-0 nor
    the single-bit flip produces a valid syndrome match, the original
    *hard_decision* is returned unchanged.

    Deterministic tie-breaking: when OSD-0 and the flipped candidate
    have equal Hamming weight, OSD-0 is preferred.

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

    # ── OSD-0 phase ──
    result_0, reliability_order, pivot_cols, rank, osd0_valid = \
        _osd0_core(H, llr, hard_decision, syndrome_vec)

    if rank == 0:
        # No pivots: nothing to flip.
        return hard_decision.copy()

    # ── OSD-1 phase: flip the single least-reliable pivot bit ──
    # pivot_cols[0] is the least-reliable pivot in permuted space
    # (columns are sorted by ascending |llr|, so index 0 = smallest |llr|).
    flip_col_perm = pivot_cols[0]
    flip_col_orig = reliability_order[flip_col_perm]

    result_1 = result_0.copy()
    result_1[flip_col_orig] ^= 1

    osd1_syn = (
        (H.astype(np.int32) @ result_1.astype(np.int32)) % 2
    ).astype(np.uint8)
    osd1_valid = np.array_equal(osd1_syn, syndrome_vec)

    # ── Select best valid candidate ──
    # Candidates are (weight, tie_order, vector).
    # tie_order=0 for OSD-0 (preferred on ties), tie_order=1 for OSD-1.
    candidates = []
    if osd0_valid:
        candidates.append((int(np.sum(result_0)), 0, result_0))
    if osd1_valid:
        candidates.append((int(np.sum(result_1)), 1, result_1))

    if not candidates:
        # Never-degrade guard: neither candidate satisfies the syndrome.
        return hard_decision.copy()

    # Select candidate with lowest Hamming weight; ties broken by order.
    candidates.sort(key=lambda x: (x[0], x[1]))
    return candidates[0][2]
