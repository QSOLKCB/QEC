"""Deterministic defect-guided candidate swap reordering."""

from __future__ import annotations

import numpy as np

from src.qec.analysis.api import SpectralDefect



def defect_guided_mutations(
    H_pc,
    defects: list[SpectralDefect],
    candidate_swaps: list[tuple[int, int, int, int]],
) -> list[tuple[int, int, int, int]]:
    del H_pc
    if not defects or not candidate_swaps:
        return list(candidate_swaps)

    support_weight: dict[int, float] = {}
    for defect_rank, defect in enumerate(defects):
        rank_weight = float(len(defects) - defect_rank)
        for vi in defect.support_nodes:
            support_weight[int(vi)] = support_weight.get(int(vi), 0.0) + float(defect.severity) * rank_weight

    ranked: list[tuple[tuple[float, int, int, int, int], tuple[int, int, int, int]]] = []
    for order, swap in enumerate(candidate_swaps):
        ci, vi, cj, vj = swap
        bias = support_weight.get(int(vi), 0.0) + support_weight.get(int(vj), 0.0)
        key = (-round(bias, 12), order, ci, vi, cj, vj)
        ranked.append((key, swap))

    ranked.sort(key=lambda item: item[0])
    return [swap for _, swap in ranked]
