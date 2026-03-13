from __future__ import annotations

from src.qec.analysis.defect_catalog import SpectralDefect


def defect_guided_mutations(
    H_pc,
    defects: list[SpectralDefect],
    candidate_swaps: list[tuple[int, int, int, int]],
) -> list[tuple[int, int, int, int]]:
    """Sort candidate swaps by summed defect severity around swap variable nodes."""
    del H_pc
    node_severity: dict[int, float] = {}
    for defect in defects:
        sev = float(defect.severity)
        for vi in defect.support_nodes:
            node_severity[int(vi)] = node_severity.get(int(vi), 0.0) + sev

    ranked = []
    for swap in candidate_swaps:
        ci, vi, cj, vj = swap
        prox = node_severity.get(int(vi), 0.0) + node_severity.get(int(vj), 0.0)
        ranked.append(((-prox, ci, vi, cj, vj), swap))

    ranked.sort(key=lambda x: x[0])
    return [item[1] for item in ranked]
