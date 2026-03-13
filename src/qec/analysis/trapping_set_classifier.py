"""Deterministic trapping-set classifier for localized spectral defects."""

from __future__ import annotations


_SMALL_TRAPPING_SET = "SMALL_TRAPPING_SET"
_CYCLE_CLUSTER = "CYCLE_CLUSTER"
_PSEUDO_CODEWORD_BASIN = "PSEUDO_CODEWORD_BASIN"
_BENIGN_GLOBAL_MODE = "BENIGN_GLOBAL_MODE"



def classify_trapping_set(
    support_nodes: list[int] | tuple[int, ...],
    a: int,
    b: int,
    localization_metrics: dict[str, float] | None = None,
) -> str:
    metrics = localization_metrics or {}
    ipr = float(metrics.get("ipr", 0.0))
    support_fraction = float(metrics.get("support_fraction", 0.0))
    topk_mass_fraction = float(metrics.get("topk_mass_fraction", 0.0))

    support_size = len(tuple(support_nodes))
    a_eff = int(a) if int(a) > 0 else int(support_size)

    if a_eff <= 6 and b > 0 and b <= max(1, a_eff):
        return _SMALL_TRAPPING_SET
    if b == 0 and support_fraction >= 0.5 and topk_mass_fraction <= 0.7:
        return _BENIGN_GLOBAL_MODE
    if ipr >= 0.25 and support_fraction <= 0.35:
        return _PSEUDO_CODEWORD_BASIN
    return _CYCLE_CLUSTER
