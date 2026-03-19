"""Normalized aggregate scoring and ranking for benchmark stress results.

Version: v71.0.0
"""

from src.qec.modules.common import _EXCLUDED_KEYS, _is_finite_numeric


def build_scores(result: dict) -> list:
    """Compute normalized aggregate scores per genome and rank them.

    Scoring pipeline:
        1. Per genome, per metric: mean across scenarios (NaN excluded).
        2. Min-max normalization per metric across genomes.
           Constant columns → 0.5.
        3. Aggregate: mean of normalized metric values.
        4. Rank: score descending, genome_id ascending for tie-break.

    Parameters
    ----------
    result : dict
        Output of ``run_benchmark_stress``.  Must contain ``"table"`` key.

    Returns
    -------
    list[dict]
        Ranked list: ``{"genome_id": str, "score": float, "rank": int}``.

    Raises
    ------
    ValueError
        If ``result`` has no ``"table"`` key.
    """
    if "table" not in result:
        raise ValueError("Result dict missing 'table' key")

    table = result["table"]
    if not table:
        return []

    # Collect genome_ids in order of first appearance
    seen_genomes: dict = {}
    for row in table:
        gid = row["genome_id"]
        if gid not in seen_genomes:
            seen_genomes[gid] = []
        seen_genomes[gid].append(row)

    # Identify numeric metric keys across all rows (exclude NaN)
    all_keys: set = set()
    for row in table:
        for key in row:
            if key in _EXCLUDED_KEYS:
                continue
            if _is_finite_numeric(row[key]):
                all_keys.add(key)
    numeric_keys = sorted(all_keys)

    if not numeric_keys:
        return []

    # Per-genome, per-metric: compute mean across scenarios (exclude NaN)
    genome_agg: dict = {}
    for gid, rows in seen_genomes.items():
        agg: dict = {}
        for key in numeric_keys:
            values = [
                r[key] for r in rows
                if _is_finite_numeric(r.get(key))
            ]
            if values:
                agg[key] = sum(values) / len(values)
        genome_agg[gid] = agg

    # Min-max normalization per metric across genomes
    normalized: dict = {gid: {} for gid in genome_agg}
    for key in numeric_keys:
        values = [
            genome_agg[gid][key]
            for gid in genome_agg
            if key in genome_agg[gid]
        ]
        if not values:
            continue
        min_val = min(values)
        max_val = max(values)
        for gid in genome_agg:
            if key not in genome_agg[gid]:
                continue
            if max_val == min_val:
                normalized[gid][key] = 0.5
            else:
                normalized[gid][key] = (
                    (genome_agg[gid][key] - min_val) / (max_val - min_val)
                )

    # Aggregate: mean of normalized values
    scores: list = []
    for gid in seen_genomes:
        norm_vals = list(normalized[gid].values())
        if norm_vals:
            score = sum(norm_vals) / len(norm_vals)
        else:
            score = 0.0
        scores.append({
            "genome_id": gid,
            "score": score,
        })

    # Rank: score DESC, genome_id ASC for tie-break
    scores.sort(key=lambda x: (-x["score"], x["genome_id"]))
    for i, s in enumerate(scores):
        s["rank"] = i + 1

    return scores
