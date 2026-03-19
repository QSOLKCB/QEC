"""
v9.0.0 — Discovery Archive.

Tracks elite candidates across multiple categories.  Each category
stores top-k candidates ordered by the category metric.

Categories:
  best_composite, lowest_instability, lowest_spectral_radius,
  highest_bethe_margin, lowest_cycle_density, highest_entropy,
  most_novel.

Layer 3 — Discovery.
Does not import or modify the decoder (Layer 1).
Fully deterministic: no randomness, no global state, no input mutation.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from qec.discovery.novelty import extract_feature_vector


_ROUND = 12

_CATEGORIES = [
    "best_composite",
    "lowest_instability",
    "lowest_spectral_radius",
    "highest_bethe_margin",
    "lowest_cycle_density",
    "highest_entropy",
    "most_novel",
]


def _category_key(category: str, entry: dict[str, Any]) -> tuple:
    """Return a deterministic sort key for a category.

    Lower sort key = better candidate for that category.
    Secondary key is candidate_id for deterministic tie-breaking.
    """
    obj = entry.get("objectives", {})
    cid = entry.get("candidate_id", "")

    if category == "best_composite":
        return (obj.get("composite_score", float("inf")), cid)
    elif category == "lowest_instability":
        return (obj.get("instability_score", float("inf")), cid)
    elif category == "lowest_spectral_radius":
        return (obj.get("spectral_radius", float("inf")), cid)
    elif category == "highest_bethe_margin":
        return (-obj.get("bethe_margin", 0.0), cid)
    elif category == "lowest_cycle_density":
        return (obj.get("cycle_density", float("inf")), cid)
    elif category == "highest_entropy":
        return (-obj.get("entropy", 0.0), cid)
    elif category == "most_novel":
        return (-entry.get("novelty", 0.0), cid)
    else:
        return (0.0, cid)


def create_archive(*, top_k: int = 5) -> dict[str, Any]:
    """Create an empty discovery archive.

    Parameters
    ----------
    top_k : int
        Maximum entries per category.

    Returns
    -------
    dict[str, Any]
        Archive dictionary with empty category lists.
    """
    return {
        "top_k": top_k,
        "categories": {cat: [] for cat in _CATEGORIES},
    }


def update_discovery_archive(
    archive: dict[str, Any],
    candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    """Update the archive with new candidates.

    Each candidate should have keys: candidate_id, H, objectives,
    metrics, generation, novelty.

    Parameters
    ----------
    archive : dict[str, Any]
        Existing archive from ``create_archive``.
    candidates : list[dict[str, Any]]
        New candidates to consider for archive inclusion.

    Returns
    -------
    dict[str, Any]
        Updated archive (new object, does not mutate input).
    """
    top_k = archive["top_k"]
    new_archive = {
        "top_k": top_k,
        "categories": {},
    }

    for cat in _CATEGORIES:
        # Combine existing entries with new candidates
        existing = list(archive["categories"].get(cat, []))
        combined = existing + [
            {
                "candidate_id": c.get("candidate_id", ""),
                "H": c.get("H"),
                "objectives": c.get("objectives", {}),
                "metrics": c.get("metrics", {}),
                "generation": c.get("generation", 0),
                "novelty": c.get("novelty", 0.0),
            }
            for c in candidates
            if c.get("is_feasible", True)
        ]

        # Deduplicate by candidate_id, keeping latest
        seen: dict[str, dict[str, Any]] = {}
        for entry in combined:
            cid = entry.get("candidate_id", "")
            seen[cid] = entry
        unique = sorted(seen.values(), key=lambda e: _category_key(cat, e))

        new_archive["categories"][cat] = unique[:top_k]

    return new_archive


def get_archive_features(archive: dict[str, Any]) -> list[np.ndarray]:
    """Extract feature vectors from all unique archive elites.

    Parameters
    ----------
    archive : dict[str, Any]
        Discovery archive.

    Returns
    -------
    list[np.ndarray]
        Feature vectors for novelty computation.
    """
    seen_ids: set[str] = set()
    features: list[np.ndarray] = []

    for cat in _CATEGORIES:
        for entry in archive["categories"].get(cat, []):
            cid = entry.get("candidate_id", "")
            if cid not in seen_ids:
                seen_ids.add(cid)
                obj = entry.get("objectives", {})
                features.append(extract_feature_vector(obj))

    return features


def get_archive_summary(archive: dict[str, Any]) -> dict[str, Any]:
    """Produce a JSON-serializable summary of the archive.

    Parameters
    ----------
    archive : dict[str, Any]
        Discovery archive.

    Returns
    -------
    dict[str, Any]
        Summary with category sizes and best values.
    """
    summary: dict[str, Any] = {}
    for cat in _CATEGORIES:
        entries = archive["categories"].get(cat, [])
        if entries:
            best = entries[0]
            summary[cat] = {
                "count": len(entries),
                "best_candidate_id": best.get("candidate_id", ""),
                "best_value": _category_key(cat, best)[0],
            }
        else:
            summary[cat] = {"count": 0, "best_candidate_id": "", "best_value": None}

    summary["total_unique"] = len(get_archive_features(archive))
    return summary
