"""Deterministic spectral defect atlas for trapping-set repair reuse."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

_ROUND = 12


class SpectralDefectAtlas:
    """Persistent atlas of defect signatures and successful repairs."""

    def __init__(self, max_patterns: int = 500):
        self.max_patterns = max(1, int(max_patterns))
        self.patterns: list[dict[str, Any]] = []

    def signature(self, vec: np.ndarray) -> str:
        arr = np.asarray(vec, dtype=np.float64)
        if arr.size == 0:
            return ""
        order = np.lexsort((np.arange(arr.size, dtype=np.int64), -np.abs(arr).astype(np.float64)))
        top = order[:4]
        return "_".join(str(int(i)) for i in top.tolist())

    def lookup(self, signature: str) -> dict[str, Any] | None:
        key = str(signature)
        for idx, pattern in enumerate(self.patterns):
            if str(pattern.get("signature", "")) == key:
                item = dict(pattern)
                item["pattern_index"] = int(idx)
                return item
        return None

    def query(self, signature: tuple[int, float, float] | str) -> str | None:
        if isinstance(signature, tuple):
            key = self._tuple_signature_key(signature)
        else:
            key = str(signature)
        hit = self.lookup(key)
        if hit is None:
            return None
        return str(hit.get("repair", ""))

    def record(
        self,
        signature: tuple[int, float, float] | str,
        repair_action: str,
        improvement: float = 1.0,
    ) -> None:
        if float(improvement) <= 0.0:
            return
        if isinstance(signature, tuple):
            sig_key = self._tuple_signature_key(signature)
        else:
            sig_key = str(signature)

        entry = {
            "signature": sig_key,
            "repair": str(repair_action),
            "improvement": round(float(improvement), _ROUND),
        }
        self.patterns.append(entry)
        if len(self.patterns) > self.max_patterns:
            self.patterns = self.patterns[-self.max_patterns :]

    def to_payload(self) -> dict[str, Any]:
        return {"patterns": list(self.patterns)}

    def save_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.to_payload(), f, sort_keys=True, indent=2)
            f.write("\n")

    def load_json(self, path: Path) -> None:
        if not path.exists():
            return
        data = json.loads(path.read_text(encoding="utf-8"))
        patterns = data.get("patterns", [])
        loaded: list[dict[str, Any]] = []
        for item in patterns:
            loaded.append(
                {
                    "signature": str(item.get("signature", "")),
                    "repair": str(item.get("repair", "")),
                    "improvement": round(float(item.get("improvement", 0.0)), _ROUND),
                }
            )
        self.patterns = loaded[-self.max_patterns :]

    def _tuple_signature_key(self, signature: tuple[int, float, float]) -> str:
        return "|".join(
            [
                str(int(signature[0])),
                str(round(float(signature[1]), _ROUND)),
                str(round(float(signature[2]), _ROUND)),
            ]
        )


def defect_signature(cluster_nodes: list[int] | tuple[int, ...] | np.ndarray, metrics: dict[str, Any]) -> tuple[int, float, float]:
    nodes = np.asarray(cluster_nodes, dtype=np.int64)
    return (
        int(nodes.size),
        round(float(metrics.get("ipr_localization_score", metrics.get("ipr_localization", 0.0))), _ROUND),
        round(float(metrics.get("nb_spectral_radius", 0.0)), _ROUND),
    )
