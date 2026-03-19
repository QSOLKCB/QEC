from __future__ import annotations

import json

import numpy as np

from qec.analysis.spectral_defect_atlas import SpectralDefectAtlas, defect_signature
from qec.discovery.threshold_search import PhaseDiagramOrchestrator, SpectralSearchConfig, run_spectral_threshold_search
from qec.generation.deterministic_construction import construct_deterministic_tanner_graph


def _small_graph() -> np.ndarray:
    spec = {
        "num_variables": 8,
        "num_checks": 4,
        "variable_degree": 2,
        "check_degree": 4,
    }
    return construct_deterministic_tanner_graph(spec)


def test_defect_signature_deterministic() -> None:
    nodes = [5, 1, 3, 7]
    metrics = {
        "ipr_localization_score": 0.0842000000004,
        "nb_spectral_radius": 4.1200000000002,
    }

    s1 = defect_signature(nodes, metrics)
    s2 = defect_signature(nodes, metrics)

    assert s1 == s2
    assert s1 == (4, 0.0842, 4.12)




def test_atlas_signature_generation_is_deterministic() -> None:
    atlas = SpectralDefectAtlas(max_patterns=5)
    vec = np.array([0.5, -0.1, 0.9, 0.2, -0.9], dtype=np.float64)
    s1 = atlas.signature(vec)
    s2 = atlas.signature(vec)
    assert s1 == s2
    assert isinstance(s1, str)

def test_atlas_lookup_returns_repair() -> None:
    atlas = SpectralDefectAtlas(max_patterns=5)
    sig = "1_2_3_4"
    atlas.record(sig, "flow_edge_3", 0.0064)

    hit = atlas.lookup(sig)
    assert hit is not None
    assert hit["repair"] == "flow_edge_3"


def test_repair_reuse_applies_existing_strategy(tmp_path, monkeypatch) -> None:
    H0 = _small_graph()

    def _fake_eval(self, H, *, max_phase_diagram_size, seed):
        return {"measured_boundary": {"mean_boundary_spectral_radius": 0.05}}

    monkeypatch.setattr(PhaseDiagramOrchestrator, "evaluate", _fake_eval)

    from qec.discovery import threshold_search as mod

    def _reject_nb_gradient(self, H, steps=1):
        return np.zeros_like(np.asarray(H, dtype=np.float64)), [{"operator": "reject"}]

    monkeypatch.setattr(mod.NBGradientMutator, "mutate", _reject_nb_gradient)

    cfg = SpectralSearchConfig(
        iterations=1,
        max_phase_diagram_size=1,
        output_dir=str(tmp_path),
        enable_nb_flow_mutation=True,
        enable_spectral_defect_atlas=True,
    )
    run_spectral_threshold_search(H0, config=cfg)

    first_payload = json.loads((tmp_path / "candidate_metrics.json").read_text(encoding="utf-8"))
    first_flow = [m for m in first_payload["candidates"] if m.get("source") == "nb_flow"][0]
    assert first_flow["atlas_hit"] is False

    run_spectral_threshold_search(H0, config=cfg)

    second_payload = json.loads((tmp_path / "candidate_metrics.json").read_text(encoding="utf-8"))
    second_flow = [m for m in second_payload["candidates"] if m.get("source") == "nb_flow"][0]
    assert second_flow["atlas_hit"] is True
    assert isinstance(second_flow["atlas_pattern_index"], int)
    assert isinstance(second_flow["defect_signature"], str)
    assert second_flow["repair_action"].startswith("flow_edge_")


def test_atlas_json_canonical_structure(tmp_path) -> None:
    atlas = SpectralDefectAtlas(max_patterns=5)
    atlas.record("11_12_13_14", "flow_edge_3", 0.0064)
    out = tmp_path / "spectral_defect_atlas.json"
    atlas.save_json(out)

    text = out.read_text(encoding="utf-8")
    payload = json.loads(text)

    assert list(payload.keys()) == ["patterns"]
    assert payload["patterns"][0]["signature"] == "11_12_13_14"
    assert payload["patterns"][0]["repair"] == "flow_edge_3"
    assert payload["patterns"][0]["improvement"] == 0.0064
