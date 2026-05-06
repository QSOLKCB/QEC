from pathlib import Path


MODULES = {
    "game_world_intake_contract.py": Path(__file__).resolve().parents[1] / "src" / "qec" / "analysis" / "game_world_intake_contract.py",
    "game_world_adapter_contract.py": Path(__file__).resolve().parents[1] / "src" / "qec" / "analysis" / "game_world_adapter_contract.py",
    "game_world_observation_snapshot.py": Path(__file__).resolve().parents[1] / "src" / "qec" / "analysis" / "game_world_observation_snapshot.py",
    "game_world_episode_trace.py": Path(__file__).resolve().parents[1] / "src" / "qec" / "analysis" / "game_world_episode_trace.py",
    "game_world_strategy_probe.py": Path(__file__).resolve().parents[1] / "src" / "qec" / "analysis" / "game_world_strategy_probe.py",
}

FORBIDDEN = [
    ".extract(", ".extractall(", "importlib", "__import__(", "subprocess", "exec(", "eval(",
    "pygame", "gym", "render", "step_world", "execute_action", "run_game", "play_game", "train_policy",
    "learned_policy", "neural", "probabilistic", "probability", "best_action", "optimal_action",
    "reward", "score_heuristic", "ChaosReplayVerdict", "GameWorldInteractionReport",
]


def test_v156x_scope_boundaries_forbidden_tokens():
    for module_name, module_path in MODULES.items():
        text = module_path.read_text(encoding="utf-8")
        for token in FORBIDDEN:
            assert token not in text, f"{module_name} contains forbidden token: {token}"

        if module_name == "game_world_intake_contract.py":
            continue
        assert "zipfile.ZipFile" not in text, f"{module_name} contains forbidden token: zipfile.ZipFile"
