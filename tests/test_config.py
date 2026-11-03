import os
from pathlib import Path

import pandas as pd
import pytest

from neural_nets.data.fen_chess_puzzles.curiculum_maker import START_STATE, load_curricula
from util.config import TrainingConfig, parse_training_config
from zero_style_chess_engine.memory import Memory

PATH_FIELDS = ["network_path", "expert_data_path", "curriculum_csv_path", "generated_data_dir", "run_dir"]
MATE_IN_TWO_FEN = "r2qkb1r/pp2nppp/3p4/2pNN1B1/2BnP3/3P4/PPP2PPP/R2bK2R w KQkq - 1 10"
MATE_IN_TWO_MOVES = "d5f6 g7f6 c4f7"
FIRST_CURRICULUM_EPISODE = 0
CURRICULUM_EPISODE_LIMIT = 800


@pytest.mark.parametrize("field_name", [pytest.param(name, id=name) for name in PATH_FIELDS])
def test_default_paths_are_relative_or_unset(field_name: str) -> None:
    path = getattr(TrainingConfig(), field_name)

    assert path is None or not os.path.isabs(path)


def test_command_line_overrides_defaults() -> None:
    sims = 16
    csv_path = "puzzles.csv"

    config = parse_training_config(["--sims", str(sims), "--curriculum_csv_path", csv_path])

    assert config.sims == sims
    assert config.curriculum_csv_path == csv_path
    assert config.maximum_lag == TrainingConfig().maximum_lag


@pytest.mark.parametrize(
    "csv_name",
    [
        pytest.param(None, id="unset"),
        pytest.param("missing.csv", id="missing_file"),
    ],
)
def test_missing_curriculum_falls_back_to_start_position(tmp_path: Path, csv_name: str) -> None:
    num_agents = 3
    csv_path = None if csv_name is None else str(tmp_path / csv_name)

    if csv_path is None:
        curricula = load_curricula(csv_path, num_agents)
    else:
        with pytest.warns(UserWarning, match="not found"):
            curricula = load_curricula(csv_path, num_agents)

    assert curricula == [None] * num_agents


def test_curriculum_gives_every_agent_puzzle_start_states(tmp_path: Path) -> None:
    num_agents = 2
    csv_path = tmp_path / "puzzles.csv"
    pd.DataFrame({"FEN": [MATE_IN_TWO_FEN], "Moves": [MATE_IN_TWO_MOVES], "Themes": ["mateIn2 short"]}).to_csv(
        csv_path, index=False
    )

    curricula = load_curricula(str(csv_path), num_agents)

    assert len(curricula) == num_agents
    for curriculum in curricula:
        assert curriculum is not None
        start_state, _ = curriculum.get_start_state(FIRST_CURRICULUM_EPISODE)
        assert start_state != START_STATE
        assert curriculum.get_start_state(CURRICULUM_EPISODE_LIMIT)[0] == START_STATE


def test_missing_expert_data_falls_back_to_empty(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)

    with pytest.warns(UserWarning, match="not found"):
        memory = Memory(
            length_buffer=10, expert_data_path=str(tmp_path / "missing"), generated_data_dir=str(tmp_path / "generated")
        )

    assert memory.expert_data == []
    assert len(memory) == 0
