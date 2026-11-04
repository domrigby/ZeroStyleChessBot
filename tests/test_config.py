import os
from pathlib import Path
from typing import Any, Dict, Iterator, List, Type

import pandas as pd
import pytest
from hydra import compose, initialize_config_dir
from hydra.errors import ConfigCompositionException
from omegaconf.errors import MissingMandatoryValue
from pydantic import BaseModel, ValidationError

from neural_nets.data.fen_chess_puzzles.curriculum import START_STATE, load_curricula
from zero_style_chess_engine.config import (
    HumanVsMachineConfig,
    SearchConfig,
    SupervisedConfig,
    TrainingConfig,
    VsStockfishConfig,
    check_paths_exist,
    from_hydra,
    register_defaults,
)
from zero_style_chess_engine.memory import Memory

CONF_DIR = str(Path(__file__).resolve().parents[1] / "conf")
PATH_FIELDS = ["network_path", "expert_data_path", "curriculum_csv_path", "generated_data_dir"]
MATE_IN_TWO_FEN = "r2qkb1r/pp2nppp/3p4/2pNN1B1/2BnP3/3P4/PPP2PPP/R2bK2R w KQkq - 1 10"
MATE_IN_TWO_MOVES = "d5f6 g7f6 c4f7"
FIRST_CURRICULUM_EPISODE = 0
CURRICULUM_EPISODE_LIMIT = 800

# The values the argparse/dataclass config had before the move to Hydra + Pydantic
EXPECTED_TRAINING_DEFAULTS: Dict[str, Any] = {
    "network_path": "neural_nets/example_network.pt",
    "expert_data_path": None,
    "curriculum_csv_path": None,
    "generated_data_dir": "neural_nets/generated_data",
    "num_agents": 4,
    "num_evaluators": 1,
    "num_trainers": 1,
    "seed": 42,
    "init_lr": 2e-6,
    "num_cnn_repeats": 32,
    "sims": 1000,
    "maximum_lag": 8,
    "evaluator_batch_size": 256,
    "trainer_batch_size": 128,
    "weights_update_freq": 100,
    "search": {
        "c_puct": 3.0,
        "gamma": 0.99,
        "dirichlet_alpha": 1.0,
        "dirichlet_epsilon": 0.25,
        "max_game_length": 300,
        "tau_switch_move": 30,
        "tau_early": 1.0,
        "tau_late": 0.1,
        "curriculum_low_tau_games": 250,
    },
}


@pytest.fixture(autouse=True)
def hydra_conf() -> Iterator[None]:
    register_defaults("training_defaults", TrainingConfig)
    register_defaults("supervised_defaults", SupervisedConfig)
    register_defaults("vs_stockfish_defaults", VsStockfishConfig)
    register_defaults("human_vs_machine_defaults", HumanVsMachineConfig)
    with initialize_config_dir(config_dir=CONF_DIR, version_base="1.3"):
        yield


def compose_training(overrides: List[str]) -> TrainingConfig:
    return from_hydra(TrainingConfig, compose(config_name="train", overrides=overrides))


def test_defaults_match_previous_values() -> None:
    assert TrainingConfig().model_dump() == EXPECTED_TRAINING_DEFAULTS


def test_yaml_without_overrides_gives_model_defaults() -> None:
    assert compose_training([]) == TrainingConfig()


@pytest.mark.parametrize("field_name", [pytest.param(name, id=name) for name in PATH_FIELDS])
def test_default_paths_are_relative_or_unset(field_name: str) -> None:
    path = getattr(TrainingConfig(), field_name)

    assert path is None or not os.path.isabs(path)


def test_command_line_overrides_defaults() -> None:
    sims = 16
    csv_path = "puzzles.csv"

    config = compose_training([f"sims={sims}", f"curriculum_csv_path={csv_path}"])

    assert config.sims == sims
    assert config.curriculum_csv_path == csv_path
    assert config.maximum_lag == TrainingConfig().maximum_lag


def test_command_line_overrides_nested_search_constants() -> None:
    c_puct = 1.5
    max_game_length = 50

    config = compose_training([f"search.c_puct={c_puct}", f"search.max_game_length={max_game_length}"])

    assert config.search.c_puct == c_puct
    assert config.search.max_game_length == max_game_length
    assert config.search.gamma == TrainingConfig().search.gamma


def test_optional_path_can_be_unset_from_command_line() -> None:
    assert compose_training(["network_path=null"]).network_path is None


@pytest.mark.parametrize("override", ["search.cpuct=1", "num_agent=2"])
def test_unknown_key_is_rejected_by_hydra(override: str) -> None:
    with pytest.raises(ConfigCompositionException):
        compose_training([override])


def test_unknown_key_is_rejected_by_model() -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        TrainingConfig.model_validate({"search": {"cpuct": 1.0}})


def test_config_is_frozen() -> None:
    config = TrainingConfig()

    with pytest.raises(ValidationError):
        config.sims = 1  # type: ignore[misc]


@pytest.mark.parametrize(
    "field_name",
    [
        "num_agents",
        "num_evaluators",
        "num_trainers",
        "sims",
        "maximum_lag",
        "evaluator_batch_size",
        "trainer_batch_size",
        "weights_update_freq",
        "num_cnn_repeats",
    ],
)
def test_training_counts_must_be_at_least_one(field_name: str) -> None:
    assert getattr(TrainingConfig(**{field_name: 1}), field_name) == 1
    with pytest.raises(ValidationError, match=field_name):
        TrainingConfig(**{field_name: 0})


def test_learning_rate_must_be_positive() -> None:
    with pytest.raises(ValidationError, match="init_lr"):
        TrainingConfig(init_lr=0.0)


# (field, a valid value at the boundary, an invalid value just past it)
SEARCH_BOUNDARIES = [
    ("max_game_length", 1, 0),
    ("tau_switch_move", 0, -1),
    ("curriculum_low_tau_games", 0, -1),
    ("c_puct", 1e-9, 0.0),
    ("dirichlet_alpha", 1e-9, 0.0),
    ("tau_late", 1e-9, 0.0),
    ("gamma", 1.0, 1.0001),
    ("gamma", 1e-9, 0.0),
    ("dirichlet_epsilon", 0.0, -0.01),
    ("dirichlet_epsilon", 1.0, 1.01),
]


@pytest.mark.parametrize(
    "field_name, valid, invalid",
    [pytest.param(*case, id=f"{case[0]}={case[2]}") for case in SEARCH_BOUNDARIES],
)
def test_search_values_are_range_checked(field_name: str, valid: float, invalid: float) -> None:
    assert getattr(SearchConfig(**{field_name: valid}), field_name) == valid
    with pytest.raises(ValidationError, match=field_name):
        SearchConfig(**{field_name: invalid})


def test_tau_early_must_be_positive() -> None:
    with pytest.raises(ValidationError, match="tau_early"):
        SearchConfig(tau_early=0.0, tau_late=0.0)


def test_tau_late_may_equal_but_not_exceed_tau_early() -> None:
    assert SearchConfig(tau_early=0.5, tau_late=0.5).tau_late == 0.5
    with pytest.raises(ValidationError, match="tau_late"):
        SearchConfig(tau_early=0.5, tau_late=0.6)


def test_search_validation_applies_through_hydra() -> None:
    with pytest.raises(ValidationError, match="tau_late"):
        compose_training(["search.tau_late=2.0"])


@pytest.mark.parametrize(
    "config_name, model, required_field",
    [
        pytest.param("supervised", SupervisedConfig, "data_dir", id="supervised"),
        pytest.param("vs_stockfish", VsStockfishConfig, "stockfish_path", id="vs_stockfish"),
    ],
)
def test_required_field_must_be_given(
    config_name: str, model: Type[BaseModel], required_field: str, tmp_path: Path
) -> None:
    with pytest.raises(MissingMandatoryValue, match=required_field):
        from_hydra(model, compose(config_name=config_name))

    config = from_hydra(model, compose(config_name=config_name, overrides=[f"{required_field}={tmp_path}"]))

    assert getattr(config, required_field) == str(tmp_path)


def test_previously_hard_coded_script_values_are_defaults() -> None:
    supervised = SupervisedConfig(data_dir="data")
    vs_stockfish = VsStockfishConfig(stockfish_path="stockfish")

    assert (supervised.batch_size, supervised.init_lr) == (64, 1e-3)
    assert (vs_stockfish.network_path, vs_stockfish.num_rollouts) == ("neural_nets/example_network.pt", 2000)
    assert (vs_stockfish.stockfish_elo, vs_stockfish.stockfish_move_time) == (1600, 1.0)


def test_check_paths_exist_reports_every_missing_path(tmp_path: Path) -> None:
    config = TrainingConfig(
        network_path=str(tmp_path / "missing.pt"),
        expert_data_path=str(tmp_path / "missing_expert"),
        curriculum_csv_path=str(tmp_path / "missing.csv"),
    )

    with pytest.raises(FileNotFoundError) as error:
        check_paths_exist(config)

    for field_name in ["network_path", "expert_data_path", "curriculum_csv_path"]:
        assert field_name in str(error.value)


def test_check_paths_exist_accepts_existing_and_unset_paths(tmp_path: Path) -> None:
    network = tmp_path / "net.pt"
    network.touch()

    check_paths_exist(TrainingConfig(network_path=str(network)))
    check_paths_exist(TrainingConfig(network_path=None))
    check_paths_exist(SupervisedConfig(data_dir=str(tmp_path)))


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


@pytest.mark.parametrize(
    "config_name, model, required",
    [
        pytest.param("human_vs_machine", HumanVsMachineConfig, [], id="human_vs_machine"),
        pytest.param("vs_stockfish", VsStockfishConfig, ["stockfish_path=stockfish"], id="vs_stockfish"),
    ],
)
def test_search_display_is_on_by_default_and_can_be_turned_off(
    config_name: str, model: Type[HumanVsMachineConfig | VsStockfishConfig], required: List[str]
) -> None:
    assert from_hydra(model, compose(config_name=config_name, overrides=required)).show_search
    assert not from_hydra(
        model, compose(config_name=config_name, overrides=[*required, "show_search=false"])
    ).show_search


def test_move_selection_defaults_to_visits_and_rejects_unknown_values() -> None:
    assert from_hydra(HumanVsMachineConfig, compose(config_name="human_vs_machine")).move_selection == "visits"
    q_config = compose(config_name="vs_stockfish", overrides=["stockfish_path=stockfish", "move_selection=q"])
    assert from_hydra(VsStockfishConfig, q_config).move_selection == "q"

    with pytest.raises(ValidationError, match="move_selection"):
        from_hydra(HumanVsMachineConfig, compose(config_name="human_vs_machine", overrides=["move_selection=value"]))


def test_display_mode_defaults_to_play_and_rejects_unknown_values() -> None:
    assert from_hydra(HumanVsMachineConfig, compose(config_name="human_vs_machine")).display_mode == "play"
    analysis = compose(config_name="human_vs_machine", overrides=["display_mode=analysis"])
    assert from_hydra(HumanVsMachineConfig, analysis).display_mode == "analysis"

    with pytest.raises(ValidationError, match="display_mode"):
        from_hydra(HumanVsMachineConfig, compose(config_name="human_vs_machine", overrides=["display_mode=cheat"]))


@pytest.mark.parametrize("mode", ["play", "analysis"])
def test_human_vs_machine_mode_presets_differ_from_the_base_config_only_in_display_mode(mode: str) -> None:
    preset = compose(config_name=f"human_vs_machine_{mode}", overrides=["num_rollouts=500"])

    assert from_hydra(HumanVsMachineConfig, preset) == HumanVsMachineConfig(num_rollouts=500, display_mode=mode)


def test_human_vs_machine_rollouts_are_set_from_the_command_line() -> None:
    assert from_hydra(HumanVsMachineConfig, compose(config_name="human_vs_machine")) == HumanVsMachineConfig()

    config = from_hydra(HumanVsMachineConfig, compose(config_name="human_vs_machine", overrides=["num_rollouts=500"]))

    assert config.num_rollouts == 500
    assert config.network_path == "neural_nets/example_network.pt"
