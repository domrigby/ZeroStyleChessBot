import argparse
from dataclasses import dataclass, field, fields
from datetime import datetime
from typing import List, Optional


@dataclass
class SearchConfig:
    """MCTS and self-play constants. Travels inside each pickled agent, like the rest of TrainingConfig."""

    c_puct: float = 3.0
    gamma: float = 0.99
    dirichlet_alpha: float = 1.0
    dirichlet_epsilon: float = 0.25
    max_game_length: int = 300
    tau_switch_move: int = 30
    tau_early: float = 1.0
    tau_late: float = 0.1
    curriculum_low_tau_games: int = 250


@dataclass
class TrainingConfig:
    """Everything parallel_train.py can be configured with. Paths are repo-relative or None (feature off)."""

    # Paths
    network_path: Optional[str] = "neural_nets/example_network.pt"
    expert_data_path: Optional[str] = None
    curriculum_csv_path: Optional[str] = None
    generated_data_dir: str = "neural_nets/generated_data"
    run_dir: Optional[str] = None

    # Processes
    num_agents: int = 4
    num_evaluators: int = 1
    num_trainers: int = 1

    # Network and optimiser
    seed: int = 42
    init_lr: float = 2e-6
    num_cnn_repeats: int = 32

    # Search and training
    sims: int = 1000
    maximum_lag: int = 8
    evaluator_batch_size: int = 256
    trainer_batch_size: int = 128
    weights_update_freq: int = 100

    # MCTS and self-play constants (flat on the command line: --c_puct, not --search.c_puct)
    search: SearchConfig = field(default_factory=SearchConfig)

    def resolved_run_dir(self) -> str:
        if self.run_dir is not None:
            return self.run_dir
        return datetime.now().strftime("sessions/run_at_%Y%m%d_%H%M%S")


HELP = {
    "network_path": "Weights to start from (scripts/download_weights.sh fetches the default)",
    "expert_data_path": "Directory of expert *.pkl data mixed into training batches",
    "curriculum_csv_path": "Lichess puzzle CSV for the puzzle curriculum; unset starts every game from the start position",
    "generated_data_dir": "Where self-play data is saved to and reloaded from",
    "run_dir": "Run directory to create or continue; defaults to sessions/run_at_<timestamp>",
    "num_agents": "Number of self-play agents",
    "num_evaluators": "Number of evaluators",
    "num_trainers": "Number of trainers",
    "seed": "Random seed",
    "init_lr": "Initial learning rate",
    "num_cnn_repeats": "Number of repeated ResNet blocks in the policy-value network",
    "sims": "MCTS simulations per move",
    "maximum_lag": "Maximum evaluations an agent may have outstanding before it waits",
    "evaluator_batch_size": "Maximum positions per evaluator forward pass",
    "trainer_batch_size": "Positions per training batch",
    "weights_update_freq": "Training steps between weight pushes to the evaluators",
    "c_puct": "PUCT exploration constant",
    "gamma": "Discount applied per ply when backing values up the search tree",
    "dirichlet_alpha": "Dirichlet noise concentration added to the priors in self-play",
    "dirichlet_epsilon": "Weight of the Dirichlet noise in the priors",
    "max_game_length": "Moves after which a self-play game is abandoned (not saved)",
    "tau_switch_move": "Move count after which moves are chosen with tau_late instead of tau_early",
    "tau_early": "Move-selection temperature up to tau_switch_move",
    "tau_late": "Move-selection temperature after tau_switch_move, and for the first curriculum games",
    "curriculum_low_tau_games": "With a curriculum, games played with tau_late from the first move",
}


def parse_training_config(argv: Optional[List[str]] = None) -> TrainingConfig:
    parser = argparse.ArgumentParser(description="Parallel self-play training")
    training_fields = [config_field for config_field in fields(TrainingConfig) if config_field.name != "search"]
    for config_field in [*training_fields, *fields(SearchConfig)]:
        argument_type = config_field.type if config_field.type in (int, float) else str
        parser.add_argument(
            f"--{config_field.name}", type=argument_type, default=config_field.default, help=HELP[config_field.name]
        )
    arguments = vars(parser.parse_args(argv))
    search = SearchConfig(
        **{config_field.name: arguments.pop(config_field.name) for config_field in fields(SearchConfig)}
    )
    return TrainingConfig(**arguments, search=search)
