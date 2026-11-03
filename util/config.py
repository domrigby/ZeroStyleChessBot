import argparse
from dataclasses import dataclass, fields
from datetime import datetime
from typing import List, Optional


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
}


def parse_training_config(argv: Optional[List[str]] = None) -> TrainingConfig:
    parser = argparse.ArgumentParser(description="Parallel self-play training")
    for field in fields(TrainingConfig):
        argument_type = field.type if field.type in (int, float) else str
        parser.add_argument(f"--{field.name}", type=argument_type, default=field.default, help=HELP[field.name])
    return TrainingConfig(**vars(parser.parse_args(argv)))
