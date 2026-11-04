"""
Typed, validated configuration models. Defaults live here; the YAML files in conf/ only hold Hydra settings and
deliberate overrides.
"""

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


# How the engine picks a move after searching. Most visited is the robust choice: a high Q on a few visits is mostly noise
MoveSelection = Literal["visits", "q"]


class _Config(BaseModel):
    # extra="forbid": a misspelt key (search.cpuct=1) fails at startup instead of being ignored
    model_config = ConfigDict(extra="forbid", frozen=True)


class SearchConfig(_Config):
    """MCTS and self-play constants. Travels inside each pickled agent, like the rest of TrainingConfig."""

    c_puct: float = Field(3.0, gt=0, description="PUCT exploration constant")
    gamma: float = Field(
        0.99, gt=0, le=1, description="Discount applied per ply when backing values up the search tree"
    )
    dirichlet_alpha: float = Field(
        1.0, gt=0, description="Dirichlet noise concentration added to the priors in self-play"
    )
    dirichlet_epsilon: float = Field(0.25, ge=0, le=1, description="Weight of the Dirichlet noise in the priors")
    max_game_length: int = Field(300, ge=1, description="Moves after which a self-play game is abandoned (not saved)")
    tau_switch_move: int = Field(
        30, ge=0, description="Move count after which moves are chosen with tau_late instead of tau_early"
    )
    tau_early: float = Field(1.0, gt=0, description="Move-selection temperature up to tau_switch_move")
    tau_late: float = Field(
        0.1, gt=0, description="Move-selection temperature after tau_switch_move, and for the first curriculum games"
    )
    curriculum_low_tau_games: int = Field(
        250, ge=0, description="With a curriculum, games played with tau_late from the first move"
    )

    @model_validator(mode="after")
    def _tau_late_not_above_tau_early(self) -> "SearchConfig":
        if self.tau_late > self.tau_early:
            raise ValueError(f"tau_late ({self.tau_late}) must not exceed tau_early ({self.tau_early})")
        return self


class TrainingConfig(_Config):
    """Everything parallel_train.py can be configured with. Paths are repo-relative or None (feature off)."""

    # Paths
    network_path: Optional[str] = Field(
        "neural_nets/example_network.pt",
        description="Weights to start from (scripts/download_weights.sh fetches the default)",
    )
    expert_data_path: Optional[str] = Field(
        None, description="Directory of expert *.pkl data mixed into training batches"
    )
    curriculum_csv_path: Optional[str] = Field(
        None,
        description="Lichess puzzle CSV for the puzzle curriculum; unset starts every game from the start position",
    )
    generated_data_dir: str = Field(
        "neural_nets/generated_data", description="Where self-play data is saved to and reloaded from"
    )

    # Processes
    num_agents: int = Field(4, ge=1, description="Number of self-play agents")
    num_evaluators: int = Field(1, ge=1, description="Number of evaluators")
    num_trainers: int = Field(1, ge=1, description="Number of trainers")

    # Network and optimiser
    seed: int = Field(42, description="Random seed")
    init_lr: float = Field(2e-6, gt=0, description="Initial learning rate")
    num_cnn_repeats: int = Field(32, ge=1, description="Number of repeated ResNet blocks in the policy-value network")

    # Search and training
    sims: int = Field(1000, ge=1, description="MCTS simulations per move")
    maximum_lag: int = Field(8, ge=1, description="Maximum evaluations an agent may have outstanding before it waits")
    evaluator_batch_size: int = Field(256, ge=1, description="Maximum positions per evaluator forward pass")
    trainer_batch_size: int = Field(128, ge=1, description="Positions per training batch")
    weights_update_freq: int = Field(100, ge=1, description="Training steps between weight pushes to the evaluators")

    # MCTS and self-play constants (nested on the command line: search.c_puct=1.5)
    search: SearchConfig = Field(default_factory=SearchConfig)


class SupervisedConfig(_Config):
    """Configuration for supervised_learning.py."""

    data_dir: str = Field(description="Directory containing train*pkl and test*pkl files")
    checkpoint: Optional[str] = Field(None, description="Checkpoint file to resume training from")
    batch_size: int = Field(64, ge=1, description="Positions per training batch")
    init_lr: float = Field(1e-3, gt=0, description="Initial learning rate (OneCycleLR peaks at 5x this)")


class HumanVsMachineConfig(_Config):
    """Configuration for human_vs_machine.py."""

    network_path: str = Field("neural_nets/example_network.pt", description="Weights the engine plays with")
    num_rollouts: int = Field(2000, ge=1, description="MCTS simulations per engine move")
    move_selection: MoveSelection = Field(
        "visits",
        description="How the engine picks its move after searching: 'visits' (most visited) or 'q' (highest value)",
    )
    show_search: bool = Field(
        True, description="Show the engine's search: a side panel of statistics and arrows for the moves it is weighing"
    )


class VsStockfishConfig(_Config):
    """Configuration for chess_engine_vs_zero_style.py."""

    stockfish_path: str = Field(description="Path to the Stockfish binary")
    network_path: str = Field("neural_nets/example_network.pt", description="Weights the network plays with")
    num_rollouts: int = Field(2000, ge=1, description="MCTS simulations per network move")
    move_selection: MoveSelection = Field(
        "visits",
        description="How the engine picks its move after searching: 'visits' (most visited) or 'q' (highest value)",
    )
    show_search: bool = Field(
        True, description="Show the engine's search: a side panel of statistics and arrows for the moves it is weighing"
    )
    stockfish_elo: int = Field(1600, ge=1, description="Elo Stockfish is capped at (UCI_Elo)")
    stockfish_move_time: float = Field(1.0, gt=0, description="Seconds Stockfish thinks per move")
