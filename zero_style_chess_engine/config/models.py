"""
Typed, validated configuration models. Defaults live here; the YAML files in conf/ only hold Hydra settings and
deliberate overrides.
"""

from typing import Annotated, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator


# How the engine picks a move after searching. Most visited is the robust choice: a high Q on a few visits is mostly noise
MoveSelection = Literal["visits", "q"]

# How much of the search to show. "play" draws no arrows and, while the engine ponders on your move, hides the moves it
# is weighing for you, so it is no hint; "analysis" shows everything
DisplayMode = Literal["play", "analysis"]

# Value function of the baseline MCTS player in place of the network's: "zero" (only game ends score), "material"
# (material balance, P=1 N=B=3 R=5 Q=9) or "pesto" (PeSTO's piece-square evaluation once the captures settle)
HeuristicValue = Literal["zero", "material", "pesto"]

# What a node passes up the tree after a rollout: "mean" (that rollout's value, so a node's value is the average over its
# rollouts, as in AlphaZero) or "max" (its best move's Q, minimax-style: optimistic for the side to move)
Backup = Literal["mean", "max"]


class _Config(BaseModel):
    # extra="forbid": a misspelt key (search.cpuct=1) fails at startup instead of being ignored
    model_config = ConfigDict(extra="forbid", frozen=True)


class SearchConfig(_Config):
    """MCTS and self-play constants. Travels inside each pickled agent, like the rest of TrainingConfig."""

    c_puct: float = Field(3.0, gt=0, description="PUCT exploration constant")
    gamma: float = Field(
        0.99, gt=0, le=1, description="Discount applied per ply when backing values up the search tree"
    )
    backup: Backup = Field(
        "mean",
        description="Value each node passes to its parent: 'mean' (the rollout's own value, as in AlphaZero) or 'max' "
        "(its best move's Q)",
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
    ponder: bool = Field(True, description="Keep searching while waiting for your move, so the engine plays stronger")
    display_mode: DisplayMode = Field(
        "play",
        description="'play': no arrows, and no candidate moves while pondering on yours; 'analysis': show everything",
    )
    ponder_max_rollouts: int = Field(
        20000, ge=1, description="Most rollouts spent pondering one position; bounds the tree's memory while you think"
    )


class ZeroStylePlayerConfig(_Config):
    """A side played by the zero-style network and MCTS."""

    type: Literal["zero_style"] = "zero_style"
    network_path: str = Field("neural_nets/example_network.pt", description="Weights this side plays with")
    num_rollouts: int = Field(2000, ge=1, description="MCTS simulations per move")
    move_selection: MoveSelection = Field(
        "visits",
        description="How the engine picks its move after searching: 'visits' (most visited) or 'q' (highest value)",
    )


class HeuristicMctsPlayerConfig(_Config):
    """A baseline side: the same MCTS with uniform priors and a hand-written value function instead of the network."""

    type: Literal["heuristic_mcts"] = "heuristic_mcts"
    value_function: HeuristicValue = Field(
        "material",
        description="'material' (material balance), 'pesto' (piece-square evaluation once the captures settle) or "
        "'zero' (only checkmates and draws score)",
    )
    num_rollouts: int = Field(2000, ge=1, description="MCTS simulations per move")
    c_puct: float = Field(
        3.0, gt=0, description="PUCT exploration constant; the default is the network's (search.c_puct in training)"
    )
    move_selection: MoveSelection = Field(
        "visits",
        description="How the engine picks its move after searching: 'visits' (most visited) or 'q' (highest value)",
    )


class UciPlayerConfig(_Config):
    """A side played by a UCI engine such as Stockfish or Lc0."""

    type: Literal["uci"] = "uci"
    engine_path: str = Field(description="Path to the engine binary, or its name if it is on $PATH")
    elo: Optional[int] = Field(
        1600, ge=1, description="Elo the engine is capped at (UCI_Elo); null plays at full strength"
    )
    move_time: float = Field(1.0, gt=0, description="Seconds the engine thinks per move")


class RandomPlayerConfig(_Config):
    """A floor baseline: a uniformly random legal move every turn."""

    type: Literal["random"] = "random"


# One side of an engine-vs-engine game, picked on the command line with player@white=uci / player@black=zero_style
PlayerConfig = Annotated[
    Union[ZeroStylePlayerConfig, HeuristicMctsPlayerConfig, UciPlayerConfig, RandomPlayerConfig],
    Field(discriminator="type"),
]


class EngineVsEngineConfig(_Config):
    """Configuration for engine_vs_engine.py."""

    white: PlayerConfig = Field(description="The player with the white pieces")
    black: PlayerConfig = Field(description="The player with the black pieces")
    show_search: bool = Field(
        True,
        description="Show an MCTS side's search: a side panel of statistics and arrows for the moves it is weighing",
    )
    move_time: Optional[float] = Field(
        None,
        gt=0,
        description="Seconds every side thinks per move, so the sides get equal time. Replaces the MCTS sides' "
        "num_rollouts and the UCI sides' move_time; null keeps each side's own limit",
    )
