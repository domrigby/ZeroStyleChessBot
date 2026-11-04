"""Configuration: Pydantic models (models.py) and the Hydra glue that fills them (hydra.py)."""

from zero_style_chess_engine.config.hydra import check_paths_exist, from_hydra, register_defaults
from zero_style_chess_engine.config.models import (
    DisplayMode,
    EngineVsEngineConfig,
    HeuristicMctsPlayerConfig,
    HeuristicValue,
    HumanVsMachineConfig,
    MoveSelection,
    PlayerConfig,
    SearchConfig,
    SupervisedConfig,
    TrainingConfig,
    UciPlayerConfig,
    ZeroStylePlayerConfig,
)

__all__ = [
    "DisplayMode",
    "EngineVsEngineConfig",
    "HeuristicMctsPlayerConfig",
    "HeuristicValue",
    "HumanVsMachineConfig",
    "MoveSelection",
    "PlayerConfig",
    "SearchConfig",
    "SupervisedConfig",
    "TrainingConfig",
    "UciPlayerConfig",
    "ZeroStylePlayerConfig",
    "check_paths_exist",
    "from_hydra",
    "register_defaults",
]
