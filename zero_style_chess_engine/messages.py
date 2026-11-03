"""Messages sent between the pipeline processes. Each queue carries exactly one of these types."""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass(frozen=True)
class EvalRequest:
    """Agent → evaluator: a position to evaluate. node_id lets the agent find the node again."""

    agent_id: int
    node_id: int
    fen: str
    legal_moves: List[str]


@dataclass(frozen=True)
class EvalResult:
    """Evaluator → agent (sent as a List[EvalResult] batch). priors is aligned with the request's legal_moves."""

    node_id: int
    value: float
    priors: np.ndarray


@dataclass(frozen=True)
class PositionRecord:
    """One position of a game, with the search's visit counts as the policy target and the outcome as the value"""

    fen: str
    moves: List[str]
    visit_counts: np.ndarray
    value: float


@dataclass(frozen=True)
class GameRecord:
    """Agent → trainer: positions from a played game, or from a line the search found"""

    agent_id: int
    positions: List[PositionRecord]


@dataclass(frozen=True)
class GameStats:
    """Agent → supervisor, once per finished game. white_win is None for a draw"""

    white_win: Optional[bool]
    game_length: int


@dataclass(frozen=True)
class TrainerStats:
    """Trainer → supervisor, every update_period seconds. Losses are means over the recent training steps"""

    experience_length: int
    total_loss: float
    value_loss: float
    policy_loss: float
