"""
A stand-in for the network evaluator, for a baseline MCTS player: uniform priors and a hand-written value function.
It answers in the searching thread, so there is no evaluator process and no network.
"""

import math
import queue
from typing import Any, Callable, Dict

import chess
import numpy as np

from zero_style_chess_engine.config import HeuristicValue
from zero_style_chess_engine.messages import EvalRequest, EvalResult

PIECE_VALUES = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}
# Material lead mapped to a value in (-1, 1) as tanh(lead / MATERIAL_SCALE): a pawn up is ~0.1, a queen up ~0.7
MATERIAL_SCALE = 10.0


def material_value(board: chess.Board) -> float:
    """Material balance from the point of view of the side to move, squashed into (-1, 1)"""
    lead = sum(
        value * (len(board.pieces(piece, board.turn)) - len(board.pieces(piece, not board.turn)))
        for piece, value in PIECE_VALUES.items()
    )
    return math.tanh(lead / MATERIAL_SCALE)


# Values are from the side to move's point of view, as the network's are
VALUE_FUNCTIONS: Dict[HeuristicValue, Callable[[chess.Board], float]] = {
    "zero": lambda board: 0.0,
    "material": material_value,
}


class HeuristicEvaluator:
    """
    Stands in for the evaluator's request queue (MCTS only calls put on it): answers every request at once, putting
    the result on the search's results queue, where the search picks it up on its next poll
    """

    def __init__(self, results_queue: "queue.Queue[Any]", value_function: HeuristicValue) -> None:
        self.results_queue = results_queue
        self.value_function = VALUE_FUNCTIONS[value_function]

    def put(self, request: EvalRequest) -> None:
        num_moves = len(request.legal_moves)
        uniform_priors = np.full(num_moves, 1.0 / num_moves) if num_moves else np.zeros(0)
        value = self.value_function(chess.Board(request.fen))
        self.results_queue.put([EvalResult(request.node_id, value, uniform_priors)])
