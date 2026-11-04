import queue
from typing import Any

import chess
import numpy as np
import pytest

from zero_style_chess_engine.config import HeuristicMctsPlayerConfig
from zero_style_chess_engine.heuristic_evaluator import HeuristicEvaluator, material_value
from zero_style_chess_engine.messages import EvalRequest
from zero_style_chess_engine.players import make_player

# White to move, an undefended black queen on d5 in the rook's line
FREE_QUEEN_FEN = "4k3/8/8/3q4/8/8/8/3RK3 w - - 0 1"
# White to move mates with Ra8 (back rank)
BACK_RANK_MATE_FEN = "6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1"


def test_material_value_is_from_the_side_to_moves_point_of_view() -> None:
    white_to_move = chess.Board(FREE_QUEEN_FEN)  # Black is a queen (9) up against a rook (5)
    black_to_move = white_to_move.copy()
    black_to_move.turn = chess.BLACK

    assert material_value(white_to_move) < 0
    assert material_value(black_to_move) == pytest.approx(-material_value(white_to_move))
    assert material_value(chess.Board()) == 0.0
    assert -1 < material_value(chess.Board("4k3/8/8/8/8/8/QQQQQQQQ/4K3 w - - 0 1")) < 1


def test_heuristic_evaluator_answers_with_uniform_priors_and_the_value_function() -> None:
    results: "queue.Queue[Any]" = queue.Queue()
    board = chess.Board(FREE_QUEEN_FEN)
    moves = [move.uci() for move in board.legal_moves]

    HeuristicEvaluator(results, "material").put(EvalRequest(0, 7, FREE_QUEEN_FEN, moves))

    [result] = results.get_nowait()
    assert result.node_id == 7
    assert result.value == pytest.approx(material_value(board))
    np.testing.assert_allclose(result.priors, np.full(len(moves), 1 / len(moves)))


@pytest.mark.parametrize(
    "value_function, fen, expected_move",
    [
        pytest.param("material", FREE_QUEEN_FEN, "d1d5", id="material_takes_a_free_queen"),
        pytest.param("zero", BACK_RANK_MATE_FEN, "a1a8", id="zero_finds_mate_in_one"),
    ],
)
def test_heuristic_mcts_player_finds_the_obvious_move(value_function: str, fen: str, expected_move: str) -> None:
    config = HeuristicMctsPlayerConfig.model_validate({"value_function": value_function, "num_rollouts": 400})
    player = make_player(config, "black")
    try:
        move = player.choose_move(chess.Board(fen))
    finally:
        player.close()  # No evaluator process: returns at once

    assert move.uci() == expected_move
