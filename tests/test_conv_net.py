from pathlib import Path
from typing import List

import chess_moves
import pytest
import torch

from neural_nets.conv_net import ChessNet

INPUT_SIZE = [12, 8, 8]
OUTPUT_SIZE = [70, 8, 8]
WHITE_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
BLACK_FEN = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
WHITE_PROMOTION_FEN = "8/4P3/8/8/8/8/8/k6K w - - 0 1"
BLACK_PROMOTION_FEN = "8/8/8/8/8/8/p7/7K b - - 0 1"
POSITIONS = [
    pytest.param(WHITE_FEN, id="white_to_move"),
    pytest.param(BLACK_FEN, id="black_to_move"),
    pytest.param(WHITE_PROMOTION_FEN, id="white_promotion"),
    pytest.param(BLACK_PROMOTION_FEN, id="black_promotion"),
]


@pytest.fixture
def chess_net(tmp_path: Path) -> ChessNet:
    torch.manual_seed(0)
    return ChessNet(input_size=INPUT_SIZE, output_size=OUTPUT_SIZE, num_repeats=1, save_dir=str(tmp_path))


def _legal_moves(fen: str) -> List[str]:
    engine = chess_moves.ChessEngine()
    engine.set_fen(fen)
    return engine.legal_moves()


@pytest.mark.parametrize("fen", POSITIONS)
def test_illegal_moves_get_no_probability(chess_net: ChessNet, fen: str) -> None:
    chess_net.eval()
    state, legal_move_mask, _ = chess_net.tensorise_inputs([fen], [_legal_moves(fen)])

    with torch.no_grad():
        _, policy = chess_net(state, legal_move_mask, infering=True)

    assert policy[legal_move_mask == 0].max().item() == 0.
    assert policy[legal_move_mask == 1].sum().item() == pytest.approx(1.)


@pytest.mark.parametrize("fen", POSITIONS)
def test_every_legal_move_target_lies_inside_the_legal_move_mask(chess_net: ChessNet, fen: str) -> None:
    moves = _legal_moves(fen)
    uniform_visits = [1. / len(moves)] * len(moves)

    _, target_policy, _, legal_move_mask = chess_net.tensorise_batch([fen], [moves], [uniform_visits], [0.])

    assert (target_policy * (1 - legal_move_mask)).sum().item() == 0.


def test_masked_training_loss_is_finite(chess_net: ChessNet) -> None:
    fens = [WHITE_FEN, BLACK_FEN, WHITE_PROMOTION_FEN, BLACK_PROMOTION_FEN]
    moves = [_legal_moves(fen) for fen in fens]
    visit_distributions = [[1. / len(move_set)] * len(move_set) for move_set in moves]
    state, target_policy, target_value, legal_move_mask = chess_net.tensorise_batch(
        fens, moves, visit_distributions, [0.] * len(fens))

    total_loss, value_loss, policy_loss = chess_net.loss_function(state, target=(target_value, target_policy),
                                                                  legal_move_mask=legal_move_mask)
    total_loss.backward()

    assert all(torch.isfinite(loss).item() for loss in (total_loss, value_loss, policy_loss))
    assert all(torch.isfinite(parameter.grad).all().item() for parameter in chess_net.parameters()
               if parameter.grad is not None)
