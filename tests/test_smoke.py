from pathlib import Path

import chess_moves
import torch

from neural_nets.conv_net import ChessNet

START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
INPUT_SIZE = [12, 8, 8]
OUTPUT_SIZE = [70, 8, 8]


def test_engine_start_position_has_twenty_legal_moves() -> None:
    expected_legal_moves = 20
    engine = chess_moves.ChessEngine()
    engine.set_fen(START_FEN)

    assert len(engine.legal_moves()) == expected_legal_moves


def test_chess_net_forward_pass_on_start_position(tmp_path: Path) -> None:
    batch_size = 1
    chess_net = ChessNet(input_size=INPUT_SIZE, output_size=OUTPUT_SIZE, num_repeats=1, save_dir=str(tmp_path))
    chess_net.eval()

    board_tensor = torch.tensor(chess_moves.ChessEngine().fen_to_tensor(START_FEN), dtype=torch.float32).unsqueeze(0)
    assert list(board_tensor.shape) == [batch_size, *INPUT_SIZE]

    with torch.no_grad():
        value, policy = chess_net(board_tensor)

    assert list(value.shape) == [batch_size, 1]
    assert list(policy.shape) == [batch_size, *OUTPUT_SIZE]
