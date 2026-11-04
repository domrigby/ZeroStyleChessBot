from pathlib import Path

import chess
import pytest
import torch

from neural_nets.conv_net import ChessNet
from zero_style_chess_engine.engine_interface import EngineInterface

pytestmark = pytest.mark.integration

# EngineInterface builds the full-size network, so the weights file must match it
ENGINE_NUM_REPEATS = 32
CLOSE_TIMEOUT_SECONDS = 10.0
ENGINE_ROLLOUTS = 16
USER_MOVE = "e2e4"


def test_close_stops_evaluator_processes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression: human_vs_machine.py hung after its window closed because the evaluator kept running."""
    monkeypatch.chdir(tmp_path)  # Keep anything the engine writes to the cwd out of the repo
    weights_path = tmp_path / "network.pt"
    network = ChessNet(input_size=[12, 8, 8], output_size=[70, 8, 8], num_repeats=ENGINE_NUM_REPEATS)
    torch.save(network.state_dict(), weights_path)

    engine = EngineInterface(network_path=str(weights_path))
    assert engine.evaluator is not None
    assert engine.evaluator.is_alive()

    engine.close(timeout=CLOSE_TIMEOUT_SECONDS)

    assert not engine.evaluator.is_alive()
    assert engine.evaluator.exitcode == 0


def test_engine_replies_with_a_legal_move(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    weights_path = tmp_path / "network.pt"
    network = ChessNet(input_size=[12, 8, 8], output_size=[70, 8, 8], num_repeats=ENGINE_NUM_REPEATS)
    torch.save(network.state_dict(), weights_path)

    engine = EngineInterface(network_path=str(weights_path), num_rollouts=ENGINE_ROLLOUTS)
    try:
        assert engine.apply_user_move(USER_MOVE)
        board_before_reply = engine.current_board.copy()
        move, fen = engine.get_engine_move()
    finally:
        engine.close(timeout=CLOSE_TIMEOUT_SECONDS)

    assert chess.Move.from_uci(move) in board_before_reply.legal_moves
    assert fen == engine.current_board.fen()
