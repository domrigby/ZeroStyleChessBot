from pathlib import Path

import pytest
import torch

from neural_nets.conv_net import ChessNet
from zero_style_chess_engine.engine_interface import EngineInterface

pytestmark = pytest.mark.integration

# EngineInterface builds the full-size network, so the weights file must match it
ENGINE_NUM_REPEATS = 32
CLOSE_TIMEOUT_SECONDS = 10.0


def test_close_stops_evaluator_processes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression: human_vs_machine.py hung after its window closed because the evaluator kept running."""
    monkeypatch.chdir(tmp_path)  # ChessNet creates its save_dir in the cwd
    weights_path = tmp_path / "network.pt"
    network = ChessNet(
        input_size=[12, 8, 8], output_size=[70, 8, 8], num_repeats=ENGINE_NUM_REPEATS, save_dir=str(tmp_path)
    )
    torch.save(network.state_dict(), weights_path)

    engine = EngineInterface(network_path=str(weights_path), parallel=False)
    assert all(evaluator.is_alive() for evaluator in engine.evaluators)

    engine.close(timeout=CLOSE_TIMEOUT_SECONDS)

    assert not any(evaluator.is_alive() for evaluator in engine.evaluators)
    assert all(evaluator.exitcode == 0 for evaluator in engine.evaluators)
