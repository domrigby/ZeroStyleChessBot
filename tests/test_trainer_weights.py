import queue
from pathlib import Path
from typing import List

import chess_moves
import pytest
import torch

from neural_nets.conv_net import ChessNet
from zero_style_chess_engine.memory import DataPoint
from zero_style_chess_engine.trainer import TrainingProcess

START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
BATCH_SIZE = 4


@pytest.fixture
def make_trainer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A TrainingProcess driven in-process, with thread queues standing in for the evaluators' weights queues."""
    monkeypatch.chdir(tmp_path)

    def make(weights_queues: List[queue.Queue], weights_update_freq: int) -> TrainingProcess:
        chess_net = ChessNet(input_size=[12, 8, 8], output_size=[70, 8, 8], num_repeats=1, save_dir=str(tmp_path))
        trainer = TrainingProcess(save_dir=str(tmp_path), neural_net=chess_net, weights_queues=weights_queues,
                                  batch_size=BATCH_SIZE, minibatch_size=BATCH_SIZE, min_num_batches_to_train=1,
                                  weights_update_freq=weights_update_freq)
        engine = chess_moves.ChessEngine()
        engine.set_fen(START_FEN)
        moves = engine.legal_moves()
        for _ in range(BATCH_SIZE):
            trainer.memory.add_data_point(DataPoint(START_FEN, moves, [1. / len(moves)] * len(moves)))
        return trainer

    return make


def _drain(weights_queue: queue.Queue) -> list:
    received = []
    while True:
        try:
            received.append(weights_queue.get_nowait())
        except queue.Empty:
            return received


@pytest.mark.parametrize("steps, weights_update_freq", [
    pytest.param(7, 3, id="not_a_multiple"),
    pytest.param(6, 3, id="exact_multiple"),
    pytest.param(2, 3, id="fewer_steps_than_freq"),
])
def test_every_evaluator_receives_exactly_one_update_per_freq_steps(make_trainer, steps: int,
                                                                    weights_update_freq: int) -> None:
    num_evaluators = 2
    weights_queues = [queue.Queue(maxsize=1) for _ in range(num_evaluators)]
    trainer = make_trainer(weights_queues, weights_update_freq)
    received_counts = [0] * num_evaluators

    for _ in range(steps):
        trainer.train_neural_network()
        for idx, weights_queue in enumerate(weights_queues):
            received_counts[idx] += len(_drain(weights_queue))

    assert received_counts == [steps // weights_update_freq] * num_evaluators


def test_undelivered_update_is_replaced_by_the_newest(make_trainer) -> None:
    weights_update_freq = 1
    steps = 3
    weights_queue = queue.Queue(maxsize=1)
    trainer = make_trainer([weights_queue], weights_update_freq)

    for _ in range(steps):
        trainer.train_neural_network()

    delivered = _drain(weights_queue)
    assert len(delivered) == 1
    for name, tensor in trainer.neural_net.state_dict().items():
        assert torch.equal(delivered[0][name], tensor.cpu())
