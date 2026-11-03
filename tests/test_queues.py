from pathlib import Path

import pytest
import torch

from neural_nets.conv_net import ChessNet
from util.queues import create_agents


def test_evaluators_do_not_share_parameters_with_the_trainer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    num_agents = 2
    num_evaluators = 2
    num_trainers = 1
    # Memory loads cwd-relative data
    monkeypatch.chdir(tmp_path)
    chess_net = ChessNet(input_size=[12, 8, 8], output_size=[70, 8, 8], num_repeats=1, save_dir=str(tmp_path))

    _, evaluators, trainers, _, _ = create_agents(str(tmp_path), num_agents, num_evaluators, num_trainers, chess_net)

    networks = [trainers[0].neural_net, *(evaluator.neural_net for evaluator in evaluators)]
    parameter_addresses = [{tensor.data_ptr() for tensor in network.state_dict().values()} for network in networks]
    for idx, addresses in enumerate(parameter_addresses):
        for other in parameter_addresses[idx + 1 :]:
            assert addresses.isdisjoint(other)

    for evaluator in evaluators:
        for name, tensor in evaluator.neural_net.state_dict().items():
            assert torch.equal(tensor, chess_net.state_dict()[name])
