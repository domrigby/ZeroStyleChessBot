import multiprocessing as mp
if __name__ == "__main__":
    try:
       mp.set_start_method('spawn', force=True)
       print("spawned")
    except RuntimeError:
        print("Failed to spawn")
        pass

import time
from multiprocessing import Queue
from typing import List, Dict

import chess.svg
import chess_moves
import torch.multiprocessing as torch_mp

from neural_nets.conv_net import ChessNet
from tree.evaluator import NeuralNetHandling
from tree.trainer import TrainingProcess
from tree.tree import GameTree

if __name__ == '__main__':

    NUM_EVALUATORS = 1
    NUM_AGENTS = 1

    chess_net = ChessNet(input_size=[12, 8, 8], output_size=[70, 8, 8], num_repeats=16)

    # Create the queues
    process_queue = Queue()
    experience_queue = Queue()

    # Create the multiprocess queue
    nn_update = torch_mp.Queue()

    evaluators: List[NeuralNetHandling] = []

    # Set the neural network parameters to be shared
    chess_net.share_memory()

    for _ in range(NUM_EVALUATORS):
        evaluators.append(NeuralNetHandling(neural_net=chess_net, batch_size=128))

    trainer = TrainingProcess(neural_net=chess_net, experience_queue=experience_queue, batch_size=128)

    tree = GameTree(chess_moves.ChessEngine, num_threads=1, training=True, multiprocess=True, evaluator=evaluators[0],
                    experience_queue=experience_queue)

    [evaluator.start() for evaluator in evaluators]
    trainer.start()

    sims = 1000
    max_length = 400

    tree.train()

