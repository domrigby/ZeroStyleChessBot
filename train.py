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
from tree.parallel_game_tree import GameTree
from util.test_fen_strings import FenTests

if __name__ == '__main__':

    NUM_EVALUATORS = 1
    NUM_AGENTS = 1

    chess_net = ChessNet(input_size=[12, 8, 8], output_size=[70, 8, 8], num_repeats=16)

    # Create the queues
    process_queue = Queue()
    experience_queue = Queue()
    results_queue = Queue()

    evaluators: List[NeuralNetHandling] = []

    # Set the neural network parameters to be shared
    chess_net.share_memory()

    trainer = TrainingProcess("/home/dom/Code/chess_bot/sessions/run_at_20250225_191610", neural_net=chess_net,
                              experience_queues=[experience_queue], batch_size=128)

    tree = GameTree("/home/dom/Code/chess_bot/sessions/run_at_20250225_191610", training=True, multiprocess=True,
                    experience_queue=experience_queue, process_queue=process_queue, results_queue=results_queue)

    for _ in range(NUM_EVALUATORS):
        evaluators.append(NeuralNetHandling(neural_net=chess_net, process_queues=[process_queue],
                                            results_queue_dict={tree.agent_id: results_queue}, batch_size=128))

    [evaluator.start() for evaluator in evaluators]
    trainer.start()

    tree.train()

