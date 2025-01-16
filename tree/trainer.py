from multiprocessing import Process, Queue, Lock
from queue import Empty

from transformers.models.musicgen_melody.modeling_musicgen_melody import MusicgenMelodyOutputWithPast

from tree.memory import Memory
import numpy as np
import chess_moves

import torch

from neural_nets.conv_net import ChessNet


class TrainingProcess(Process):
    """ This is meant constantly run the neural network evaluation and training in parallel with MCTS"""

    def __init__(self, neural_net: ChessNet, experience_queue, nn_queue, batch_size: int = 64):
        """
        :param queue: queue from the tree search
        :param lock:
        """
        super().__init__()

        self.experience_queue = experience_queue

        self.nn_queue = nn_queue

        # Currently going to update it off a training count... in future want to add a seperate validation set of which
        # we only add the best
        self.training_count = 0

        self.running = True

        self.batch_size = batch_size

        self.neural_net = neural_net

        self.memory = Memory(100000)

    def run(self):

        # If there are states in the qy

        while self.running:

            for _ in range(self.batch_size):
                try:
                    state, moves, visit_counts, predicted_value, is_root_node = self.experience_queue.get_nowait()
                    self.memory.save_state_to_moves(state, moves, visit_counts, predicted_value, is_root_node)
                except Empty:
                    break

            self.train_neural_network()

            if len(self.memory.data) % 1000 == 0:
                self.memory.save_data()

    def train_neural_network(self):
        """ Train the neural network using the experiences from the memory """
        if len(self.memory) < 32:
            return

        states, moves, probs, wins = self.memory.get_batch(32)
        state, moves, wins, legal_move_mask = self.neural_net.tensorise_batch(states, moves, probs, wins)
        self.neural_net.loss_function(state, target=(wins, moves), legal_move_mask=legal_move_mask)

        self.training_count += 1

        # if self.training_count % 1000 == 0:
        #     # Save the netowrk and put it in the queue to be loaded by the evaluator
        #     self.neural_net.save_network(f"networks/network_{self.training_count}.pt")
        #     self.nn_queue.put(self.neural_net.state_dict())

    def update_node_and_edges(self, state, evaluation):
        pass

    def stop(self):
        self.running = False