from multiprocessing import Process, Queue, Lock
from queue import Empty

from transformers.models.musicgen_melody.modeling_musicgen_melody import MusicgenMelodyOutputWithPast

from tree.memory import Memory
import numpy as np
import chess_moves

from neural_nets.conv_net import ChessNet


class TrainingProcess(Process):
    """ This is meant constantly run the neural network evaluation and training in parallel with MCTS"""

    def __init__(self, neural_net: ChessNet, experience_queue, batch_size: int = 64, nn_load_path: str = None):
        """
        :param queue: queue from the tree search
        :param lock:
        """
        super().__init__()

        self.experience_queue = experience_queue

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

    def train_neural_network(self):
        if len(self.memory) < 32:
            return
        states, moves, probs, wins = self.memory.get_batch(32)
        state, moves, wins, legal_move_mask = self.neural_net.tensorise_batch(states, moves, probs, wins)
        self.neural_net.loss_function(state, target=(wins, moves), legal_move_mask=legal_move_mask)

    def update_node_and_edges(self, state, evaluation):
        pass

    def stop(self):
        self.running = False