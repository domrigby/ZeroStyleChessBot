from multiprocessing import Process, Queue, Lock
from queue import Empty

from transformers.models.musicgen_melody.modeling_musicgen_melody import MusicgenMelodyOutputWithPast

from tree.memory import Memory
import numpy as np
import chess_moves

from neural_nets.conv_net import ChessNet

class NeuralNetHandling(Process):
    """ This is meant constantly run the neural network evaluation and training in parallelwith MCTS"""

    def __init__(self, neural_net: ChessNet, process_queue, results_queue,
                 batch_size: int = 64, nn_load_path: str = None):
        """
        :param queue: queue from the tree search
        :param lock:
        """
        super().__init__()
        self.process_queue = process_queue
        self.results_queue = results_queue

        self.running = True

        self.batch_size = batch_size

        self.neural_net = neural_net


    def run(self):
        # If there are states in the qy

        while self.running:

            if not self.process_queue.empty():
                hashes = []
                states = []
                legal_moves = []

                for _ in range(min(self.process_queue.qsize(), self.batch_size)):
                    try:
                        the_hash, state, legal_moves_strings= self.process_queue.get_nowait()
                        hashes.append(the_hash)
                        states.append(state)
                        legal_moves.append(legal_moves_strings)
                    except Empty:
                        break

                # Check that items have actually been received
                if len(hashes) > 0:

                    # Create the input tensors
                    state_tens, legal_move_mask, legal_move_key = self.neural_net.tensorise_inputs(states, legal_moves)

                    # Perform forward pass
                    self.neural_net.eval()
                    values, policies = self.neural_net(state_tens, legal_move_mask)
                    self.neural_net.train()

                    # Results will come in tuples of (node, value, [[edge, prob]])
                    for idx, the_hash in enumerate(hashes):

                        move_probs = []
                        for edge, move_idx in legal_move_key[idx]:
                            move_probs.append([edge, policies[idx][move_idx].item()])

                        self.results_queue.put((the_hash, values[idx].item(), move_probs, legal_move_key[idx]))

                
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