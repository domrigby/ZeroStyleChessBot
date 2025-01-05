from multiprocessing import Process, Queue, Lock

from transformers.models.musicgen_melody.modeling_musicgen_melody import MusicgenMelodyOutputWithPast

from tree.memory import Memory
import numpy as np
import chess_moves

from neural_nets.conv_net import ChessNet

class NeuralNetHandling(Process):
    """ This is meant constantly run the neural network evaluation and training in parallelwith MCTS"""

    def __init__(self, neural_network_class: ChessNet, nn_kwargs, process_queue, results_queue, experience_queue,
                 batch_size: int = 32, nn_load_path: str = None):
        """
        :param queue: queue from the tree search
        :param lock:
        """
        super().__init__()
        self.process_queue = process_queue
        self.results_queue = results_queue
        self.experience_queue = experience_queue
        self.running = True

        self.batch_size = batch_size

        self.neural_network = neural_network_class(**nn_kwargs)

        self.memory = Memory(100000)

    def run(self):
        # If there are states in the qy

        while self.running:

            if not self.process_queue.empty():

                # Process inference
                nodes_to_evaluate = [self.process_queue.get() for _ in range(min(self.process_queue.qsize(), self.batch_size))]

                states = [node.state for node in nodes_to_evaluate]
                legal_moves = [node.legal_move_strings for node in nodes_to_evaluate]

                state_tens, legal_move_mask, legal_move_key = self.neural_network.tensorise_inputs(states, legal_moves)

                # Perform forward pass
                values, policies = self.neural_network(state_tens, legal_move_mask)

                # Results will come in tuples of (node, value, [[edge, prob]])
                for idx, node in enumerate(nodes_to_evaluate):
                    move_probs = []
                    for edge, move_idx in legal_move_key:
                        move_probs.append([edge, policies[idx][move_idx].item()])
                    self.results_queue.put((node, values[idx].item(), move_probs))

            if not self.experience_queue.empty():
                pass

            self.train_neural_network()
                
    def train_neural_network(self):
        pass

    def update_node_and_edges(self, state, evaluation):
        pass


    def stop(self):
        self.running = False