from multiprocessing import Process, Queue, Lock
from queue import Empty

from tree.memory import Memory
import chess_moves

from typing import List

from neural_nets.conv_net import ChessNet


class TrainingProcess(Process):
    """ This is meant constantly run the neural network evaluation and training in parallel with MCTS"""

    def __init__(self, neural_net: ChessNet, experience_queues: List[Queue] = None, batch_size: int = 64, min_num_batches_to_train: int = 32):
        """
        :param queue: queue from the tree search
        :param lock:
        """
        super().__init__()

        self.experience_queue = experience_queues

        # Currently going to update it off a training count... in future want to add a seperate validation set of which
        # we only add the best
        self.training_count = 0

        self.running = True

        self.batch_size = batch_size
        self.min_num_batches_to_train = min_num_batches_to_train

        self.neural_net = neural_net

        self.memory = Memory(100000)

    def set_experience_queue(self, experience_queue: Queue):
        self.experience_queue = experience_queue

    def run(self):

        # If there are states in the qy

        while self.running:

            for queue in self.experience_queue:

                for _ in range(self.batch_size):
                    try:
                        state, moves, visit_counts, predicted_value, is_root_node = queue.get_nowait()
                        self.memory.save_state_to_moves(state, moves, visit_counts, predicted_value, is_root_node)
                    except Empty:
                        break

            self.train_neural_network()

            if len(self.memory.data) % 10000 == 0 and len(self.memory.data) > 0:
                self.memory.save_data()

            if self.training_count % 10000 == 0:
                self.neural_net.save_network(f'networks/RL_tuned_{self.training_count}.pt')

    def train_neural_network(self):
        """ Train the neural network using the experiences from the memory """
        if len(self.memory) < self.min_num_batches_to_train * self.batch_size:
            return

        states, moves, probs, wins = self.memory.get_batch(32)
        state, moves, wins, legal_move_mask = self.neural_net.tensorise_batch(states, moves, probs, wins)
        self.neural_net.loss_function(state, target=(wins, moves), legal_move_mask=legal_move_mask)

        self.training_count += 1

    def update_node_and_edges(self, state, evaluation):
        pass

    def stop(self):
        self.running = False