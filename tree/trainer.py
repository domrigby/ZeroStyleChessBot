from multiprocessing import Process, Queue, Lock
from queue import Empty
from tree.memory import Memory
import chess_moves
from typing import List
from neural_nets.conv_net import ChessNet
import numpy as np
import time
import os.path
from util.parallel_error_log import error_logger



class TrainingProcess(Process):
    """ This is meant constantly run the neural network evaluation and training in parallel with MCTS"""
    min_games_to_train = 100

    def __init__(self, save_dir: str, neural_net: ChessNet, experience_queues: List[Queue] = None, batch_size: int = 64,
                 min_num_batches_to_train: int = 128, num_agents: int = 1, data_queue: Queue = None):
        """
        :param queue: queue from the tree search
        :param lock:
        """
        super().__init__()

        self.log_file_name = os.path.join(save_dir, f"{self.__class__.__name__}.txt")

        self.experience_queue = experience_queues
        self.data_queue = data_queue

        # Currently going to update it off a training count... in future want to add a seperate validation set of which
        # we only add the best
        self.training_count = 0

        self.running = True

        self.batch_size = batch_size
        self.min_num_batches_to_train = min_num_batches_to_train

        self.neural_net = neural_net

        self.memory = Memory(100000, num_agents=num_agents)

        self.update_period = 25

        self.value_loss_window = np.zeros(self.update_period)
        self.pol_loss_window = np.zeros(self.update_period)
        self.total_loss_window = np.zeros(self.update_period)

        self.last_update_time = time.time()

        self.games_played = 0

    def set_experience_queue(self, experience_queue: Queue):
        self.experience_queue = experience_queue

    @error_logger
    def run(self):

        # If there are states in the qy

        while self.running:

            for queue in self.experience_queue:

                for _ in range(self.batch_size):
                    try:
                        game_states, agent_id = queue.get_nowait()
                        self.memory.save_game_to_memory(game_states, agent_id)
                        self.games_played += 1
                        print(f"Games played: {self.memory.games_played}")
                    except Empty:
                        break

            # Ensure we have enough diverse enough data to train
            if self.memory.games_played >= self.min_games_to_train:
                self.train_neural_network()

            if len(self.memory.data) % 1000 == 0 and len(self.memory.data) > 0:
                self.memory.save_data()

            if self.training_count % 1000 == 0:
                self.neural_net.save_network(f'networks/RL_tuned_{self.training_count}.pt')

            time_now = time.time()

            if time_now - self.last_update_time > self.update_period and self.data_queue is not None:
                new_data = {'experience_length': len(self.memory.data)}
                self.data_queue.put_nowait(new_data)
                self.last_update_time = time_now

    def train_neural_network(self):
        """ Train the neural network using the experiences from the memory """

        if len(self.memory) < self.min_num_batches_to_train * self.batch_size:
            return

        states, moves, probs, wins = self.memory.get_batch(self.batch_size)
        state, moves, wins, legal_move_mask = self.neural_net.tensorise_batch(states, moves, probs, wins)
        total_loss, value_loss, pol_loss = self.neural_net.loss_function(state, target=(wins, moves),
                                                                         legal_move_mask=legal_move_mask)

        self.value_loss_window[self.training_count // self.update_period] = value_loss
        self.pol_loss_window[self.training_count // self.update_period] = pol_loss
        self.total_loss_window[self.training_count // self.update_period] = total_loss

        if (self.training_count - 1) % self.update_period == 0:

            # Send data back to the main core
            data_dict = {'total_loss': self.total_loss_window.mean(),
                         'value_loss': self.value_loss_window.mean(),
                         'policy_loss': self.pol_loss_window.mean()}

            self.data_queue.put_nowait(data_dict)

            self.value_loss_window = np.zeros(self.update_period)
            self.pol_loss_window = np.zeros(self.update_period)
            self.total_loss_window = np.zeros(self.update_period)

        self.training_count += 1

    def update_node_and_edges(self, state, evaluation):
        pass

    def stop(self):
        self.running = False