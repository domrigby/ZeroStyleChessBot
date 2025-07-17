from multiprocessing import Process, Queue, Lock
from queue import Empty
from zero_style_chess_engine.memory import Memory
import chess_moves
from typing import List
from neural_nets.conv_net import ChessNet
import numpy as np
import time
import os.path
from util.parallel_error_log import error_logger
from datetime import datetime
import torch


class TrainingProcess(Process):
    """ This is meant constantly run the neural network evaluation and training in parallel with MCTS"""
    min_games_to_train = 0

    def __init__(self, save_dir: str, neural_net: ChessNet, experience_queues: List[Queue] = None, weights_queue: Queue = None,
                 batch_size: int = 32, min_num_batches_to_train: int = 128, num_agents: int = 1, data_queue: Queue = None, load_path: str = None,
                 weights_update_freq: int = 100):
        """
        :param queue: queue from the tree search
        :param lock:
        """

        super().__init__()

        self.save_dir = save_dir
        self.log_file_name = os.path.join(save_dir, f"{self.__class__.__name__}.txt")

        #  Set up the queues
        self.experience_queue = experience_queues
        self.data_queue = data_queue
        self.weights_queue = weights_queue

        # Currently going to update it off a training count... in future want to add a seperate validation set of which
        # we only add the best
        self.training_count = 0
        self.weights_update_freq = weights_update_freq

        self.running = True

        self.batch_size = batch_size
        self.min_num_batches_to_train = min_num_batches_to_train

        self.neural_net = neural_net

        self.memory = Memory(1000000, num_agents=num_agents, expert_data_path=load_path)

        self.update_period = 25

        #  Set up loss plotting windows
        self.value_loss_window = np.zeros(self.update_period)
        self.pol_loss_window = np.zeros(self.update_period)
        self.total_loss_window = np.zeros(self.update_period)

        self.last_update_time = time.time()
        self.last_network_save = time.time()

        self.games_played = 0

    def set_experience_queue(self, experience_queue: Queue):
        """
        Func to set the experience queue
        :param experience_queue:
        :return:
        """
        self.experience_queue = experience_queue

    @error_logger
    def run(self):
        """
        Main running loop for the trainer. It waits for incoming data and then begins to train the network once there is
        sufficient data.
        :return:
        """

        # Create cuda stream
        stream = torch.cuda.Stream()

        while True:

            for queue in self.experience_queue:

                for _ in range(self.batch_size):
                    try:
                        game_states, agent_id = queue.get_nowait()
                        self.memory.save_game_to_memory(game_states, agent_id)
                        self.games_played += 1
                        print(f"Games played: {self.games_played}")
                        
                        if self.games_played >= self.min_games_to_train:
                            with torch.cuda.stream(stream):
                                self.train_neural_network()
                    except Empty:
                        break

            time_now = time.time()

            if time_now - self.last_update_time > self.update_period and self.data_queue is not None:

                new_data = {'experience_length': len(self.memory.data),
                            'total_loss': self.total_loss_window.mean(),
                            'value_loss': self.value_loss_window.mean(),
                            'policy_loss': self.pol_loss_window.mean()}

                self.data_queue.put_nowait(new_data)

                self.last_update_time = time_now

                with open(self.save_dir+'/debug.txt', 'w') as f:
                    now = datetime.now()
                    f.write(f'Still alive at time: {now.strftime("%m/%d/%Y, %H:%M:%S")}\n')
                    f.write(f'experience_length: {len(self.memory.data):.5f} total_loss: {self.total_loss_window.mean():.5f}\n'
                            f'value_loss: {self.value_loss_window.mean():.5f} policy_loss: {self.pol_loss_window.mean():.5f}\n'
                            f'Games played: {self.games_played}')

            if time_now - self.last_network_save > 600 and self.data_queue is not None:
                self.neural_net.save_network(f'networks/RL_tuned_{self.training_count}.pt')
                self.last_network_save = time.time()
                self.memory.save_data()

            # Do a weights update every N training steps
            if self.training_count > 0 and self.training_count%self.weights_update_freq == 0:
                self.send_weights_update()

    def train_neural_network(self):
        """ Train the neural network using the experiences from the memory """

        if len(self.memory) < self.min_num_batches_to_train * self.batch_size:
            return

        states, moves, probs, wins = self.memory.get_batch(self.batch_size, 0.5)
        state, moves, wins, legal_move_mask = self.neural_net.tensorise_batch(states, moves, probs, wins)
        total_loss, value_loss, pol_loss = self.neural_net.loss_function(state, target=(wins, moves),
                                                                         legal_move_mask=legal_move_mask)

        self.value_loss_window[self.training_count % self.update_period] = value_loss
        self.pol_loss_window[self.training_count % self.update_period] = pol_loss
        self.total_loss_window[self.training_count % self.update_period] = total_loss

        self.training_count += 1

    def send_weights_update(self):
        weights = {k: v.cpu() for k,v in self.neural_net.state_dict().items()}
        self.weights_queue.put(weights)

    def stop(self):
        self.running = False

if __name__ == "__main__":
    from multiprocessing import Queue

    chess_net = ChessNet(input_size=[12, 8, 8], output_size=[70, 8, 8], num_repeats=32, init_lr=0.001)
    training_data_queue = Queue()

    trainer = TrainingProcess(save_dir="/home/dom/Code/chess_bot/sessions/run_at_20250218_220245",
                              neural_net=chess_net, experience_queues=[Queue()], batch_size=128,
                              num_agents=1, data_queue=Queue(), load_path="/home/dom/Code/chess_bot/new_data.pkl")

    trainer.games_played = 100000

    trainer.run()
