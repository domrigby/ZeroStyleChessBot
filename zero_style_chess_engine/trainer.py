from multiprocessing import Process, Queue, Event
from queue import Empty, Full
from zero_style_chess_engine.memory import Memory
from util.parallel_error_log import error_logger
from neural_nets.conv_net import ChessNet
from neural_nets.generic_net import GenericNet

import chess_moves
import contextlib
from typing import List
import numpy as np
import time
import os.path
from datetime import datetime
import torch


class TrainingProcess(Process):
    """ This is meant constantly run the neural network evaluation and training in parallel with MCTS"""
    min_games_to_train = 0
    idle_sleep_seconds = 0.01

    def __init__(self, save_dir: str, neural_net: ChessNet, experience_queues: List[Queue] = None, weights_queues: List[Queue] = None,
                 batch_size: int = 1024, minibatch_size: int = 32, min_num_batches_to_train: int = 128, num_agents: int = 1, data_queue: Queue = None, load_path: str = None,
                 weights_update_freq: int = 100, expert_ratio: float = 0.9, expert_ratio_decay_rate: float = 0.995, update_period: int = 25,
                 stop_event: Event = None):
        """
        :param queue: queue from the tree search
        :param lock:
        """

        super().__init__()

        # Saving directories
        self.save_dir = save_dir
        self.log_file_name = os.path.join(save_dir, f"{self.__class__.__name__}.txt")

        #  Set up the queues
        self.experience_queue = experience_queues
        self.data_queue = data_queue
        self.weights_queues: List[Queue] = weights_queues if weights_queues is not None else []

        self.stop_event = stop_event if stop_event is not None else Event()

        #  Set hyperparameters
        self.weights_update_freq: int = weights_update_freq
        self.batch_size: int = batch_size
        self.minibatch_size: int = minibatch_size
        self._batch_to_minibatch_ratio = self.batch_size // self.minibatch_size
        self.min_num_batches_to_train: int  = min_num_batches_to_train
        self.update_period: int = update_period
        self.expert_ratio: float = expert_ratio
        self.expert_ratio_decay_rate: float = expert_ratio_decay_rate

        # Save the neural network
        self.neural_net: GenericNet = neural_net

        # Initialise the memory... this handles all the experiences
        self.memory = Memory(1000000, num_agents=num_agents, expert_data_path=load_path)

        #  Set up loss plotting windows
        self.value_loss_window = np.zeros(self.update_period)
        self.pol_loss_window = np.zeros(self.update_period)
        self.total_loss_window = np.zeros(self.update_period)

        # Initialise data storing vairables
        self.last_update_time: float = time.time()
        self.last_network_save: float = time.time()
        self.games_played: int = 0
        self.training_count: int = 0

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

        stream_context = torch.cuda.stream(torch.cuda.Stream()) if torch.cuda.is_available() \
            else contextlib.nullcontext()

        while not self.stop_event.is_set():

            if not self.collect_experience(stream_context):
                time.sleep(self.idle_sleep_seconds)

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
                self.neural_net.save_network(f'{self.save_dir}/RL_tuned_{self.training_count}.pt')
                self.last_network_save = time.time()
                self.memory.save_data()

        # Final checkpoint so a clean shutdown never loses training progress
        self.neural_net.save_network(f'{self.save_dir}/RL_tuned_{self.training_count}.pt')
        self.memory.save_data()

        # Don't block exit flushing queues whose readers have already stopped
        for queue in self.weights_queues + [self.data_queue]:
            if queue is not None:
                queue.cancel_join_thread()

    def collect_experience(self, stream_context) -> bool:
        """
        Move finished games from the agents into memory, doing a training step per game.
        :return: whether any game was received
        """
        received = False
        for queue in self.experience_queue:

            for _ in range(self.batch_size):
                try:
                    game_states, agent_id = queue.get_nowait()
                except Empty:
                    break

                received = True
                self.memory.save_game_to_memory(game_states, agent_id)
                self.games_played += 1
                print(f"Games played: {self.games_played}")

                # New game recieved... lets do a training batch
                if self.games_played >= self.min_games_to_train:
                    with stream_context:
                        self.train_neural_network()
        return received

    def train_neural_network(self):
        """ Train the neural network using the experiences from the memory """

        if len(self.memory) < self.min_num_batches_to_train * self.batch_size:
            return

        # Decay the expert ratio
        self.expert_ratio *= self.expert_ratio_decay_rate

        # Do a minibatch of gradient calculation
        states, moves, probs, wins = self.memory.get_batch(self.batch_size, self.expert_ratio)
        state, moves, wins, legal_move_mask = self.neural_net.tensorise_batch(states, moves, probs, wins)
        total_loss, value_loss, pol_loss = self.neural_net.loss_function(state, target=(wins, moves),
                                                                         legal_move_mask=legal_move_mask)
        # Compute the gradients
        total_loss.backward()

        if self.training_count % self._batch_to_minibatch_ratio == 0:
            # Every N gradients we actually step the optimiser
            torch.nn.utils.clip_grad_norm_(self.neural_net.parameters(), max_norm=1.0)
            self.neural_net.optimiser.step()
            self.neural_net.optimiser.zero_grad()

        #  Store some values for plotting
        self.value_loss_window[self.training_count % self.update_period] = value_loss
        self.pol_loss_window[self.training_count % self.update_period] = pol_loss
        self.total_loss_window[self.training_count % self.update_period] = total_loss

        self.training_count += 1

        # Checked here, not in the run loop, so exactly one update is sent per weights_update_freq steps
        if self.training_count % self.weights_update_freq == 0:
            self.send_weights_update()

    def send_weights_update(self):
        weights = {k: v.cpu() for k,v in self.neural_net.state_dict().items()}
        for queue in self.weights_queues:
            try:
                queue.put_nowait(weights)
            except Full:
                # The evaluator hasn't taken the previous update yet: replace it with this newer one
                try:
                    queue.get(timeout=1.)
                except Empty:
                    pass  # the evaluator took it in the meantime
                queue.put_nowait(weights)

if __name__ == "__main__":
    from multiprocessing import Queue

    chess_net = ChessNet(input_size=[12, 8, 8], output_size=[70, 8, 8], num_repeats=32, init_lr=0.001)
    training_data_queue = Queue()

    trainer = TrainingProcess(save_dir="/home/dom/Code/chess_bot/sessions/run_at_20250218_220245",
                              neural_net=chess_net, experience_queues=[Queue()], batch_size=128,
                              num_agents=1, data_queue=Queue(), load_path="/home/dom/1TB_drive/chess_data")

    trainer.games_played = 100000

    while True:
        trainer.memory.get_batch(32, 1.)
