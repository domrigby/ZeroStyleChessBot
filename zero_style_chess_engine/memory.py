from typing import List, Dict, Optional
import numpy as np
from enum import Enum
import pickle as pkl
from datetime import datetime
import glob, os

class Turn(Enum):
    BLACK = 'black'
    WHITE ='white'


class DataPoint:
    def __init__(self, state: str, moves: List[str], probs: List[float], win_val: float = 0.,
                 parent_datapoint = None):
        self.state = state
        self.moves = moves
        self.probs = probs
        self.win_val = win_val
        self.player = state.split()[1]
        self.parent_datapont = parent_datapoint


class Memory:

    def __init__(self, length_buffer, expert_data_path: str = None, num_agents: int = 1, ):

        # Save the states
        self.data: List[DataPoint] = []

        # So we can go back and update them with who won
        self.turn_list: List[DataPoint] = []

        self.index = 0
        self.games_played = 0
        self.max_len = length_buffer

        self.last_moves: Dict[int, Optional[DataPoint]] = {}
        for agent_num in range(num_agents):
            self.last_moves[agent_num] = None

        # Deal with expert data
        self.expert_data = []
        if expert_data_path is not None:
            self.load_expert_data(expert_data_path)

        #  Load data from previous games
        self.load_directory()
        #  Get the length of the current data
        self.last_index_saved = len(self.data)

    def __len__(self):
        return len(self.data)

    def save_game_to_memory(self, game, agent_id):
        """
        Create a sequence of data points
        :param game:
        :param agent_id:
        :return:
        """

        parent_move = None

        for idx, game_state in enumerate(game):

            state = game_state['state']
            observed_moves = game_state['moves']
            visits = game_state['visit_counts']
            value = game_state['value']

            moves: List[str] = []
            probs: List[float] = []

            total = np.sum(visits)
            for move, count in zip(observed_moves, visits):
                moves.append(move)
                probs.append(count/total)

            data_point = DataPoint(state, moves, probs, value, parent_move)
            self.data.append(data_point)
            parent_move = data_point

    def get_batch(self, batch_size: int = 32, expert_ratio: float = 1.):

        # Set batch size for real and expert data
        real_batch = round(batch_size * expert_ratio)
        expert_batch = batch_size - real_batch

        # Retrieve the batch
        batch = []
        batch.extend(self.get_real_data_batch(batch_size=real_batch))
        batch.extend(self.get_expert_batch(batch_size=expert_batch))
        return batch


    def get_real_data_batch(self, batch_size: int = 32):
        """
        Generate a random batch of data
        :param batch_size: size of the batch
        :return:
        """
        idxs = np.random.choice(len(self.data), batch_size)

        batch_data = [self.data[idx] for idx in idxs]

        states = [turn.state for turn in batch_data]
        moves = [turn.moves for turn in batch_data]
        probs = [turn.probs for turn in batch_data]
        wins = [turn.win_val for turn in batch_data]

        return states, moves, probs, wins

    def end_game(self, last_node: DataPoint, winner: str):

        node = last_node

        while node.parent_datapont:
            if winner is not None:
                if node.player == winner:
                    last_node.win_val = 1
                else:
                    last_node.win_val = -1

    def load_directory(self):
        """
        Function loads data from previous runs
        :return:
        """
        path = r"neural_nets/generated_data"
        files = glob.glob(os.path.join(path, "*.pkl"))

        for file in files:
            with open(file, 'rb') as f:
                moves = pkl.load(f)
                self.data.extend(moves)

    def save_data(self, file_path: str = None):
        if file_path is None:
            now = datetime.now()
            time_string = now.strftime("data_at%Y%m%d_%H%M%S")
            file_path = f"neural_nets/generated_data/{time_string}_len_{len(self.data)-self.last_index_saved}.pkl"
        with open(file_path, 'wb') as f:
            pkl.dump(self.data[self.last_index_saved:], f)
        self.last_index_saved = len(self.data)

    def load_expert_data(self, path: str, file_format: str = "pkl", max_len: int = 250000):
        """
        Collect a certain amount of expert data from the expert data directory. Data should be in the form given by networks/data
        :param path: path to directory
        :param file_format: format the data is in
        :param max_len: maximum number to retrieve
        :return:
        """
        files = glob.glob(os.path.join(path, "*" + file_format))
        num_collected = 0
        for file in files:
            with open(file, 'rb') as f:
                moves = pkl.load(f)
                self.expert_data.extend(moves)
                num_collected += len(moves)
            if num_collected >= max_len:
                break

    def get_expert_batch(self, batch_size: int = 32):
        """
        Generate a random batch of data
        :param batch_size: size of the batch
        :return:
        """
        idxs = np.random.choice(len(self.expert_data), batch_size)

        batch_data = [self.expert_data[idx] for idx in idxs]

        states = [turn['fen'] for turn in batch_data]
        moves = [turn['moves'] for turn in batch_data]
        probs = moves
        wins = [turn['value'] for turn in batch_data]

        return states, moves, probs, wins
