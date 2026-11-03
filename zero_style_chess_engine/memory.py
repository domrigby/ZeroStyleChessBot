from typing import List, Dict, Optional
import numpy as np
from enum import Enum
import pickle as pkl
from datetime import datetime
import glob, os
import warnings

class Turn(Enum):
    BLACK = 'black'
    WHITE ='white'


class DataPoint:
    def __init__(self, state: str, moves: List[str], probs: List[float], win_val: float = 0.):
        self.state: str = state
        self.moves: List[str] = moves
        self.probs: List[float] = probs
        self.win_val: float = win_val
        self.player: str = state.split()[1]


class Memory:

    def __init__(self, length_buffer, expert_data_path: str = None, num_agents: int = 1,
                 generated_data_dir: str = "neural_nets/generated_data"):

        # Save the states. A ring buffer: once full, self.index is the oldest entry, overwritten next
        self.data: List[DataPoint] = []

        # Data points added since the last save_data call
        self.unsaved_data: List[DataPoint] = []

        # So we can go back and update them with who won
        self.turn_list: List[DataPoint] = []

        # Store counters
        self.index: int = 0
        self.games_played: int = 0

        self.max_len: int = length_buffer
        self.generated_data_dir: str = generated_data_dir

        self.last_moves: Dict[int, Optional[DataPoint]] = {}
        for agent_num in range(num_agents):
            self.last_moves[agent_num] = None

        # Deal with expert data
        self.expert_data = []
        if expert_data_path is not None and os.path.isdir(expert_data_path):
            self.load_expert_data(expert_data_path)
        elif expert_data_path is not None:
            warnings.warn(f"Expert data directory {expert_data_path} not found: training on self-play data only")

        #  Load data from previous games
        self.load_directory()

    def __len__(self):
        return len(self.data)

    def add_data_point(self, data_point: DataPoint):
        """ Add to the buffer, overwriting the oldest data point once it holds max_len """
        if len(self.data) < self.max_len:
            self.data.append(data_point)
        else:
            self.data[self.index] = data_point
        self.index = (self.index + 1) % self.max_len

    def save_game_to_memory(self, game, agent_id):
        """
        Create a sequence of data points
        :param game:
        :param agent_id:
        :return:
        """

        game_moves: List[DataPoint] = []

        entropy_of_batch: np.array = np.zeros(len(game))

        for idx, game_state in enumerate(game):

            # Extract the states
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

            np_probs = np.array(probs)
            entropy = -np.sum(np_probs * np.log(np_probs, where=(np_probs > 0), out=np.zeros_like(np_probs)))
            entropy_of_batch[idx] = entropy

            # Append the move to the set
            data_point: DataPoint = DataPoint(state, moves, probs, value)
            game_moves.append(data_point)

        # In order to maintain semi-IID data and prevent overfitting, I take two moves from each game from the training.
        # These are chosen due according to which ever have the highest entropy (lowest certain of move)
        # Reason for max adding... so that max entropy is only maximum 10 times more likely than the lowest possible moves
        print(f"Entropy = {entropy_of_batch}")
        entropy_of_batch += 0.1 * entropy_of_batch.max()
        chosen_moves = np.random.choice(game_moves, size=min(2, len(game_moves)),
                                        p=entropy_of_batch/entropy_of_batch.sum(), replace=False)
        for data_point in chosen_moves:
            self.add_data_point(data_point)
            self.unsaved_data.append(data_point)

    def get_batch(self, batch_size: int = 32, expert_ratio: float = 1.):
        """
        Mixes real and generate data to prevent overfitting to the new data
        """

        # Set batch size for real and expert data
        expert_batch = round(batch_size * expert_ratio) if self.expert_data else 0
        real_batch = batch_size - expert_batch

        # Retrieve the batch
        if real_batch > 0:
            states, moves, probs, wins = self.get_real_data_batch(batch_size=real_batch)
        else:
            states, moves, probs, wins = [], [], [], []

        # Extend the current batch with some new data
        if expert_batch > 0:
            new_batch = self.get_expert_batch(batch_size=expert_batch)

            states.extend(new_batch[0])
            moves.extend(new_batch[1])
            probs.extend(new_batch[2])
            wins.extend(new_batch[3])

        return states, moves, probs, wins


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

    def load_directory(self):
        """
        Function loads data from previous runs: the most recent files, up to max_len data points
        :return:
        """
        files = sorted(glob.glob(os.path.join(self.generated_data_dir, "*.pkl")), key=os.path.getmtime, reverse=True)

        newest_first: List[List[DataPoint]] = []
        num_loaded = 0
        for file in files:
            if num_loaded >= self.max_len:
                break
            with open(file, 'rb') as f:
                moves = pkl.load(f)
            newest_first.append(moves)
            num_loaded += len(moves)

        # Add oldest first so the ring buffer keeps the newest max_len
        for moves in reversed(newest_first):
            for data_point in moves:
                self.add_data_point(data_point)

    def save_data(self, file_path: str = None):
        if file_path is None:
            now = datetime.now()
            time_string = now.strftime("data_at%Y%m%d_%H%M%S")
            file_path = os.path.join(self.generated_data_dir, f"{time_string}_len_{len(self.unsaved_data)}.pkl")
        if os.path.dirname(file_path):
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as f:
            pkl.dump(self.unsaved_data, f)
        self.unsaved_data = []

    def load_expert_data(self, path: str, file_format: str = "pkl", max_len: int = 250000):
        """
        Collect a certain amount of expert data from the expert data directory. Data should be in the form given by neural_nets/data
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
        moves = [[turn['moves']] for turn in batch_data]
        probs = [[1.] for turn in batch_data] # Expert data only has one move
        wins = [turn['value'] for turn in batch_data]

        return states, moves, probs, wins
