from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Optional
import numpy as np
from enum import Enum
import pickle as pkl

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

    def __init__(self, length_buffer, preload_data: str = None, num_agents: int = 1):

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

        if preload_data is not None:
            self.load_data(preload_data)

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

    def get_batch(self, batch_size: int = 32):

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

    def load_data(self, path: str = "neural_nets/data/games.pkl", sample_size: int = 100000):
        # Slow but we only do it once
        with open(path, "rb") as f:
            moves = pkl.load(f)

        moves = np.random.choice(moves, sample_size)

        for idx, move in enumerate(moves):
            probs = []
            for legal_move in move['legal_moves']:
                if legal_move == move['move']:
                    probs.append(1.)
                else:
                    probs.append(0.)

            data_point = DataPoint(move['fen'], move['legal_moves'], probs)

            if len(self.data) < sample_size:
                self.data.append(data_point)
            else:
                self.data[idx] = data_point

    def save_data(self):
        with open('new_data.pkl', 'wb') as f:
            pkl.dump(self.data, f)