from torch.utils.data import Dataset, DataLoader
from typing import List
import numpy as np
from enum import Enum
import pickle as pkl


class Turn(Enum):
    BLACK = 'black'
    WHITE ='white'


class DataPoint:
    def __init__(self, state: str, moves: List[str], probs: List[float]):
        self.state = state
        self.moves = moves
        self.probs = probs
        self.win_val = 0
        self.white_turn = True if state.split()[1] == 'w' else False


class Memory:

    def __init__(self, length_buffer, preload_data: str = None):

        # Save the states
        self.data: List[DataPoint] = []

        # So we can go back and update them with who won
        self.turn_list: List[DataPoint] = []

        self.index = 0

        self.games_played = 0

        self.max_len = length_buffer

        if preload_data is not None:
            self.load_data(preload_data)

    def __len__(self):
        return len(self.data)

    def save_state_to_moves(self, state: str, observed_moves: List[str], visits: List[int]):

        #TODO: its prob faster to do tensor conversion here

        moves: List[str] = []
        probs: List[float] = []

        total = np.sum(visits)
        for move, count in zip(observed_moves, visits):
            moves.append(move)
            probs.append(count/total)

        data_point = DataPoint(state, moves, probs)

        self.turn_list.append(data_point)
        self.data.append(data_point)

    def get_batch(self, batch_size: int = 32):

        idxs = np.random.choice(len(self.data), batch_size)

        batch_data = [self.data[idx] for idx in idxs]

        states = [turn.state for turn in batch_data]
        moves = [turn.moves for turn in batch_data]
        probs = [turn.probs for turn in batch_data]
        wins = [turn.win_val for turn in batch_data]

        return states, moves, probs, wins


    def end_game(self, white_win: bool):

        # Go through the memories and update the winner
        for memory in self.turn_list:
            if white_win is None:
                memory.win_val = 0
            elif not memory.white_turn ^ white_win:
                memory.win_val = 1
            else:
                memory.win_val = -1

        # Empty list
        self.turn_list = []
        self.games_played += 1

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