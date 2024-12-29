from torch.utils.data import Dataset, DataLoader
from typing import List
import numpy as np
from enum import Enum


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

    def __init__(self, length_buffer):

        # Save the states
        self.data: List[DataPoint] = []

        # So we can go back and update them with who won
        self.turn_list: List[DataPoint] = []

        self.index = 0

        self.games_played = 0

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

        idxs = np.random.choice(len(self.turn_list), batch_size)

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