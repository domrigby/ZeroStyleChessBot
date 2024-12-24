from torch.utils.data import Dataset, DataLoader
from typing import List
import numpy as np

class Move:
    def __init__(self, move_str: str, probability: float):
        self.move_str = move_str
        self.probability = probability

class Memory:

    def __init__(self, length_buffer):

        # Save the states
        self.states: List[str] = []

        # Save the corresponding moves
        self.moves: List[Move] = []

        self.index = 0

    def save_state_to_moves(self, state: str, moves: List[str], visits: List[int]):

        self.states.append(state)

        total = np.sum(visits)
        for move, count in zip(moves, visits):
            self.moves.append(Move(move, count / total))