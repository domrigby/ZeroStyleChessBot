from torch.utils.data import Dataset, DataLoader
from typing import List
import numpy as np

class Move:
    def __init__(self, move_str: str, probability: float, white_turn: bool):
        self.move_str = move_str
        self.probability = probability
        self.white_turn = white_turn

class DataPoint:
    def __init__(self, state: str, moves: List[Move]):
        self.state = state
        self.move = moves
        self.win_val = 0


class Memory:

    def __init__(self, length_buffer):

        # Save the states
        self.data: List[DataPoint]

        # So we can go back and update them with who won
        self.turn_list: List[DataPoint] = []

        self.index = 0

    def save_state_to_moves(self, state: str, observed_moves: List[str], visits: List[int]):

        #TODO: its prob faster to do tensor conversionn here

        moves: List[Move] = []

        player_to_move = state.split()[1]  # 'w' for white, 'b' for black
        if player_to_move == 'w':
            white_turn = True
        elif player_to_move == 'b':
            white_turn = False
        else:
            raise ValueError(f"Invalid player to move: {player_to_move}")

        total = np.sum(visits)
        for move, count in zip(observed_moves, visits):
            new_move = Move(move, count / total, white_turn)
            moves.append(new_move)

        data_point = DataPoint(state, moves)

        self.turn_list.append(data_point)

    def get_batch(self, batch_size: int = 32):

        idxs = np.random.choice(len(self.data), batch_size)

        batch_states = [self.data[idx] for idx in idxs]

    def end_game(self, white_win: bool):

        # Go through the memories
        for memory in self.turn_list:
            if white_win is None:
                memory.win_val = 0
            elif not memory.white_turn ^ white_win:
                memory.win_val = 1
            else:
                memory.win_val = -1

        # Empty list
        self.turn_list = []