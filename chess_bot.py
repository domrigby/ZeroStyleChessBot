
import numpy as np
from numba import njit
import chess

from tree.edge import Edge
from tree.node import Node

# Initialize board
board = chess.Board()

# Numba-accelerated MCTS simulation
# @njit
def simulate_game(state, max_moves=100):

    moves = []
    legal_moves = list(state.legal_moves)
    if not legal_moves:
        return
    move = np.random.choice(len(legal_moves))
    state.push(legal_moves[move])
    moves.append(legal_moves[move])

    return state

# Example simulation
moves = simulate_game(board)
print(moves)
