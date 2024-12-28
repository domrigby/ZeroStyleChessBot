import unittest
from chess_moves import ChessEngine
import numpy as np

chess_moves_list = [
    # Queen-like moves: Vertical (Up)
    ("a2a3", (0, 6, 0)), ("a2a4", (8, 6, 0)), ("a2a5", (16, 6, 0)),
    ("a2a6", (24, 6, 0)), ("a2a7", (32, 6, 0)), ("a2a8", (40, 6, 0)), ("a2a1", (48, 6, 0)),

    # Queen-like moves: Vertical (Down)
    ("a7a6", (1, 1, 0)), ("a7a5", (9, 1, 0)), ("a7a4", (17, 1, 0)),
    ("a7a3", (25, 1, 0)), ("a7a2", (33, 1, 0)), ("a7a1", (41, 1, 0)), ("a7a8", (49, 1, 0)),

    # Queen-like moves: Horizontal (Right)
    ("a2b2", (2, 6, 0)), ("a2c2", (10, 6, 0)), ("a2d2", (18, 6, 0)),
    ("a2e2", (26, 6, 0)), ("a2f2", (34, 6, 0)), ("a2g2", (42, 6, 0)), ("a2h2", (50, 6, 0)),

    # Queen-like moves: Horizontal (Left)
    ("h2g2", (3, 6, 7)), ("h2f2", (11, 6, 7)), ("h2e2", (19, 6, 7)),
    ("h2d2", (27, 6, 7)), ("h2c2", (35, 6, 7)), ("h2b2", (43, 6, 7)), ("h2a2", (51, 6, 7)),

    # Queen-like moves: Diagonal (Up-Right)
    ("a2b3", (4, 6, 0)), ("a2c4", (12, 6, 0)), ("a2d5", (20, 6, 0)),
    ("a2e6", (28, 6, 0)), ("a2f7", (36, 6, 0)), ("a2g8", (44, 6, 0)),

    # Queen-like moves: Diagonal (Up-Left)
    ("h2g3", (5, 6, 7)), ("h2f4", (13, 6, 7)), ("h2e5", (21, 6, 7)),
    ("h2d6", (29, 6, 7)), ("h2c7", (37, 6, 7)), ("h2b8", (45, 6, 7)),

    # Queen-like moves: Diagonal (Down-Right)
    ("a7b6", (6, 1, 0)), ("a7c5", (14, 1, 0)), ("a7d4", (22, 1, 0)),
    ("a7e3", (30, 1, 0)), ("a7f2", (38, 1, 0)), ("a7g1", (46, 1, 0)),

    # Queen-like moves: Diagonal (Down-Left)
    ("h7g6", (7, 1, 7)), ("h7f5", (15, 1, 7)), ("h7e4", (23, 1, 7)),
    ("h7d3", (31, 1, 7)), ("h7c2", (39, 1, 7)), ("h7b1", (47, 1, 7)),

    # Knight Moves
    ("g1h3", (56, 7, 6)), ("g1f3", (57, 7, 6)), ("g1e2", (58, 7, 6)), ("g1e4", (59, 7, 6)),
    ("b1c3", (60, 7, 1)), ("b1a3", (61, 7, 1)), ("g3f2", (62, 7, 7)),

    ("o-o", (63, 0, 0))]


chess_engine = ChessEngine()

for (move, idxs) in chess_moves_list:

    try:
        output = chess_engine.move_to_target(move)
        arg_max_idx = np.argmax(output)

        channel, row, column = np.unravel_index(arg_max_idx, np.array(output).shape)

        pred_idxs = chess_engine.move_to_target_indices(move)

        print(pred_idxs, (channel, row, column))

        move_returned = chess_engine.indices_to_move(channel, row, column)

        print(move, move_returned)

    except Exception as e:
        print(f"Error with move {move}: {e}")