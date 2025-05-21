# Lichess puzzles start after one move. We therefore have to apply the first move to get a proper dataset

import chess_moves
import pandas as pd
import pickle as pkl
import numpy as np

csv = pd.read_csv("/home/dom/Code/chess_bot/neural_nets/data/fen_chess_puzzles/lichess_db_puzzle_sorted.csv")

board = chess_moves.ChessEngine()

num = len(csv)
count = 0
train_moves = []
test_moves = []

for rows in csv.iterrows():
    fen = rows[1]['FEN']
    moves = rows[1]['Moves'].split()

    # Select a move at random to add to the data set, not the first because thats a bad move
    move_to_include = np.random.choice(moves[1:])

    loser = fen.split()[1]
    if 'opening' in rows[1]['Themes']:
        win_val = 0
    else:
        win_val = 1

    train = np.random.random() < 0.9

    for idx in range(len(moves) - 1):
        try:
            fen = board.push(fen, moves[idx])
        except RuntimeError:
            print('Skipping')
            break

        player = fen.split()[1]

        win = win_val if player != loser else -win_val

        new_dp = {'fen': fen, 'moves': moves[idx + 1], 'value': win}

        if moves[idx] == move_to_include:
            if train:
                train_moves.append(new_dp)
            else:
                test_moves.append(new_dp)
            # Continue to next move
            break

    count += 1
    print(f"{100*count/num:.3f}% done. ({count}/{num}) Training moves: {len(train_moves)} Test moves:{len(test_moves)}")

# with open(f'/home/dom/Code/chess_bot/neural_nets/data/fen_chess_puzzles/pickled_data/training_moves.pkl', 'wb') as f:
#     pkl.dump(train_moves, f)
#
# with open('/home/dom/Code/chess_bot/neural_nets/data/fen_chess_puzzles/pickled_data/test_moves.pkl', 'wb') as f:
#     np.random.shuffle(test_moves)
#     pkl.dump(test_moves, f)

import glob
appending_to = "/home/dom/1TB_drive/chess_data"
train_files = glob.glob(appending_to + "/train*.pkl")

chunk_size = int(len(train_moves) / len(train_files))

#  Shuffle first
np.random.shuffle(train_moves)

for chunk_idx, file in enumerate(train_files):
    with open(file, 'rb') as f:
        moves = pkl.load(f)

    lb = chunk_idx * chunk_size
    ub = max((chunk_idx + 1) * chunk_size, len(train_moves) + 1)

    moves.extend(train_moves[lb:ub])

    with open(file, 'wb') as f:
        pkl.dump(moves, f)

    print(f"{file} complete. Extended by ")

test_files = glob.glob(appending_to + "/test*.pkl")

chunk_size = int(len(test_moves) / len(test_files))

#  Shuffle first
np.random.shuffle(test_moves)

for chunk_idx, file in enumerate(test_files):
    with open(file, 'rb') as f:
        moves = pkl.load(f)

    lb = chunk_idx * chunk_size
    ub = max((chunk_idx + 1) * chunk_size, len(test_moves) + 1)

    moves.extend(test_moves[lb:ub])

    with open(file, 'wb') as f:
        pkl.dump(moves, f)
