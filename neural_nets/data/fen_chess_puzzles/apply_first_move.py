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

    loser = fen.split()[1]
    if 'opening' in rows[1]['Themes']:
        loser = None

    train = np.random.random() < 0.9

    for idx in range(len(moves) - 1):
        try:
            fen = board.push(fen, moves[idx])
        except RuntimeError:
            print('Skipping')
            break

        player = fen.split()[1]
        
        if loser is None:
            win = 0
        else:
            win = 1 if player != loser else -1

        new_dp = {'fen': fen, 'moves': moves[idx + 1], 'value': win}

        if train:
            train_moves.append(new_dp)
        else:
            test_moves.append(new_dp)

    count += 1
    print(f"{100*count/num:.3f}% done. ({count}/{num}) Training moves: {len(train_moves)} Test moves:{len(test_moves)}")
#
# csv.to_csv("/home/dom/Code/chess_bot/neural_nets/data/fen_chess_puzzles/one_move_pushed.csv", index=False)

with open('training_moves.pkl', 'wb') as f:
    np.random.shuffle(train_moves)
    pkl.dump(train_moves, f)

with open('test_moves.pkl', 'wb') as f:
    np.random.shuffle(test_moves)
    pkl.dump(test_moves, f)