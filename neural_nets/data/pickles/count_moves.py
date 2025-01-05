import pickle
import glob

total_moves = 0
for filename in glob.glob("/home/dom/Code/chess_bot/neural_nets/data/pickles/*pkl"):
    with open(filename, 'rb') as f:
        while True:
            try:
                moves = pickle.load(f)
                total_moves += len(moves)
            except EOFError:
                break
print(f"Total number of moves: {total_moves}")
