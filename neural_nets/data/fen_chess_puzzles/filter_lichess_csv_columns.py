import pandas as pd

# Define file path
input_csv = "/home/dom/Code/chess_bot/neural_nets/data/fen_chess_puzzles/lichess_db_puzzle.csv"
output_csv = "/home/dom/Code/chess_bot/neural_nets/data/fen_chess_puzzles/lichess_db_puzzle_sorted.csv"

# Load only selected columns to save memory
columns_to_keep = ['FEN', 'Moves', 'Themes', 'Rating']
df = pd.read_csv(input_csv, usecols=columns_to_keep)

# Save the reduced dataset to a new CSV
df.to_csv(output_csv, index=False)

