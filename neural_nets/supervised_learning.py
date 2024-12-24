import pandas as pd
import random
from torch.utils.data import Dataset, DataLoader
import chess.pgn
import chess_moves

class ChessDataset(Dataset):
    def __init__(self, csv_paths=None, pgn_paths=None):
        """
        Initialize the dataset with data from multiple CSV and PGN files.
        """
        self.games = []

        # Load data from multiple CSV files if provided
        if csv_paths:
            for csv_path in csv_paths:
                self.load_csv(csv_path)

        # Load data from multiple PGN files if provided
        if pgn_paths:
            for pgn_path in pgn_paths:
                self.load_pgn(pgn_path)

    def load_csv(self, csv_path):
        """
        Load game data from a CSV file.
        """
        csv_data = pd.read_csv(csv_path)
        print(f"Loading {csv_path}")
        for _, row in csv_data.iterrows():
            moves = row['moves'].split()
            winner = row['winner']
            self.games.append({'moves': moves, 'winner': winner})

    def load_pgn(self, pgn_path):
        """
        Load game data from a PGN file.
        """
        with open(pgn_path, 'r') as pgn_file:
            print(f"Loading {pgn_path}")
            while True:
                game = chess.pgn.read_game(pgn_file)
                if game is None:
                    break
                # Extract moves and winner
                board = game.board()
                moves = [move.uci() for move in game.mainline_moves()]

                if len(moves) == 0:
                    continue  # Skip games with no moves

                winner = game.headers.get("Result", "*")
                if winner == "1-0":
                    winner = "white"
                elif winner == "0-1":
                    winner = "black"
                else:
                    winner = "draw"
                self.games.append({'moves': moves, 'winner': winner})

    def __len__(self):
        """
        Return the number of games.
        """
        return len(self.games)

    def __getitem__(self, idx):
        """
        Get a random move from a random game.
        """
        # Select a random game
        game = self.games[idx]

        # Randomly select a move index
        moves = game['moves']
        move_idx = random.randint(0, len(moves) - 1)
        move = moves[:move_idx]

        # Determine the player
        player = 'white' if move_idx % 2 == 0 else 'black'

        # Determine the result for the player
        result = 1 if game['winner'] == player else 0

        return {
            'move': move,
            'player': player,
            'result': result
        }

# Example Usage
# Provide lists of paths to your CSV and PGN files
csv_paths = [r"/home/dom/Code/chess_bot/neural_nets/data/games.csv"]  # Replace with your CSV file paths
pgn_paths = ["/home/dom/Code/chess_bot/neural_nets/data/MagnusCarlsen-black.pgn",
             "/home/dom/Code/chess_bot/neural_nets/data/MagnusCarlsen-white.pgn",
             "/home/dom/Code/chess_bot/neural_nets/data/chesstianchessington-black.pgn",
             "/home/dom/Code/chess_bot/neural_nets/data/chesstianchessington-white.pgn"]  # Replace with your PGN file paths

# Initialize dataset with multiple CSV and PGN files
chess_dataset = ChessDataset(csv_paths=csv_paths, pgn_paths=pgn_paths)
dataloader = DataLoader(chess_dataset, batch_size=1, shuffle=True)

# Example: Iterate over the dataloader
for batch in dataloader:
    print(batch)