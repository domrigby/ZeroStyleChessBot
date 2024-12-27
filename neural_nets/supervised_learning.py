import pandas as pd
import random
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import chess.pgn
import chess_moves
import pickle
import os

from neural_nets.conv_net import ChessNet

class ChessDataset(Dataset):
    def __init__(self, csv_paths=None, pgn_paths=None, pickle_file=None):
        """
        Initialize the dataset with data from multiple CSV and PGN files.
        """
        self.games = []  # Each game will have precomputed FEN states and moves
        self.chess_engine = chess_moves.ChessEngine()

        # Load data from multiple CSV files if provided
        if csv_paths:
            for csv_path in csv_paths:
                self.load_csv(csv_path)

        # Load data from multiple PGN files if provided
        if pgn_paths:
            for pgn_path in pgn_paths:
                self.load_pgn(pgn_path)

        if csv_paths or pgn_paths:
            # Save self.games to a file
            with open("data/games.pkl", "wb") as f:
                pickle.dump(self.games, f)

        if pickle_file:
            # Load self.games from a file
            with open("data/games.pkl", "rb") as f:
                self.games = pickle.load(f)

    def load_csv(self, csv_path):
        """
        Load game data from a CSV file.
        """
        csv_data = pd.read_csv(csv_path)
        print(f"Loading {csv_path}")
        for _, row in csv_data.iterrows():
            moves = row['moves'].split()
            if len(moves) < 5:
                continue  # Skip games with no moves
            winner = row['winner']
            self.add_game_to_dataset(moves, winner, 'algebraic')

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
                moves = [move.uci() for move in game.mainline_moves()]
                if len(moves) < 5:
                    continue  # Skip games with no moves
                winner = game.headers.get("Result", "*")
                if winner == "1-0":
                    winner = "white"
                elif winner == "0-1":
                    winner = "black"
                else:
                    winner = "draw"
                self.add_game_to_dataset(moves, winner, 'pgn')

    def add_game_to_dataset(self, moves, winner, move_type):
        """
        Precompute FEN states and moves for a single game and add to the dataset.
        """
        board = chess.Board()
        game_data = []
        for move in moves:
            fen = board.fen()
            try:

                move = board.parse_san(move)

                # Get legal moves for the current FEN state
                legal_moves = board.legal_moves
                legal_moves = [str(move) for move in legal_moves if len(str(move)) < 5]

                # Update the board for the next move
                board.push(move)

                if len(str(move)) > 4:
                    #TODO: integrate piece upgrades
                    continue

                if len(legal_moves) == 0:
                    print(f"Game state {fen} with no legal moves.")
                    continue

                game_data.append({'fen': fen, 'move': str(move), 'legal_moves': legal_moves})

            except Exception as e:
                # Skip invalid moves
                print(f"Invalid move: {move} due to {e}")
                break

        if len(game_data) >= 5:  # Ensure the game has enough moves
            self.games.append({'data': game_data, 'winner': winner, 'type': move_type})

    def __len__(self):
        """
        Return the number of games.
        """
        return len(self.games)

    def __getitem__(self, idx):
        """
        Get a precomputed FEN state and move for a game.
        """
        game = self.games[idx]
        data = random.choice(game['data'])  # Randomly select a FEN state and move
        fen = data['fen']
        move = data['move']

        # Determine the result for the player
        player = 'white' if game['data'].index(data) % 2 == 0 else 'black'
        result = 1. if game['winner'] == player else -1.

        # Convert FEN state and move to tensors
        board_tensor = self.chess_engine.fen_to_tensor(fen)
        move_tensor = torch.tensor(self.chess_engine.move_to_target(str(move)))

        # Create the illegal move mask
        legal_moves = data['legal_moves']
        legal_move_mask = torch.zeros(move_tensor.size(), dtype=torch.int)
        for legal_move in legal_moves:
            legal_move_mask[self.chess_engine.move_to_target_indices(str(legal_move))] = 1

        return {
            'state': board_tensor,
            'move': move_tensor,
            'player': player,
            'result': result,
            'mask': legal_move_mask
        }

if __name__ == '__main__':
    # Example Usage
    # Provide lists of paths to your CSV and PGN files
    csv_paths = [] #[r"/home/dom/Code/chess_bot/neural_nets/data/games.csv"]  # Replace with your CSV file paths
    pgn_paths =  [] #["/home/dom/Code/chess_bot/neural_nets/data/MagnusCarlsen-black.pgn" ,
                 # "/home/dom/Code/chess_bot/neural_nets/data/MagnusCarlsen-white.pgn",
                 # "/home/dom/Code/chess_bot/neural_nets/data/chesstianchessington-black.pgn",
                 # "/home/dom/Code/chess_bot/neural_nets/data/chesstianchessington-white.pgn"]  # Replace with your PGN file paths

    # Initialize dataset with multiple CSV and PGN files
    chess_dataset = ChessDataset(csv_paths=csv_paths, pgn_paths=pgn_paths, pickle_file='data/game.pkl')
    dataloader = DataLoader(chess_dataset, batch_size=32, shuffle=True  , num_workers=3)

    idx = 0
    chess_net = ChessNet(input_size=[12, 8, 8], output_size=[66, 8, 8], num_repeats=16, init_lr=0.00005)

    import matplotlib.pyplot as plt

    # Initialize the live plot
    plt.ion()  # Turn on interactive mode
    fig, ax = plt.subplots(figsize=(10, 6))
    loss_values = []
    rolling_avg_values = []
    epochs = []
    rolling_avg_epochs = []  # Track epochs for rolling averages

    line, = ax.plot([], [], linestyle='-', label="Batch Loss")  # Line for batch loss
    rolling_avg_line, = ax.plot([], [], linestyle='--', label="100-Batch Rolling Avg")
    ax.set_title("Training Loss (Live Update)")
    ax.set_xlabel("Batch")
    ax.set_ylabel("Loss")
    ax.grid(True)
    ax.legend()

    best_rolling_avg = float('inf')  # Initialize best rolling average
    save_path = "best_model.pth"  # Path to save the model

    # Training loop
    rolling_window = 100
    batch_counter = 0
    for epoch in range(10000):
        for batch in dataloader:
            state_tensor = batch['state']
            move_target_tensor = batch['move']
            player = batch['player']
            value_target = batch['result']

            # Calculate loss
            loss = chess_net.loss_function(state_tensor, (value_target.float(), move_target_tensor.float()),
                                           legal_move_mask=batch['mask'])

            # Update loss values
            loss_values.append(loss.item())
            epochs.append(batch_counter)
            batch_counter += 1

            # Compute rolling average if enough data is available
            if len(loss_values) >= rolling_window:
                rolling_avg = sum(loss_values[-rolling_window:]) / rolling_window
                rolling_avg_values.append(rolling_avg)
                rolling_avg_epochs.append(batch_counter - 1)

                # Save the model if this is the best rolling average
                if rolling_avg < best_rolling_avg:
                    best_rolling_avg = rolling_avg
                    torch.save(chess_net.state_dict(), f"best_model_{epoch}.pt")
                    print(f"New best model saved with rolling avg loss: {best_rolling_avg:.4f}")

            # Update the plot
            line.set_xdata(epochs)
            line.set_ydata(loss_values)

            # Update the rolling average plot only if values exist
            if rolling_avg_values:
                rolling_avg_line.set_xdata(rolling_avg_epochs)
                rolling_avg_line.set_ydata(rolling_avg_values)

            ax.set_xlim(0, batch_counter)  # Update x-axis limit
            ax.set_ylim(0, max(max(loss_values), max(rolling_avg_values, default=0)) * 1.1)  # Update y-axis limit

            plt.draw()
            plt.pause(0.01)  # Pause briefly to update the plot

            print(f"\rEpoch {epoch}: Batch {batch_counter}", end="")

            chess_net.scheduler.step()

    plt.ioff()  # Turn off interactive mode after training
    plt.show()  # Show the final plot

    print(f"Training complete. Best rolling average: {best_rolling_avg:.4f}")
    print(f"Model saved to {os.path.abspath(save_path)}")