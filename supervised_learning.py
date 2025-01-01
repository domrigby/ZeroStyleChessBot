import pandas as pd
import random
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import chess.pgn
import chess_moves
import pickle
import os

from typing import List

from neural_nets.conv_net import ChessNet

class ChessDataset(Dataset):
    def __init__(self, csv_paths=None, pgn_paths=None, pickle_file=None, sample_size: int= 100000,
                 accepted_win_types: List[str] = None, rejected_win_types: List[str] = None):
        """
        Initialize the dataset with data from multiple CSV and PGN files.
        """
        self.games = []  # Each game will have precomputed FEN states and moves
        self.chess_engine = chess_moves.ChessEngine()

        # Only accept certain win types
        self.accepted_win_types = accepted_win_types

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
            with open("neural_nets/data/games.pkl", "wb") as f:
                pickle.dump(self.games, f)

        if pickle_file:
           self.load_pickled_file(sample_size=sample_size)

    def load_pickled_file(self, path: str = "neural_nets/data/games.pkl", sample_size: int = None):
        with open(path, "rb") as f:
            loaded_games = pickle.load(f)
            if sample_size:
                loaded_games = np.random.choice(loaded_games, sample_size)
            self.games = loaded_games


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
            self.add_game_to_dataset(moves, winner)

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
                game_win_type = game.headers.get("Termination", "*")

                for win_type in self.reject_win_types:
                    if win_type in game_win_types:
                        continue
                if winner == "1-0":
                    winner = "white"
                elif winner == "0-1":
                    winner = "black"
                else:
                    winner = "draw"
                self.add_game_to_dataset(moves, winner)

    def add_game_to_dataset(self, moves, winner):
        """
        Precompute FEN states and moves for a single game and add to the dataset.
        """
        board = chess.Board()

        if len(moves) < 5:
            # Skip this game
            return

        for idx, move in enumerate(moves):
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

                self.games.append({'fen': fen, 'move': str(move), 'legal_moves': legal_moves, 'winner': winner, 'turn_num': idx})

            except Exception as e:
                # Skip invalid moves
                print(f"Invalid move: {move} due to {e}")
                break

    def __len__(self):
        """
        Return the number of games.
        """
        return len(self.games)

    def __getitem__(self, idx):
        """
        Get a precomputed FEN state and move for a game.
        """
        data = self.games[idx]
        fen = data['fen']
        move = data['move']

        # Determine the result for the player
        player = "white" if fen.split()[1]== 'w' else "black"

        if data['winner'] == player:
            result = 1
        elif data['winner'] == "draw":
            result = 0
        else:
            result = -1

        # Convert FEN state and move to tensors
        board_tensor = self.chess_engine.fen_to_tensor(fen)

        # Create the illegal move mask
        legal_moves = data['legal_moves']
        legal_move_mask = torch.zeros((66, 8, 8), dtype=torch.int)
        for legal_move in legal_moves:
            if player == "black":
                legal_move = self.chess_engine.unflip_move(str(legal_move))
            legal_move_mask[self.chess_engine.move_to_target_indices(str(legal_move))] = 1

        num_legal_moves = len(legal_move_mask)

        # Give all other moves 2.5%
        true_move_prob = 0.8
        other_move_probs = (1 - true_move_prob) / (num_legal_moves - 1)
        scaling_factor = 1. / other_move_probs
        move_tensor = legal_move_mask / scaling_factor # Set all 1 values to 0.01
        if player == "black":
            move = self.chess_engine.unflip_move(move)
        move_tensor[self.chess_engine.move_to_target_indices(str(move))] = true_move_prob

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
    pgn_paths =  [] # ["/home/dom/Code/chess_bot/neural_nets/data/MagnusCarlsen-black.pgn" ,
                 # "/home/dom/Code/chess_bot/neural_nets/data/MagnusCarlsen-white.pgn",
                 # "/home/dom/Code/chess_bot/neural_nets/data/chesstianchessington-black.pgn",
                 # "/home/dom/Code/chess_bot/neural_nets/data/chesstianchessington-white.pgn"]  # Replace with your PGN file paths

    # Initialize dataset with multiple CSV and PGN files
    batch_size = 32
    chess_dataset = ChessDataset(csv_paths=csv_paths, pgn_paths=pgn_paths, pickle_file='data/game.pkl')

    idx = 0
    chess_net = ChessNet(input_size=[12, 8, 8], output_size=[66, 8, 8], num_repeats=16, init_lr=0.0001)

    import matplotlib.pyplot as plt

    # Initialize the live plot
    plt.ion()  # Turn on interactive mode
    fig, ax = plt.subplots(figsize=(10, 6))
    loss_values = []
    policy_loss_values = []
    value_loss_values = []
    rolling_avg_values = []
    rolling_avg_policy = []
    rolling_avg_value = []
    epochs = []
    rolling_avg_epochs = []  # Track epochs for rolling averages

    rolling_avg_line, = ax.plot([], [], linestyle='--', label="100-Batch Rolling Avg (Total Loss)")
    rolling_avg_policy_line, = ax.plot([], [], linestyle='--', label="100-Batch Rolling Avg (Policy Loss)")
    rolling_avg_value_line, = ax.plot([], [], linestyle='--', label="100-Batch Rolling Avg (Value Loss)")

    ax.set_title("Training Loss")
    ax.set_xlabel("Batch")
    ax.set_ylabel("Loss")
    ax.grid(True)
    ax.legend()

    best_rolling_avg = float('inf')  # Initialize best rolling average
    save_path = "best_model.pth"  # Path to save the model

    # Training loop
    rolling_window = 1000
    batch_counter = 0

    for epoch in range(10000):

        chess_dataset.load_pickled_file(sample_size=100000)
        dataloader = DataLoader(chess_dataset, batch_size=batch_size, shuffle=True, num_workers=1)

        print(f"Data points: {len(chess_dataset.games)} Batches: {len(chess_dataset.games) // batch_size}")

        for batch in dataloader:
            state_tensor = batch['state']
            move_target_tensor = batch['move']
            player = batch['player']
            value_target = batch['result']

            # Calculate loss
            loss, value_loss, policy_loss = chess_net.loss_function(state_tensor, (
            value_target.float(), move_target_tensor.float()),legal_move_mask=batch['mask'])

            # Update loss values
            loss_values.append(loss.item())
            policy_loss_values.append(policy_loss.item())
            value_loss_values.append(value_loss.item())
            epochs.append(batch_counter)
            batch_counter += 1

            # Compute rolling averages if enough data is available
            if len(loss_values) >= rolling_window:
                rolling_avg = sum(loss_values[-rolling_window:]) / rolling_window
                rolling_avg_policy_loss = sum(policy_loss_values[-rolling_window:]) / rolling_window
                rolling_avg_value_loss = sum(value_loss_values[-rolling_window:]) / rolling_window

                rolling_avg_values.append(rolling_avg)
                rolling_avg_policy.append(rolling_avg_policy_loss)
                rolling_avg_value.append(rolling_avg_value_loss)
                rolling_avg_epochs.append(batch_counter - 1)

                # Save the model if this is the best rolling average
                if rolling_avg < best_rolling_avg:
                    best_rolling_avg = rolling_avg
                    torch.save(chess_net.state_dict(), f"networks/best_model_{epoch}.pt")
                    print(f"New best model saved with rolling avg loss: {best_rolling_avg:.4f}")

            # Update the plot

            if rolling_avg_values:
                rolling_avg_line.set_xdata(rolling_avg_epochs)
                rolling_avg_line.set_ydata(rolling_avg_values)
                rolling_avg_policy_line.set_xdata(rolling_avg_epochs)
                rolling_avg_policy_line.set_ydata(rolling_avg_policy)
                rolling_avg_value_line.set_xdata(rolling_avg_epochs)
                rolling_avg_value_line.set_ydata(rolling_avg_value)

            if batch_counter % 100 == 0:
                ax.set_xlim(0, batch_counter)  # Update x-axis limit
                ax.set_ylim(0, max(rolling_avg_values + rolling_avg_policy + rolling_avg_value,
                                   default=0) * 1.1)  # Update y-axis limit

                plt.draw()
                plt.pause(0.01)  # Pause briefly to update the plot

            print(f"\rEpoch {epoch}: Batch {batch_counter}", end="")

            chess_net.scheduler.step()

    plt.ioff()  # Turn off interactive mode after training
    plt.show()  # Show the final plot

    print(f"Training complete. Best rolling average: {best_rolling_avg:.4f}")
    print(f"Model saved to {os.path.abspath(save_path)}")
