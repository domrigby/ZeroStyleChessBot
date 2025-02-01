import pandas as pd
import random
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import chess.pgn
import chess_moves
import pickle
import os
import glob

from typing import List

from neural_nets.conv_net import ChessNet

class ChessDataset(Dataset):
    def __init__(self, moves_dir: str, accepted_win_types: List[str] = None, rejected_win_types: List[str] = None):
        """
        Initialize the dataset with data from multiple CSV and PGN files.
        """
        self.games = []  # Each game will have precomputed FEN states and moves
        self.chess_engine = chess_moves.ChessEngine()

        # Only accept certain win types
        self.accepted_win_types = accepted_win_types
        self.rejected_win_types = rejected_win_types

        self.move_dirs = glob.glob(moves_dir + '/*')

        self.all_moves: List[str] = []  # List of all pickle files containing precomputed FEN states and moves
        for dir in self.move_dirs:
            move_num = os.path.basename(dir)
            try:
                if int(move_num) > 75:
                    self.all_moves = self.all_moves + glob.glob(dir + '/*.pkl')
            except ValueError:
                print(f"{move_num} was skipped as it could not be converted to an int")

        # self.all_moves = np.random.choice(self.all_moves, 100000)


        print(f"Dataset built with {len(self.all_moves)} moves.")

    def __len__(self):
        """
        Return the number of games.
        """
        return len(self.all_moves)

    def __getitem__(self, idx: int):
        """
        Get a precomputed FEN state and move for a game.
        """

        pkl_path = self.all_moves[idx]

        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)

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

        value = data['value']

        # Convert FEN state and move to tensors
        board_tensor = self.chess_engine.fen_to_tensor(fen)

        # Create the illegal move mask
        legal_moves = data['legal_moves']
        legal_move_mask = torch.zeros((70, 8, 8), dtype=torch.int)
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
            'mask': legal_move_mask,
            'value': value
        }

if __name__ == '__main__':
    # Example Usage

    batch_size = 32
    chess_dataset = ChessDataset(moves_dir=r'/home/dom/Code/chess_bot/neural_nets/data/real_moves')

    idx = 0
    chess_net = ChessNet(input_size=[12, 8, 8], output_size=[70, 8, 8], num_repeats=32, init_lr=0.0001)

    import matplotlib.pyplot as plt
    import os
    import torch

    # Initialize live plots with three subplots for Total, Value, and Policy losses
    plt.ion()  # Turn on interactive mode
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15), sharex=True)

    # Total Loss
    loss_values = []
    rolling_avg_values = []
    rolling_avg_epochs = []

    # Value Loss
    value_loss_values = []
    rolling_avg_value = []

    # Policy Loss
    policy_loss_values = []
    rolling_avg_policy = []

    # Lines for rolling averages
    rolling_avg_total_line, = ax1.plot([], [], linestyle='--', label="100-Batch Rolling Avg (Total Loss)")
    rolling_avg_value_line, = ax2.plot([], [], linestyle='--', label="100-Batch Rolling Avg (Value Loss)")
    rolling_avg_policy_line, = ax3.plot([], [], linestyle='--', label="100-Batch Rolling Avg (Policy Loss)")

    # Plot configurations for Total Loss
    ax1.set_title("Total Loss")
    ax1.set_ylabel("Loss")
    ax1.grid(True)
    ax1.legend()

    # Plot configurations for Value Loss
    ax2.set_title("Value Loss")
    ax2.set_ylabel("Loss")
    ax2.grid(True)
    ax2.legend()

    # Plot configurations for Policy Loss
    ax3.set_title("Policy Loss")
    ax3.set_xlabel("Batch")
    ax3.set_ylabel("Loss")
    ax3.grid(True)
    ax3.legend()

    # Training parameters
    best_rolling_avg = float('inf')  # Initialize best rolling average
    rolling_window = 100
    batch_counter = 0

    # Training loop
    for epoch in range(10000):

        dataloader = DataLoader(chess_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

        print(f"Data points: {len(chess_dataset.all_moves)} Batches: {len(chess_dataset.all_moves) // batch_size}")

        for batch in dataloader:
            state_tensor = batch['state']
            move_target_tensor = batch['move']
            player = batch['player']
            value_target = batch['value']

            # Calculate loss
            loss, value_loss, policy_loss = chess_net.loss_function(
                state_tensor,
                (value_target.float(), move_target_tensor.float()),
                legal_move_mask=batch['mask']
            )

            # Update loss values
            loss_values.append(loss.item())
            value_loss_values.append(value_loss.item())
            policy_loss_values.append(policy_loss.item())
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

            # Update the plots
            if rolling_avg_epochs:
                rolling_avg_total_line.set_xdata(rolling_avg_epochs)
                rolling_avg_total_line.set_ydata(rolling_avg_values)
                rolling_avg_value_line.set_xdata(rolling_avg_epochs)
                rolling_avg_value_line.set_ydata(rolling_avg_value)
                rolling_avg_policy_line.set_xdata(rolling_avg_epochs)
                rolling_avg_policy_line.set_ydata(rolling_avg_policy)

            if batch_counter % 100 == 0:
                # Update x-axis limits
                ax1.set_xlim(0, batch_counter)
                ax2.set_xlim(0, batch_counter)
                ax3.set_xlim(0, batch_counter)

                # Update y-axis limits for each subplot
                ax1.set_ylim(0, max(rolling_avg_values, default=0) * 1.1)
                ax2.set_ylim(0, max(rolling_avg_value, default=0) * 1.1)
                ax3.set_ylim(0, max(rolling_avg_policy, default=0) * 1.1)

                plt.draw()
                plt.pause(0.01)  # Pause briefly to update the plot

            print(f"\rEpoch {epoch}: Batch {batch_counter}", end="")

            chess_net.scheduler.step()

    plt.ioff()  # Turn off interactive mode after training
    plt.show()  # Show the final plot

    print(f"Training complete. Best rolling average: {best_rolling_avg:.4f}")
    print(f"Model saved to {os.path.abspath(save_path)}")

