import argparse
import pandas as pd
import random
from torch.utils.data import Dataset, DataLoader, random_split
import torch
import numpy as np
import chess.pgn
import chess_moves
import pickle
import pickle as pkl
import glob
import time

from typing import List

from neural_nets.conv_net import ChessNet


class ChessDataset(Dataset):

    def __init__(self, moves_dir: str, accepted_win_types: List[str] = None, rejected_win_types: List[str] = None,
                 sample_size: int = 2500000, random_sample: bool = True):
        """
        Initialize the dataset with data from multiple CSV and PGN files.
        """
        self.sample_size = sample_size
        self.dataset_path = moves_dir

        self.games = []  # Each game will have precomputed FEN states and moves
        self.chess_engine = chess_moves.ChessEngine()

        # Only accept certain win types
        self.accepted_win_types = accepted_win_types
        self.rejected_win_types = rejected_win_types

        self.move_dirs = glob.glob(moves_dir + '/*')
        self.all_moves: List[str] = []  # List of all pickle files containing precomputed FEN states and moves

        if random_sample:
            self.sample_dataset()
        else:
            self.non_random_dataset_chunk()

        self.chunk_count = 0

    def sample_dataset(self):
        with open(self.dataset_path, 'rb') as f:
            pickle_moves = pkl.load(f)
            this_sample_size = min(len(pickle_moves), self.sample_size)
            self.all_moves = np.random.choice(pickle_moves, this_sample_size, replace=False)

    def non_random_dataset_chunk(self):

        lb = self.chunk_count * self.sample_size
        ub = (self.chunk_count + 1) * self.sample_size

        with open(self.dataset_path, 'rb') as f:
            pickle_moves = pkl.load(f)
            ub = min(ub, len(pickle_moves))
            self.all_moves = pickle_moves[lb:ub]

    def __len__(self):
        """
        Return the number of games.
        """
        return len(self.all_moves)

    def __getitem__(self, idx: int):
        """
        Get a precomputed FEN state and move for a game.
        """
        data = self.all_moves[idx]

        fen = data['fen']
        move = data['moves']

        # Determine the result for the player
        player = "white" if fen.split()[1] == 'w' else "black"

        value = data['value']

        # Convert FEN state and move to tensors
        board_tensor = self.chess_engine.fen_to_tensor(fen)

        # Create move tensor
        move_tensor = torch.zeros((70, 8, 8), dtype=torch.int)

        if fen.split()[1] == "b":
            move = self.chess_engine.unflip_move(str(move))
            indices = self.chess_engine.move_to_target_indices(move)
        else:
            indices = self.chess_engine.move_to_target_indices(str(move))

        move_tensor[indices] = 1

        # # Create the legal move mask
        # self.chess_engine.set_fen(fen)
        # legal_moves = self.chess_engine.legal_moves()
        # legal_move_mask, leg_move_lookup = self.create_legal_move_mask(legal_moves, player)
        #
        # if legal_move_mask[torch.unravel_index(move_tensor.argmax(), (70, 8, 8))] != 1:
        #     # raise RuntimeError(f"Masking tensor does include the move target. \n\tDetails:\n\t\tPlayer = {player}"
        #     #                    f"\n\t\tFEN = {fen}"
        #     #                    f"\n\t\tOriginal move = {data['moves']}"
        #     #                    f"\n\t\tMove = {move}"
        #     #                    f"\n\t\tLegal moves = {[move for move, _ in leg_move_lookup]}")
        #     return self.__getitem__(np.random.choice(len(self)))

        return {
            'state': board_tensor,
            'move': move_tensor,
            'player': player,
            'value': value
        }

    def create_legal_move_mask(self, moves, team):
        move_to_indices_lookup = []

        # Collect indices for batch updates
        all_indices = []

        for legal_move in moves:
            # Ignore pawn promotions for now
            if len(str(legal_move)) < 5:
                if team == "black":
                    legal_move_str = self.chess_engine.unflip_move(str(legal_move))
                    indices = self.chess_engine.move_to_target_indices(legal_move_str)
                else:
                    indices = self.chess_engine.move_to_target_indices(str(legal_move))

                all_indices.append(indices)
                move_to_indices_lookup.append([legal_move, indices])

        # Create mask efficiently with batched updates
        legal_move_mask = torch.zeros((70, 8, 8), dtype=torch.float32)
        if all_indices:  # only if we have legal moves
            idx_tensor = torch.tensor(all_indices, dtype=torch.long)  # (N,3)
            c, r, k = idx_tensor[:, 0], idx_tensor[:, 1], idx_tensor[:, 2]
            legal_move_mask[c, r, k] = 1.0  # set exactly those cells to 1

        return legal_move_mask, move_to_indices_lookup


if __name__ == '__main__':
    # Parse arguments for resuming training from a checkpoint
    parser = argparse.ArgumentParser(description="Train ChessNet with checkpointing.")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint file to resume training.")
    args = parser.parse_args()

    # Parameters
    batch_size = 64

    # Initialize the network
    chess_net = ChessNet(input_size=[12, 8, 8], output_size=[70, 8, 8], num_repeats=32, num_filters=64, init_lr=1e-3)

    import matplotlib.pyplot as plt

    # Initialize live plots with four subplots:
    # ax1: Total Loss, ax2: Value Loss, ax3: Policy Loss, ax4: Validation Loss
    plt.ion()  # Turn on interactive mode
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 20), sharex=True)

    # Loss tracking lists for training
    loss_values = []
    rolling_avg_values = []
    rolling_avg_epochs = []
    value_loss_values = []
    rolling_avg_value = []
    policy_loss_values = []
    rolling_avg_policy = []

    # Tracking validation loss per epoch
    epoch_val_losses = []
    epoch_val_epochs = []

    # Lines for rolling averages (training)
    rolling_avg_total_line, = ax1.plot([], [], linestyle='--', label="100-Batch Rolling Avg (Total Loss)")
    rolling_avg_value_line, = ax2.plot([], [], linestyle='--', label="100-Batch Rolling Avg (Value Loss)")
    rolling_avg_policy_line, = ax3.plot([], [], linestyle='--', label="100-Batch Rolling Avg (Policy Loss)")

    # Line for validation loss
    val_loss_line, = ax4.plot([], [], linestyle='-', label="Validation Loss")

    # Configure the plots
    ax1.set_title("Total Loss")
    ax1.set_ylabel("Loss")
    ax1.grid(True)
    ax1.legend()

    ax2.set_title("Value Loss")
    ax2.set_ylabel("Loss")
    ax2.grid(True)
    ax2.legend()

    ax3.set_title("Policy Loss")
    ax3.set_ylabel("Loss")
    ax3.grid(True)
    ax3.legend()

    ax4.set_title("Validation Loss")
    ax4.set_xlabel("Epoch")
    ax4.set_ylabel("Loss")
    ax4.grid(True)
    ax4.legend()

    # Training parameters
    best_val_loss = float('inf')
    best_rolling_avg = float('inf')
    rolling_window = 100
    batch_counter = 0

    # Check if we need to resume from a checkpoint
    starting_epoch = 0
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint)
        starting_epoch = checkpoint['epoch'] + 1
        batch_counter = checkpoint['batch_counter']
        chess_net.load_state_dict(checkpoint['model_state_dict'])
        # chess_net.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        loss_values = checkpoint['loss_values']
        value_loss_values = checkpoint['value_loss_values']
        policy_loss_values = checkpoint['policy_loss_values']
        rolling_avg_values = checkpoint['rolling_avg_values']
        rolling_avg_value = checkpoint['rolling_avg_value']
        rolling_avg_policy = checkpoint['rolling_avg_policy']
        rolling_avg_epochs = checkpoint['rolling_avg_epochs']
        best_val_loss = checkpoint['best_val_loss']
        best_rolling_avg = checkpoint['best_rolling_avg']
        print(f"Resumed training from checkpoint: {args.checkpoint} at epoch {starting_epoch}")

    # Record start time for elapsed time reporting and checkpointing
    start_time = time.time()
    last_checkpoint_time = time.time()

    import glob
    train_files = glob.glob("/home/dom/1TB_drive/chess_data/train*pkl")
    test_files = glob.glob("/home/dom/1TB_drive/chess_data/test*pkl")


    # Define the test data set size (these are completely unseen games)
    test_dataset = ChessDataset(moves_dir=np.random.choice(test_files), sample_size=100000)
    val_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # Training loop
    for epoch in range(starting_epoch, 500000):

        # Build the datasets
        train_dataset = ChessDataset(moves_dir=np.random.choice(train_files), sample_size=250000)

        print(f"Train dataset: {len(train_dataset)} samples, Validation dataset: {len(test_dataset)} samples.")

        # Create dataloaders for training and validation
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=5)
        chess_net.train()
        for batch in train_dataloader:
            state_tensor = batch['state']
            move_target_tensor = batch['move']
            value_target = batch['value']

            # Calculate loss
            loss, value_loss, policy_loss = chess_net.loss_function(
                state_tensor,
                (value_target.float(), move_target_tensor.float())
            )

            # Update loss values for plotting
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

                # Save the model if the rolling average of training loss is the best so far
                if rolling_avg < best_rolling_avg:
                    best_rolling_avg = rolling_avg
                    model_save_path = f"networks/best_model_{epoch}.pt"
                    torch.save(chess_net.state_dict(), model_save_path)
                    print(f"New best model saved with rolling avg loss: {best_rolling_avg:.4f} at {model_save_path}")

            # Update the training plots every batch
            if rolling_avg_epochs:
                rolling_avg_total_line.set_xdata(rolling_avg_epochs)
                rolling_avg_total_line.set_ydata(rolling_avg_values)
                rolling_avg_value_line.set_xdata(rolling_avg_epochs)
                rolling_avg_value_line.set_ydata(rolling_avg_value)
                rolling_avg_policy_line.set_xdata(rolling_avg_epochs)
                rolling_avg_policy_line.set_ydata(rolling_avg_policy)

            if batch_counter % 1000 == 0:
                ax1.set_xlim(0, batch_counter)
                ax2.set_xlim(0, batch_counter)
                ax3.set_xlim(0, batch_counter)
                # Adjust y-limits based on current data
                ax1.set_ylim(0, max(rolling_avg_values, default=0) * 1.1)
                ax2.set_ylim(0, max(rolling_avg_value, default=0) * 1.1)
                ax3.set_ylim(0, max(rolling_avg_policy, default=0) * 1.1)
                plt.draw()
                plt.pause(0.01)

            # Checkpoint saving every 10 minutes (600 seconds)
            if time.time() - last_checkpoint_time >= 600:
                checkpoint_dict = {
                    'epoch': epoch,
                    'batch_counter': batch_counter,
                    'model_state_dict': chess_net.state_dict(),
                    'scheduler_state_dict': chess_net.scheduler.state_dict(),
                    'loss_values': loss_values,
                    'value_loss_values': value_loss_values,
                    'policy_loss_values': policy_loss_values,
                    'rolling_avg_values': rolling_avg_values,
                    'rolling_avg_value': rolling_avg_value,
                    'rolling_avg_policy': rolling_avg_policy,
                    'rolling_avg_epochs': rolling_avg_epochs,
                    'best_val_loss': best_val_loss,
                    'best_rolling_avg': best_rolling_avg,
                }
                checkpoint_path = "training_checkpoint.pt"
                torch.save(checkpoint_dict, checkpoint_path)
                # Save the current plot as an image
                plt.savefig("training_progress.png")
                print(f"\nCheckpoint saved at epoch {epoch}, batch {batch_counter} to {checkpoint_path}")
                last_checkpoint_time = time.time()

            # Calculate elapsed time and print training progress
            current_lr = chess_net.scheduler.get_last_lr()[0]
            elapsed_time = time.time() - start_time
            hours, rem = divmod(elapsed_time, 3600)  # Get hours and remaining seconds
            minutes, seconds = divmod(rem, 60)  # Get minutes and remaining seconds
            print(
                f"\rTime: {int(hours):02}:{int(minutes):02}:{int(seconds):02} - Epoch: {epoch} Batch: {batch_counter} LR: {current_lr:.3e}",
                end="")

            # Update learning rate
            chess_net.scheduler.step()

        # -------------------- Validation Step --------------------
        chess_net.eval()
        val_loss_total = 0.0
        val_batches = 0
        with torch.no_grad():
            for batch in val_dataloader:
                state_tensor = batch['state']
                move_target_tensor = batch['move']
                value_target = batch['value']
                loss, _, _ = chess_net.loss_function(
                    state_tensor,
                    (value_target.float(), move_target_tensor.float()),
                    training=False
                )
                val_loss_total += loss.item()
                val_batches += 1
        avg_val_loss = val_loss_total / val_batches if val_batches > 0 else float('inf')
        print(f"\nEpoch {epoch} validation loss: {avg_val_loss:.4f}")

        # Update validation loss tracking and plot
        epoch_val_losses.append(avg_val_loss)
        epoch_val_epochs.append(batch_counter)
        val_loss_line.set_xdata(epoch_val_epochs)
        val_loss_line.set_ydata(epoch_val_losses)
        ax4.set_xlim(0, epoch + 1)
        ax4.set_ylim(0, max(epoch_val_losses, default=0) * 1.1)
        plt.draw()
        plt.pause(0.01)

        # Optionally, save the best model based on validation loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model_save_path = f"networks/best_model_val_{epoch}.pt"
            torch.save(chess_net.state_dict(), model_save_path)
            print(f"New best model based on validation loss saved at {model_save_path}")

    plt.ioff()  # Turn off interactive mode after training
    plt.show()

    print(f"Training complete. Best rolling average: {best_rolling_avg:.4f}")
    print(f"Best validation loss: {best_val_loss:.4f}")
