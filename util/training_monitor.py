import os
import json
from torch.utils.tensorboard import SummaryWriter
import numpy as np


class ChessEngineMonitor:
    def __init__(self, save_dir="runs/chess_engine"):
        self.writer = SummaryWriter(log_dir=save_dir)
        self.save_file = os.path.join(save_dir, "chess_monitor_data.json")

        # Data placeholders
        self.win_counts = {"White Wins": 0, "Black Wins": 0, "Draws": 0}
        self.game_lengths = []
        self.loss_history = {"Total Loss": [], "Value Loss": [], "Policy Loss": []}
        self.experience_length = []
        self.exploration_history = []

        if os.path.exists(self.save_file):
            self.load_progress()

        self.iteration = 0  # Tracks the iteration for x-axis in TensorBoard

        print(f"Tensorboard command: \ntensorboard --logdir={save_dir}")

    def update_wins(self, white_win):
        if white_win:
            self.win_counts["White Wins"] += 1
        elif white_win is not None:
            self.win_counts["Black Wins"] += 1
        else:
            self.win_counts["Draws"] += 1

        total_games = sum(self.win_counts.values())
        for key, value in self.win_counts.items():
            self.writer.add_scalar(f"Wins/{key}", value, total_games)

    def update_game_lengths(self, new_length):
        self.game_lengths.append(new_length)
        avg_length = sum(self.game_lengths) / len(self.game_lengths)
        self.writer.add_scalar("Game Length/Average", avg_length, self.iteration)
        self.writer.add_histogram("Game Length/Histogram", np.array(self.game_lengths), self.iteration)

    def update_losses(self, total_loss, value_loss, policy_loss):
        self.loss_history["Total Loss"].append(total_loss)
        self.loss_history["Value Loss"].append(value_loss)
        self.loss_history["Policy Loss"].append(policy_loss)

        self.writer.add_scalar("Loss/Total", total_loss, self.iteration)
        self.writer.add_scalar("Loss/Value", value_loss, self.iteration)
        self.writer.add_scalar("Loss/Policy", policy_loss, self.iteration)

    def update_experience(self, new_elo):
        self.experience_length.append(new_elo)
        self.writer.add_scalar("Elo/Rating", new_elo, self.iteration)

    def update_exploration(self, exploration_value):
        self.exploration_history.append(exploration_value)
        self.writer.add_scalar("Exploration/Factor", exploration_value, self.iteration)

    def save_progress(self):
        data = {
            "win_counts": self.win_counts,
            "game_lengths": self.game_lengths,
            "loss_history": self.loss_history,
            "experience_length": self.experience_length,
            "exploration_history": self.exploration_history
        }
        os.makedirs(os.path.dirname(self.save_file), exist_ok=True)
        with open(self.save_file, "w") as f:
            json.dump(data, f, indent=4)

    def load_progress(self):
        try:
            with open(self.save_file, "r") as f:
                data = json.load(f)
                self.win_counts = data.get("win_counts", self.win_counts)
                self.game_lengths = data.get("game_lengths", self.game_lengths)
                self.loss_history = data.get("loss_history", self.loss_history)
                self.experience_length = data.get("experience_length", self.experience_length)
                self.exploration_history = data.get("exploration_history", self.exploration_history)
            print("Progress loaded.")
        except FileNotFoundError:
            print("No saved progress found.")

    def step(self):
        self.iteration += 1
