import matplotlib.pyplot as plt
import numpy as np
import json
import os

class ChessEngineMonitor:
    def __init__(self, save_dir=""):
        plt.ion()  # Interactive mode on
        self.fig, self.axs = plt.subplots(2, 3, figsize=(15, 10))  # 5 subplots in a 2x3 grid
        self.axs = self.axs.flatten()
        self.save_file = save_dir + "/chess_monitor_data.json"

        # Data placeholders
        self.win_counts = {"White Wins": 0, "Black Wins": 0, "Draws": 0}
        self.game_lengths = []
        self.loss_history = {"Total Loss": [], "Value Loss": [], "Policy Loss": []}
        self.experience_length = []
        self.exploration_history = []

        if os.path.exists(self.save_file):
            # If file exists then replace it
            self.load_progress()

        # Initialize plots
        self._initialize_plots()

    def _initialize_plots(self):
        plt.style.use("ggplot")

        # Win distribution
        self.axs[0].bar(self.win_counts.keys(), self.win_counts.values(), color=['white', 'black', 'gray'])
        self.axs[0].set_title("Win Distribution")
        self.axs[0].set_ylim(0, 1)

        # Game lengths
        self.axs[1].hist([], bins=20, alpha=0.75, color="blue")
        self.axs[1].set_title("Game Length Distribution")
        self.axs[1].set_xlabel("Moves per Game")
        self.axs[1].set_ylabel("Frequency")

        # Losses
        self.axs[2].plot([], label="Total Loss", color="red")
        self.axs[2].plot([], label="Value Loss", color="green")
        self.axs[2].plot([], label="Policy Loss", color="blue")
        self.axs[2].legend()
        self.axs[2].set_title("Loss Over Time")
        self.axs[2].set_xlabel("Training Iteration")
        self.axs[2].set_ylabel("Loss")

        # Elo Ratings
        self.axs[3].plot([], label="Elo Rating", color="purple")
        self.axs[3].set_title("Elo Rating Progression")
        self.axs[3].set_xlabel("Training Iteration")
        self.axs[3].set_ylabel("Elo Rating")

        # Exploration factor
        self.axs[4].plot([], label="Exploration Factor", color="orange")
        self.axs[4].set_title("Exploration Parameter Over Time")
        self.axs[4].set_xlabel("Training Iteration")
        self.axs[4].set_ylabel("Exploration Value")

        plt.tight_layout()

    def update_wins(self, white_win):
        if white_win:
            self.win_counts["White Wins"] += 1
        elif white_win is not None:
            self.win_counts["Black Wins"] += 1
        else:
            self.win_counts["Draws"] += 1

        self.axs[0].cla()
        self.axs[0].bar(self.win_counts.keys(), self.win_counts.values(), color=['blue', 'red', 'green'])
        self.axs[0].set_title("Win Distribution")
        self.axs[0].set_ylim(0, max(self.win_counts.values()) * 1.1)

    def update_game_lengths(self, new_lengths):
        self.game_lengths.append(new_lengths)
        self.axs[1].cla()
        self.axs[1].hist(self.game_lengths, bins=20, alpha=0.75, color="blue")
        self.axs[1].set_title("Game Length Distribution")
        self.axs[1].set_xlabel("Moves per Game")
        self.axs[1].set_ylabel("Frequency")

    def update_losses(self, total_loss, value_loss, policy_loss):
        self.loss_history["Total Loss"].append(total_loss)
        self.loss_history["Value Loss"].append(value_loss)
        self.loss_history["Policy Loss"].append(policy_loss)

        self.axs[2].cla()
        self.axs[2].plot(self.loss_history["Total Loss"], label="Total Loss", color="red")
        self.axs[2].plot(self.loss_history["Value Loss"], label="Value Loss", color="green")
        self.axs[2].plot(self.loss_history["Policy Loss"], label="Policy Loss", color="blue")
        self.axs[2].legend()
        self.axs[2].set_title("Loss Over Time")
        self.axs[2].set_xlabel("Training Iteration")
        self.axs[2].set_ylabel("Loss")

    def update_experience(self, new_elo):
        self.experience_length.append(new_elo)
        self.axs[3].cla()
        self.axs[3].plot(self.experience_length, label="Elo Rating", color="purple")
        self.axs[3].set_title("Experience Queue")
        self.axs[3].set_xlabel("Training Iteration")
        self.axs[3].set_ylabel("Experience Length")

    def update_exploration(self, exploration_value):
        self.exploration_history.append(exploration_value)
        self.axs[4].cla()
        self.axs[4].plot(self.exploration_history, label="Exploration Factor", color="orange")
        self.axs[4].set_title("Exploration Parameter Over Time")
        self.axs[4].set_xlabel("Training Iteration")
        self.axs[4].set_ylabel("Exploration Value")

    def save_progress(self):
        data = {
            "win_counts": self.win_counts,
            "game_lengths": self.game_lengths,
            "loss_history": self.loss_history,
            "experience_length": self.experience_length,
            "exploration_history": self.exploration_history
        }
        with open(self.save_file, "w") as f:
            json.dump(data, f, indent=4)
        print("Progress saved.")

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

    def refresh(self):
        plt.pause(0.1)
