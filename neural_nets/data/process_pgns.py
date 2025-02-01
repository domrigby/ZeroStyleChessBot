import chess.pgn
from enum import Enum
from typing import List
import glob
import os
import pickle

class WinTypes(Enum):
    TIME = "time"

class PGNLoader:

    def __init__(self, pgn_dir: str, rejected_win_types: List[WinTypes] = None):

        if rejected_win_types is None:
            rejected_win_types = []

        self.rejected_win_types = rejected_win_types

        self.pgn_dir = pgn_dir
        self.pgns = glob.glob(pgn_dir + "/*.pgn")

        self.game_count = 0

        for pgn in self.pgns:
            self.load_pgn(pgn)

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

                winner = game.headers.get("Result", "*")
                game_win_type = game.headers.get("Termination", "*")

                for win_type in self.rejected_win_types:
                    if win_type.value in game_win_type:
                        continue

                if winner == "1-0":
                    winner = "white"
                elif winner == "0-1":
                    winner = "black"
                else:
                    winner = "draw"

                self.pickle_moves(moves, winner)

                self.game_count += 1

    def pickle_moves(self, moves, winner):
        """
        Precompute FEN states and moves for a single game and add to the dataset.
        """
        board = chess.Board()

        if len(moves) < 5:
            # Skip this game
            return

        for idx, move in enumerate(moves):

            fen = board.fen()

            if not os.path.exists(self.pgn_dir + f"/pickles/{idx}"):
                os.makedirs(self.pgn_dir + f"/pickles/{idx}")

            try:

                move = board.parse_san(move)

                # Get legal moves for the current FEN state
                legal_moves = board.legal_moves

                # Update the board for the next move
                board.push(move)

                if len(list(legal_moves)) == 0:
                    print(f"Game state {fen} with no legal moves.")
                    continue

                with open(self.pgn_dir + f"/pickles/{idx}/game_{self.game_count}.pkl", "wb") as f:
                    move_dict = {'fen': fen, 'move': str(move), 'legal_moves': legal_moves, 'winner': winner, 'turn_num': idx}
                    pickle.dump(move_dict, f)

            except Exception as e:
                # Skip invalid moves
                print(f"Invalid move: {move} due to {e}")
                break

if __name__ == "__main__":
    pgn_loader = PGNLoader(r"/home/dom/Code/chess_bot/neural_nets/data", rejected_win_types=[WinTypes.TIME])
    print(pgn_loader.pgns)