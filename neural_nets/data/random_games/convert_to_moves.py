import chess
import chess.pgn
import random
import pickle
import sys
import os

# Number of moves per file
SPLIT_SIZE = 250000

def flush_dataset(dataset, prefix, file_counter):
    """
    Dumps moves in chunks of SPLIT_SIZE from dataset into pickle files.
    Returns the updated dataset (with flushed moves removed) and next file counter.
    """
    while len(dataset) >= SPLIT_SIZE:
        to_dump = dataset[:SPLIT_SIZE]
        filename = f"{prefix}_moves_{file_counter}.pkl"
        with open(filename, "wb") as f:
            pickle.dump(to_dump, f)
        print(f"\nSaved {SPLIT_SIZE} {prefix} moves to {filename}")
        dataset = dataset[SPLIT_SIZE:]
        file_counter += 1
    return dataset, file_counter


def process_pgn(pgn_file_path):
    """
    Processes the PGN file and for each game saves one white move and one black move.
    Each move dictionary contains:
       'fen'   : FEN string of the board before the move,
       'moves' : move in UCI (square-to-square) format (e.g. "a2a4"),
       'value' : game outcome from the perspective of the moving side:
                 1 if that player eventually wins,
                -1 if they lose,
                 0 if drawn.
    Each game is randomly assigned to either the training (90%) or test (10%) dataset.
    Moves are saved to files of 100,000 samples.
    """
    train_moves = []
    test_moves = []
    train_file_count = 0
    test_file_count = 0

    total_size = os.path.getsize(pgn_file_path)
    with open(pgn_file_path, "r", encoding="utf-8") as pgn:
        game_number = 0
        while True:
            game = chess.pgn.read_game(pgn)
            if game is None:
                break
            game_number += 1

            # Determine winning side from header (or draw)
            result = game.headers.get("Result", "")
            if result == "1-0":
                winning_side = chess.WHITE
            elif result == "0-1":
                winning_side = chess.BLACK
            else:
                winning_side = None  # draw

            # We'll collect candidate moves for white and black.
            white_candidates = []
            black_candidates = []
            board = game.board()

            # Iterate through moves while updating board state.
            for move in game.mainline_moves():
                # Record candidate move before applying it.
                current_turn = board.turn
                fen = board.fen()
                move_str = move.uci()
                if current_turn == chess.WHITE:
                    # For white move: value is 1 if white wins, -1 if white loses, 0 for draw.
                    value = 0 if winning_side is None else (1 if winning_side == chess.WHITE else -1)
                    white_candidates.append({"fen": fen, "moves": move_str, "value": value})
                else:  # black move
                    value = 0 if winning_side is None else (1 if winning_side == chess.BLACK else -1)
                    black_candidates.append({"fen": fen, "moves": move_str, "value": value})
                board.push(move)

            # Only add moves if both colors have at least one move.
            if white_candidates and black_candidates:
                chosen_white = random.choice(white_candidates)
                chosen_black = random.choice(black_candidates)

                # Assign both moves to training or testing set for this game.
                if random.random() < 0.9:
                    train_moves.append(chosen_white)
                    train_moves.append(chosen_black)
                else:
                    test_moves.append(chosen_white)
                    test_moves.append(chosen_black)
            else:
                print(f"\rSkipping game {game_number} due to missing white or black moves.", end="")

            # Flush datasets if size exceeds threshold.
            train_moves, train_file_count = flush_dataset(train_moves, "train", train_file_count)
            test_moves, test_file_count = flush_dataset(test_moves, "test", test_file_count)

            # Print progress update based on file position.
            progress = (pgn.tell() / total_size) * 100
            print(f"\rProcessed {game_number} games. Progress: {progress:.2f}% complete", end="")

    # Flush any remaining moves.
    if train_moves:
        filename = f"train_moves_{train_file_count}.pkl"
        with open(filename, "wb") as f:
            pickle.dump(train_moves, f)
        print(f"\nSaved final {len(train_moves)} train moves to {filename}")
    if test_moves:
        filename = f"test_moves_{test_file_count}.pkl"
        with open(filename, "wb") as f:
            pickle.dump(test_moves, f)
        print(f"\nSaved final {len(test_moves)} test moves to {filename}")

    print("\nFinished processing PGN.")
    return


if __name__ == "__main__":
    process_pgn("/home/dom/Code/chess_bot/neural_nets/data/random_games/lichess_db_standard_rated_2017-02.pgn")
