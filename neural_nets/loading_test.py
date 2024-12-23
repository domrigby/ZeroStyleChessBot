import chess.pgn
import io

def extract_games_from_lines(pgn_file_path, start_line, end_line):
    game_strings = []
    current_lines = []

    with open(pgn_file_path, 'r') as file:
        for i, line in enumerate(file):
            if i < start_line:
                continue
            if i >= end_line:
                break
            current_lines.append(line)
            # End of a game in PGN is marked by an empty line
            if line.strip() == "":
                game_text = "".join(current_lines)
                current_lines = []
                # Parse the game
                game = chess.pgn.read_game(io.StringIO(game_text))
                if game:
                    moves = []
                    board = game.board()
                    for move in game.mainline_moves():
                        moves.append(board.san(move))
                        board.push(move)
                    game_strings.append(" ".join(moves))

    return game_strings

# Example usage
pgn_path = "lichess_games.pgn"
start_line = 100000  # Line number to start
end_line = 200000    # Line number to stop
game_moves = extract_games_from_lines(pgn_path, start_line, end_line)

# Print the first few games
for i, game in enumerate(game_moves[:5]):
    print(f"Game {i+1}: {game}\n")
