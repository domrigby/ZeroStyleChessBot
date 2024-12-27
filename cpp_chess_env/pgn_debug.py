import chess_moves

# Initialize the ChessEngine
engine = chess_moves.ChessEngine()

# Example PGN string
pgn = "1. e4 e5 2. Nf3 Nc6 3. Bc4 Nf6 4. O-O O-O 5. d3"

# Convert PGN to long algebraic moves
moves = engine.pgn_to_moves(pgn)
print("PGN converted to long algebraic moves:")
for move in moves:
    print(move)

# Apply the moves to the board and get the final FEN
starting_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
current_fen = starting_fen
for move in moves:
    current_fen = engine.make_move(current_fen, move)
    print("FEN after move", move, ":", current_fen)

# Get the board tensor for the final state
tensor = engine.fen_to_tensor(current_fen)