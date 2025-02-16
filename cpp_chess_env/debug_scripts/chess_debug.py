import chess_moves  # Import the Pybind11 module
import numpy as np

def test_chess_engine():
    # Create an instance of the ChessEngine
    engine = chess_moves.ChessEngine()

    # === Test FEN Parsing ===
    print("=== Test FEN Parsing ===")
    # starting_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    starting_fen = "8/8/1N6/4p3/3P4/8/8/K7 w KQkq - 0 1"
    engine.set_fen(starting_fen)
    print("Starting position set.")
    board = engine.get_board()  # Expected to be an 8x8 list-of-lists of chars
    print("Parsed board state:")
    for row in range(8):
        print("".join(board[row]))

    # === Test Move Generation ===
    print("\n=== Test Move Generation ===")
    starting_fen = "8/8/1N6/4p3/3P4/8/8/K7 b KQkq - 0 1"
    engine.set_fen(starting_fen)
    legal_moves = engine.legal_moves()
    print("Legal moves from the starting position:")
    print(legal_moves)
    assert "e5d4" in legal_moves, "Expected starting moves missing!"
    print("Starting position set.")
    board = engine.get_board()  # Expected to be an 8x8 list-of-lists of chars
    print("Parsed board state:")
    for row in range(8):
        print("".join(board[row]))

    # === Test Move Generation ===
    print("\n=== Test Move Generation ===")
    starting_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    engine.set_fen(starting_fen)
    legal_moves = engine.legal_moves()
    print("Legal moves from the starting position:")
    print(legal_moves)
    assert "e2e4" in legal_moves and "b1c3" in legal_moves, "Expected starting moves missing!"

    # === Test Pawn Logic (forward and diagonal) ===
    print("\n=== Test Pawn Logic ===")
    # A position where a white pawn on e5 can move forward or capture diagonally.
    pawn_fen = "8/8/3p4/4P3/8/8/8/8 w - - 0 1"
    engine.set_fen(pawn_fen)
    pawn_moves = engine.legal_moves()
    board = engine.get_board()
    print("Board state for pawn test:")
    for row in range(8):
        print("".join(board[row]))
    print("Legal moves for white pawn:")
    print(pawn_moves)
    assert "e5d6" in pawn_moves and "e5d6" in pawn_moves, "Pawn moves not generated correctly!"

    # === Test Move Application (push) ===
    print("\n=== Test Move Application ===")
    move = "e5d6"
    new_fen = engine.push(pawn_fen, move)
    print(f"After move {move}, new FEN:")
    print(new_fen)
    board = engine.get_board()
    print("Board after move:")
    for row in range(8):
        print("".join(board[row]))
    # (Optional: add an assert to compare with an expected FEN string)

    # === Test Game Continuation ===
    print("\n=== Test Game Continuation ===")
    engine.set_fen("8/8/8/8/8/8/4r3/K4r2 w - - 0 1")
    game_over, status = engine.is_game_over()
    print("Is the game over?", game_over, status)
    assert game_over, "Should be checkmate!"

    # === Test FEN to Tensor Conversion ===
    print("\n=== Test FEN to Tensor ===")
    tensor = engine.fen_to_tensor(starting_fen)
    print("Tensor shape:", tensor.shape)  # Expected shape: (12, 8, 8)
    print("Tensor:", tensor)

    # === Test Move to Target Tensor ===
    print("\n=== Test Move to Target Tensor ===")
    move = "b2b8"
    target = engine.move_to_target(move)
    print("Target tensor shape:", target.shape)  # Expected shape: (70, 8, 8) or as defined in your engine
    print("Target tensor:", target)
    target_np = np.array(target)
    print("Indices where target==1:", np.where(target_np == 1))

    # === Test Move to Target Indices ===
    print("\n=== Test Move to Target Indices ===")
    move = "g1f3"  # Knight move
    channel, row, col = engine.move_to_target_indices(move)
    print(f"Move: {move}")
    print(f"Channel: {channel}, Row: {row}, Col: {col}")

    # === Test Moves to Board Tensor ===
    print("\n=== Test Moves to Board Tensor ===")
    moves = ["e2e4", "e7e5", "g1f3", "b8c6"]  # Standard opening moves
    # Here we assume white_to_play flag as given (False means black is "friendly" from the tensor's perspective)
    white_to_play = False
    board_tensor = engine.moves_to_board_tensor(moves, white_to_play)
    board_tensor_np = np.array(board_tensor)
    print("Shape of board tensor:", board_tensor_np.shape)  # Expected shape: (12, 8, 8)
    print("Board tensor (first channel):\n", board_tensor_np[0])
    print("Argmax over channels (board positions):\n", np.argmax(board_tensor_np, axis=0))

    print("\nAll tests passed!")

if __name__ == "__main__":
    test_chess_engine()
