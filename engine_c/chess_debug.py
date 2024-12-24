import chess_moves  # Import the Pybind11 module

def test_chess_engine():
    # Create an instance of the ChessEngine
    engine = chess_moves.ChessEngine()

    # Test FEN Parsing
    print("=== Test FEN Parsing ===")
    starting_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    engine.set_fen(starting_fen)
    print("Starting position set.")

    # Check parsed board state
    print("Parsed board state:")
    board = engine.get_board()
    for row in range(8):
        print("".join(board[row]))

    # Test Move Generation
    print("\n=== Test Move Generation ===")
    legal_moves = engine.legal_moves()
    print("Legal moves from the starting position:")
    print(legal_moves)
    assert "e2e4" in legal_moves and "b1c3" in legal_moves, "Expected starting moves missing!"

    # Test Pawn Logic (forward and diagonal)
    print("\n=== Test Pawn Logic ===")
    pawn_fen = "8/8/3p4/4P3/8/8/8/8 w - - 0 1"  # White pawn can capture d5 or move to e6
    engine.set_fen(pawn_fen)
    pawn_moves = engine.legal_moves()
    board = engine.get_board()
    for row in range(8):
        print("".join(board[row]))
    print("Legal moves for white pawn at e5:")
    print(pawn_moves)
    assert "e5e6" in pawn_moves and "e5d6" in pawn_moves, "Pawn moves not generated correctly!"

    # Test Move Application (push)
    print("\n=== Test Move Application ===")
    move = "e2e4"
    new_fen = engine.push(starting_fen, move)
    print(f"After move {move}, new FEN:")
    print(new_fen)
    board = engine.get_board()
    for row in range(8):
        print("".join(board[row]))
    # assert new_fen.startswith("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3"), "FEN after move is incorrect!"

    # Test Game Continuation
    print("\n=== Test Game Continuation ===")
    game_over = engine.is_game_over()
    print("Is the game over?", game_over)
    assert not game_over, "Game should not be over at the start!"

    engine = chess_moves.ChessEngine()
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    tensor = engine.fen_to_tensor(fen)
    print(tensor.shape)  # Should print (12, 8, 8)
    print(tensor)  # View the tensor

    engine = chess_moves.ChessEngine()
    move = "b2b7"
    target = engine.move_to_target(move)
    print(target.shape)  # Should print (64, 8, 8)
    print(target)  # View the one-hot encoded target tensor
    import numpy as np
    print(np.where(target==1))

    # Create an instance of the ChessEngine
    engine = chess_moves.ChessEngine()
    # Example move string
    move = "g1f3"  # Knight move from g1 to f3
    # Call the move_to_target_indices function
    channel, row, col = engine.move_to_target_indices(move)
    # Print the result
    print(f"Move: {move}")
    print(f"Channel: {channel}, Row: {row}, Col: {col}")

    # Define a sequence of moves
    moves = ["e2e4", "e7e5", "g1f3", "b8c6"]  # Standard opening moves

    # Get the board tensor for these moves, assuming white is playing
    white_to_play = False
    board_tensor = engine.moves_to_board_tensor(moves, white_to_play)

    # Convert the tensor to a NumPy array for inspection
    board_tensor_np = np.array(board_tensor)

    # Print details about the tensor
    print("Shape of board tensor:", board_tensor_np.shape)  # Should be (12, 8, 8)
    print("Board tensor (first channel):\n", board_tensor_np[0])  # Example: First
    print("All channels:\n", np.argmax(board_tensor_np, 0))

    print("\nAll tests passed!")

if __name__ == "__main__":
    test_chess_engine()
