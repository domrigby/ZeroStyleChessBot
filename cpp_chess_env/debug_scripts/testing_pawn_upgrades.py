import chess_moves

def test_chess_engine():
    engine = chess_moves.ChessEngine()

    # Test setup position with pawns ready to promote
    fen = "8/P7/8/8/8/8/8/k6K w - - 0 1"
    engine.push(fen, "a7a8q")
    board = engine.get_board()
    print(board)
    assert board[0][0] == "Q", "Pawn should have promoted to a Queen."

    # Test move_to_target
    target_tensor = engine.move_to_target("a7a8q")
    assert target_tensor.shape == (70, 8, 8), "Target tensor should have shape (70, 8, 8)."
    assert target_tensor[66, 1, 0] == 1.0, "Promotion to Queen should be in the correct channel."

    # Test move_to_target_indices
    channel, from_row, from_col = engine.move_to_target_indices("a7a8q")
    assert channel == 66, "Channel for Queen promotion should be 66."
    assert from_row == 1 and from_col == 0, "Indices should correspond to the starting position of the pawn."

    # Test indices_to_move
    move = engine.indices_to_move(66, 1, 0)
    assert move == "a7a8q", f"Expected move 'a7a8q', but got '{move}'."

    # Test unflip_move
    flipped_move = engine.unflip_move("a7a8q")
    assert flipped_move == "h2h1q", f"Expected flipped move 'h2h1q', but got '{flipped_move}'."

    # Test generate_pawn_moves
    fen = "8/4P3/8/8/8/8/3P4/k6K w - - 0 1"
    engine.set_fen(fen)
    moves = engine.legal_moves()
    print(moves)
    assert "e7e8q" in moves and "e7e8r" in moves and "e7e8b" in moves and "e7e8n" in moves, \
        "Pawn promotion moves should be generated correctly."

    print("All tests passed!")

if __name__ == "__main__":
    test_chess_engine()
