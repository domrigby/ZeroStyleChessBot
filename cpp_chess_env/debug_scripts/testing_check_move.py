import chess_moves

import chess_moves

def test_blocking_check():
    engine = chess_moves.ChessEngine()

    # Set up a position where capturing the attacker is possible
    # Example FEN: Black bishop on b4 gives check to white king on d1,
    # and white rook or knight can capture the bishop.
    fen = "3k4/8/8/8/1B1r4/8/8/3K4 w - - 0 1"
    engine.set_fen(fen)

    print("Testing capturing the attacker...")
    legal_moves = engine.legal_moves()
    print("Legal moves:", legal_moves)

    # Expect legal moves that capture the attacking bishop
    # Expected moves: Nd3xb4 (knight captures bishop), Rc1xb4 (rook captures bishop)
    expected_moves = ["b4d2"]
    assert any(move in legal_moves for move in expected_moves)

    print("Capture attacker test passed!\n")

def test_capture_attacker():
    engine = chess_moves.ChessEngine()

    # Set up a position where capturing the attacker is possible
    # Example FEN: Black bishop on b4 gives check to white king on d1,
    # and white rook or knight can capture the bishop.
    fen = "3k4/8/8/1B6/8/8/8/3K1r2 w - - 0 1"
    engine.set_fen(fen)

    print("Testing capturing the attacker...")
    legal_moves = engine.legal_moves()
    print("Legal moves:", legal_moves)

    # Expect legal moves that capture the attacking bishop
    # Expected moves: Nd3xb4 (knight captures bishop), Rc1xb4 (rook captures bishop)
    expected_moves = ["b5f1"]
    assert any(move in legal_moves for move in expected_moves)

    print("Capture attacker test passed!\n")


if __name__ == "__main__":
    test_blocking_check()
    test_capture_attacker()
    print("All tests passed!")


