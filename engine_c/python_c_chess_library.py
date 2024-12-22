import chess_moves

print(dir(chess_moves))

fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
obj = chess_moves.ChessEngine()

obj.set_position(fen)

print(obj.get_legal_moves())
