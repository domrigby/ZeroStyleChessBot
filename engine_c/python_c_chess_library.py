import chess_moves
import chess

print(dir(chess_moves))

fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
obj = chess_moves.ChessEngine()

board = chess.Board()

import time
start_time = time.time()
obj.set_fen(fen)
moves = obj.legal_moves()
end_time = time.time()

print(end_time - start_time)
