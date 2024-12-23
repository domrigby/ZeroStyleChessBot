import chess
import chess_moves
import time

board = chess_moves.ChessEngine()
fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

start_time = time.time()
for _ in range(25000):
    board.set_fen(fen)
    moves = board.legal_moves()
    board.push(fen, str(moves[0]))
end_time = time.time()

print(f"C++ engine took {end_time-start_time} seconds")

board = chess.Board()
fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

start_time = time.time()
for _ in range(25000):
    board.set_fen(fen)
    moves = board.legal_moves
    board.push(list(moves)[0])
end_time = time.time()

print(f"Python engine took {end_time-start_time} seconds")
