import chess_moves

def game_string_to_nn_input(game_string):

    chess_engine = chess_moves.ChessEngine()
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

    for move in game_string:
        fen = chess_engine.make_move(fen, move)

    return chess_moves.fen_to_tensor(fen)

def move_to_tensor(move):
    pass