import chess.svg
import chess
from typing import Optional


def save_chess_board_as_im(board: chess.Board, file_name: str, move: Optional[chess.Move] = None) -> None:

    if move:
        svg = chess.svg.board(
            board=board, arrows=[chess.svg.Arrow(move.from_square, move.to_square, color="#0000cccc")]
        )
    else:
        svg = chess.svg.board(board=board)

    with open(file_name, "w") as file:
        file.write(svg)
