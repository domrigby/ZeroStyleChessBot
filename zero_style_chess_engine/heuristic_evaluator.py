"""
A stand-in for the network evaluator, for a baseline MCTS player: uniform priors and a hand-written value function.
It answers in the searching thread, so there is no evaluator process and no network.
"""

import math
import queue
from typing import Any, Callable, Dict, List, Tuple

import chess
import numpy as np

from zero_style_chess_engine.config import HeuristicValue
from zero_style_chess_engine.messages import EvalRequest, EvalResult

PIECE_VALUES = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}
# Material lead mapped to a value in (-1, 1) as tanh(lead / MATERIAL_SCALE): a pawn up is ~0.1, a queen up ~0.7
MATERIAL_SCALE = 10.0


def material_value(board: chess.Board) -> float:
    """Material balance from the point of view of the side to move, squashed into (-1, 1)"""
    lead = sum(
        value * (len(board.pieces(piece, board.turn)) - len(board.pieces(piece, not board.turn)))
        for piece, value in PIECE_VALUES.items()
    )
    return math.tanh(lead / MATERIAL_SCALE)


# PeSTO's evaluation (Ronald Friederich, chessprogramming.org/PeSTO's_Evaluation_Function): material and
# piece-square tables, one set for the middlegame and one for the endgame, blended by how much material is left.
# Centipawns. Tables are from White's point of view, laid out a8..h8 first down to a1..h1, as printed
PESTO_MG_VALUES = {
    chess.PAWN: 82,
    chess.KNIGHT: 337,
    chess.BISHOP: 365,
    chess.ROOK: 477,
    chess.QUEEN: 1025,
    chess.KING: 0,
}
PESTO_EG_VALUES = {
    chess.PAWN: 94,
    chess.KNIGHT: 281,
    chess.BISHOP: 297,
    chess.ROOK: 512,
    chess.QUEEN: 936,
    chess.KING: 0,
}
# Phase weight of each piece: 24 with all of them on the board (middlegame), 0 with only pawns and kings (endgame)
PESTO_PHASE = {chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 1, chess.ROOK: 2, chess.QUEEN: 4, chess.KING: 0}
PESTO_MAX_PHASE = 24

# fmt: off
_PESTO_MG_TABLES = {
    chess.PAWN: [
          0,   0,   0,   0,   0,   0,  0,   0,
         98, 134,  61,  95,  68, 126, 34, -11,
         -6,   7,  26,  31,  65,  56, 25, -20,
        -14,  13,   6,  21,  23,  12, 17, -23,
        -27,  -2,  -5,  12,  17,   6, 10, -25,
        -26,  -4,  -4, -10,   3,   3, 33, -12,
        -35,  -1, -20, -23, -15,  24, 38, -22,
          0,   0,   0,   0,   0,   0,  0,   0,
    ],
    chess.KNIGHT: [
        -167, -89, -34, -49,  61, -97, -15, -107,
         -73, -41,  72,  36,  23,  62,   7,  -17,
         -47,  60,  37,  65,  84, 129,  73,   44,
          -9,  17,  19,  53,  37,  69,  18,   22,
         -13,   4,  16,  13,  28,  19,  21,   -8,
         -23,  -9,  12,  10,  19,  17,  25,  -16,
         -29, -53, -12,  -3,  -1,  18, -14,  -19,
        -105, -21, -58, -33, -17, -28, -19,  -23,
    ],
    chess.BISHOP: [
        -29,   4, -82, -37, -25, -42,   7,  -8,
        -26,  16, -18, -13,  30,  59,  18, -47,
        -16,  37,  43,  40,  35,  50,  37,  -2,
         -4,   5,  19,  50,  37,  37,   7,  -2,
         -6,  13,  13,  26,  34,  12,  10,   4,
          0,  15,  15,  15,  14,  27,  18,  10,
          4,  15,  16,   0,   7,  21,  33,   1,
        -33,  -3, -14, -21, -13, -12, -39, -21,
    ],
    chess.ROOK: [
         32,  42,  32,  51, 63,  9,  31,  43,
         27,  32,  58,  62, 80, 67,  26,  44,
         -5,  19,  26,  36, 17, 45,  61,  16,
        -24, -11,   7,  26, 24, 35,  -8, -20,
        -36, -26, -12,  -1,  9, -7,   6, -23,
        -45, -25, -16, -17,  3,  0,  -5, -33,
        -44, -16, -20,  -9, -1, 11,  -6, -71,
        -19, -13,   1,  17, 16,  7, -37, -26,
    ],
    chess.QUEEN: [
        -28,   0,  29,  12,  59,  44,  43,  45,
        -24, -39,  -5,   1, -16,  57,  28,  54,
        -13, -17,   7,   8,  29,  56,  47,  57,
        -27, -27, -16, -16,  -1,  17,  -2,   1,
         -9, -26,  -9, -10,  -2,  -4,   3,  -3,
        -14,   2, -11,  -2,  -5,   2,  14,   5,
        -35,  -8,  11,   2,   8,  15,  -3,   1,
         -1, -18,  -9,  10, -15, -25, -31, -50,
    ],
    chess.KING: [
        -65,  23,  16, -15, -56, -34,   2,  13,
         29,  -1, -20,  -7,  -8,  -4, -38, -29,
         -9,  24,   2, -16, -20,   6,  22, -22,
        -17, -20, -12, -27, -30, -25, -14, -36,
        -49,  -1, -27, -39, -46, -44, -33, -51,
        -14, -14, -22, -46, -44, -30, -15, -27,
          1,   7,  -8, -64, -43, -16,   9,   8,
        -15,  36,  12, -54,   8, -28,  24,  14,
    ],
}
_PESTO_EG_TABLES = {
    chess.PAWN: [
          0,   0,   0,   0,   0,   0,   0,   0,
        178, 173, 158, 134, 147, 132, 165, 187,
         94, 100,  85,  67,  56,  53,  82,  84,
         32,  24,  13,   5,  -2,   4,  17,  17,
         13,   9,  -3,  -7,  -7,  -8,   3,  -1,
          4,   7,  -6,   1,   0,  -5,  -1,  -8,
         13,   8,   8,  10,  13,   0,   2,  -7,
          0,   0,   0,   0,   0,   0,   0,   0,
    ],
    chess.KNIGHT: [
        -58, -38, -13, -28, -31, -27, -63, -99,
        -25,  -8, -25,  -2,  -9, -25, -24, -52,
        -24, -20,  10,   9,  -1,  -9, -19, -41,
        -17,   3,  22,  22,  22,  11,   8, -18,
        -18,  -6,  16,  25,  16,  17,   4, -18,
        -23,  -3,  -1,  15,  10,  -3, -20, -22,
        -42, -20, -10,  -5,  -2, -20, -23, -44,
        -29, -51, -23, -15, -22, -18, -50, -64,
    ],
    chess.BISHOP: [
        -14, -21, -11,  -8, -7,  -9, -17, -24,
         -8,  -4,   7, -12, -3, -13,  -4, -14,
          2,  -8,   0,  -1, -2,   6,   0,   4,
         -3,   9,  12,   9, 14,  10,   3,   2,
         -6,   3,  13,  19,  7,  10,  -3,  -9,
        -12,  -3,   8,  10, 13,   3,  -7, -15,
        -14, -18,  -7,  -1,  4,  -9, -15, -27,
        -23,  -9, -23,  -5, -9, -16,  -5, -17,
    ],
    chess.ROOK: [
        13, 10, 18, 15, 12,  12,   8,   5,
        11, 13, 13, 11, -3,   3,   8,   3,
         7,  7,  7,  5,  4,  -3,  -5,  -3,
         4,  3, 13,  1,  2,   1,  -1,   2,
         3,  5,  8,  4, -5,  -6,  -8, -11,
        -4,  0, -5, -1, -7, -12,  -8, -16,
        -6, -6,  0,  2, -9,  -9, -11,  -3,
        -9,  2,  3, -1, -5, -13,   4, -20,
    ],
    chess.QUEEN: [
         -9,  22,  22,  27,  27,  19,  10,  20,
        -17,  20,  32,  41,  58,  25,  30,   0,
        -20,   6,   9,  49,  47,  35,  19,   9,
          3,  22,  24,  45,  57,  40,  57,  36,
        -18,  28,  19,  47,  31,  34,  39,  23,
        -16, -27,  15,   6,   9,  17,  10,   5,
        -22, -23, -30, -16, -16, -23, -36, -32,
        -33, -28, -22, -43,  -5, -32, -20, -41,
    ],
    chess.KING: [
        -74, -35, -18, -18, -11,  15,   4, -17,
        -12,  17,  14,  17,  17,  38,  23,  11,
         10,  17,  23,  15,  20,  45,  44,  13,
         -8,  22,  24,  27,  26,  33,  26,   3,
        -18,  -4,  21,  24,  27,  23,   9, -11,
        -19,  -3,  11,  21,  23,  16,   7,  -9,
        -27, -11,   4,  13,  14,   4,  -5, -17,
        -53, -34, -21, -11, -28, -14, -24, -43,
    ],
}
# fmt: on


def _square_scores(values: Dict[int, int], tables: Dict[int, List[int]]) -> Dict[Tuple[int, bool], List[int]]:
    """Piece value plus table entry, indexed by python-chess square (a1 = 0). A table's index of a1 is 56, so White's
    square is flipped vertically (sq ^ 56); Black's is the mirror image, so its square is used as is"""
    return {
        (piece, color): [
            values[piece] + tables[piece][square ^ 56 if color == chess.WHITE else square] for square in chess.SQUARES
        ]
        for piece in tables
        for color in chess.COLORS
    }


_MG_SCORES = _square_scores(PESTO_MG_VALUES, _PESTO_MG_TABLES)
_EG_SCORES = _square_scores(PESTO_EG_VALUES, _PESTO_EG_TABLES)


def pesto_eval(board: chess.Board) -> int:
    """PeSTO's static evaluation in centipawns, from the point of view of the side to move"""
    mg = eg = phase = 0
    for square, piece in board.piece_map().items():
        sign = 1 if piece.color == board.turn else -1
        mg += sign * _MG_SCORES[piece.piece_type, piece.color][square]
        eg += sign * _EG_SCORES[piece.piece_type, piece.color][square]
        phase += PESTO_PHASE[piece.piece_type]
    phase = min(phase, PESTO_MAX_PHASE)  # Early promotions can push it past 24
    return (mg * phase + eg * (PESTO_MAX_PHASE - phase)) // PESTO_MAX_PHASE


# Captures (and check evasions) searched past a leaf before it is scored, so a position mid-exchange isn't scored as
# if the exchange were over. Capped, as check evasions are not captures and can go on
QUIESCENCE_DEPTH = 6
MATE_CP = 100_000
# Centipawns mapped to a value in (-1, 1) as tanh(cp / VALUE_SCALE_CP): the same curve as the common expected score
# 1 / (1 + 10^(-cp / 400)) rescaled to (-1, 1). A pawn up is ~0.27, a minor piece up ~0.7
VALUE_SCALE_CP = 400 / math.log(10) * 2


def _mvv_lva(board: chess.Board, move: chess.Move) -> int:
    """Most valuable victim first, then least valuable attacker; en passant takes a pawn the target square lacks"""
    victim = board.piece_type_at(move.to_square) or chess.PAWN
    attacker = board.piece_type_at(move.from_square) or chess.PAWN
    return PESTO_MG_VALUES[victim] * 10 - PESTO_MG_VALUES[attacker]


def quiescence(board: chess.Board, alpha: int, beta: int, depth: int = QUIESCENCE_DEPTH) -> int:
    """
    Fail-soft alpha-beta over captures only, from the side to move's point of view, in centipawns. Out of check the
    side to move may also stand pat (decline every capture and keep the static score); in check it must answer it, so
    every legal move is searched and having none is mate. Pushes and pops moves on board, leaving it as it was
    """
    in_check = board.is_check()
    if depth == 0:
        return pesto_eval(board)
    if in_check:
        best = -MATE_CP
        moves = list(board.legal_moves)
    else:
        best = pesto_eval(board)
        if best >= beta:
            return best
        alpha = max(alpha, best)
        moves = list(board.generate_legal_captures())
    moves.sort(key=lambda move: _mvv_lva(board, move) if board.is_capture(move) else -MATE_CP, reverse=True)
    for move in moves:
        board.push(move)
        score = -quiescence(board, -beta, -alpha, depth - 1)
        board.pop()
        if score > best:
            best = score
            if score >= beta:
                break
            alpha = max(alpha, score)
    return best


def pesto_value(board: chess.Board) -> float:
    """PeSTO's evaluation after the captures settle, from the side to move's point of view, squashed into (-1, 1)"""
    return math.tanh(quiescence(board, -MATE_CP, MATE_CP) / VALUE_SCALE_CP)


# Values are from the side to move's point of view, as the network's are
VALUE_FUNCTIONS: Dict[HeuristicValue, Callable[[chess.Board], float]] = {
    "zero": lambda board: 0.0,
    "material": material_value,
    "pesto": pesto_value,
}


class HeuristicEvaluator:
    """
    Stands in for the evaluator's request queue (MCTS only calls put on it): answers every request at once, putting
    the result on the search's results queue, where the search picks it up on its next poll
    """

    def __init__(self, results_queue: "queue.Queue[Any]", value_function: HeuristicValue) -> None:
        self.results_queue = results_queue
        self.value_function = VALUE_FUNCTIONS[value_function]

    def put(self, request: EvalRequest) -> None:
        num_moves = len(request.legal_moves)
        uniform_priors = np.full(num_moves, 1.0 / num_moves) if num_moves else np.zeros(0)
        value = self.value_function(chess.Board(request.fen))
        self.results_queue.put([EvalResult(request.node_id, value, uniform_priors)])
