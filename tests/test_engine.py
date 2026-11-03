"""Checks migrated from cpp_chess_env/debug_scripts: move legality, promotions and the move <-> index mapping."""

import random
from typing import List

import chess
import chess_moves
import pytest

START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
ITALIAN_FEN = "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4"
PROMOTION_PIECES = ["q", "r", "b", "n"]
FIRST_PROMOTION_CHANNEL = 66
RANDOM_GAMES = 20
PLIES_PER_GAME = 80


def _legal_moves(fen: str) -> List[str]:
    engine = chess_moves.ChessEngine()
    engine.set_fen(fen)
    return engine.legal_moves()


@pytest.mark.parametrize(
    "fen, expected_moves",
    [
        pytest.param(START_FEN, ["e2e4", "b1c3"], id="start_position"),
        pytest.param("k7/8/3p4/4P3/8/8/8/K7 w - - 0 1", ["e5e6", "e5d6"], id="white_pawn_push_and_capture"),
        pytest.param("3k4/8/8/8/1B1r4/8/8/3K4 w - - 0 1", ["b4d2"], id="block_check"),
        pytest.param("3k4/8/8/1B6/8/8/8/3K1r2 w - - 0 1", ["b5f1"], id="capture_checking_piece"),
    ],
)
def test_expected_moves_are_legal(fen: str, expected_moves: List[str]) -> None:
    legal_moves = _legal_moves(fen)

    assert all(move in legal_moves for move in expected_moves)


def test_black_pawns_capture_white_pawns() -> None:
    assert "e5d4" in _legal_moves("k7/8/8/4p3/3P4/8/8/K7 b - - 0 1")


def test_legal_moves_match_python_chess_in_random_games() -> None:
    """Castling is excluded: the engine doesn't generate castling moves."""
    rng = random.Random(0)
    for _ in range(RANDOM_GAMES):
        board = chess.Board()
        for _ in range(PLIES_PER_GAME):
            if board.is_game_over():
                break
            expected = {move.uci() for move in board.legal_moves if not board.is_castling(move)}
            assert set(_legal_moves(board.fen())) == expected, board.fen()
            board.push(rng.choice(list(board.legal_moves)))


def test_start_position_has_no_illegal_extras() -> None:
    expected_legal_moves = 20

    assert len(set(_legal_moves(START_FEN))) == expected_legal_moves


def test_back_rank_mate_is_game_over() -> None:
    checkmate_code = 1
    engine = chess_moves.ChessEngine()
    engine.set_fen("8/8/8/8/8/8/4r3/K4r2 w - - 0 1")

    assert engine.is_game_over() == (True, checkmate_code)


def test_all_promotion_moves_are_generated() -> None:
    legal_moves = _legal_moves("8/4P3/8/8/8/8/8/k6K w - - 0 1")

    assert all(f"e7e8{piece}" in legal_moves for piece in PROMOTION_PIECES)


def test_promotion_push_places_promoted_piece() -> None:
    engine = chess_moves.ChessEngine()

    new_fen = engine.push("8/P7/8/8/8/8/8/k6K w - - 0 1", "a7a8q")

    assert new_fen.split()[0].startswith("Q7/8")


@pytest.mark.parametrize(
    "piece_idx, piece", [pytest.param(idx, piece, id=piece) for idx, piece in enumerate(PROMOTION_PIECES)]
)
def test_promotion_maps_to_its_own_channel(piece_idx: int, piece: str) -> None:
    engine = chess_moves.ChessEngine()

    channel, _, _ = engine.move_to_target_indices(f"a7a8{piece}")

    assert channel == FIRST_PROMOTION_CHANNEL + piece_idx


@pytest.mark.parametrize(
    "fen",
    [
        pytest.param(START_FEN, id="start_position"),
        pytest.param(ITALIAN_FEN, id="italian_game"),
    ],
)
def test_move_index_round_trip(fen: str) -> None:
    engine = chess_moves.ChessEngine()

    for move in _legal_moves(fen):
        assert engine.indices_to_move(*engine.move_to_target_indices(move)) == move
