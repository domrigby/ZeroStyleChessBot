import os
from typing import Tuple

import chess
import pygame
import pytest

from zero_style_chess_engine.ui.search_panel import SearchPanel, _figurine_san, _numbered_san_line
from tests.test_search import MATE_IN_ONE_FEN, NUM_ROLLOUTS, START_FEN, _all_nodes, _drain, _make_tree
from zero_style_chess_engine.mcts import MCTS
from zero_style_chess_engine.search_insights import SearchSummary, summarise_search

TOP_K = 5


def _searched_summary(fen: str) -> Tuple[MCTS, SearchSummary]:
    tree, _ = _make_tree(fen)
    tree.search(number_of_expansions=NUM_ROLLOUTS)
    _drain(tree)
    return tree, summarise_search(tree.root, rollouts_this_search=NUM_ROLLOUTS, elapsed_seconds=1.0, top_k=TOP_K)


def test_candidates_are_the_most_visited_root_moves_in_order() -> None:
    tree, summary = _searched_summary(START_FEN)

    visits = [candidate.visits for candidate in summary.candidates]
    assert len(summary.candidates) == TOP_K
    assert visits == sorted(visits, reverse=True)
    assert visits[0] == tree.root.Ns.max()
    assert summary.total_visits == NUM_ROLLOUTS
    assert abs(sum(c.visit_share for c in summary.candidates) - sum(visits) / NUM_ROLLOUTS) < 1e-9


def test_principal_variation_is_a_legal_line_starting_with_the_top_candidate() -> None:
    _, summary = _searched_summary(START_FEN)

    assert summary.principal_variation[0] == summary.candidates[0].move
    board = chess.Board(summary.fen)
    for move in summary.principal_variation:
        assert chess.Move.from_uci(move) in board.legal_moves
        board.push_uci(move)


def test_tree_statistics_match_the_tree() -> None:
    tree, summary = _searched_summary(START_FEN)

    assert summary.tree_size == sum(1 for _ in _all_nodes(tree.root))
    assert summary.max_depth >= len(summary.principal_variation)


def test_values_report_the_network_and_the_search_separately() -> None:
    # The instant evaluator values every position at 0; the search finds the mate and values it close to a win
    _, summary = _searched_summary(MATE_IN_ONE_FEN)

    assert summary.net_value == 0.0
    assert summary.search_value is not None and summary.search_value > 0.9


def test_search_value_is_the_q_of_the_most_visited_move_with_a_value() -> None:
    tree, _ = _searched_summary(START_FEN)
    root = tree.root
    well_tested, long_shot, in_flight = sorted(root.child_nodes)[:3]
    root.Ns[:], root.backed_up[:], root.Qs[:] = 0, 0, 0.0
    root.Ns[well_tested], root.backed_up[well_tested], root.Qs[well_tested] = 150, 150, -0.2
    root.Ns[long_shot], root.backed_up[long_shot], root.Qs[long_shot] = 2, 2, 0.9  # Lucky, not the search's view
    root.Ns[in_flight] = 300  # Every visit still being evaluated: no value yet, however many visits

    summary = summarise_search(root, rollouts_this_search=NUM_ROLLOUTS, elapsed_seconds=1.0)

    assert summary.search_value == -0.2


def test_unsearched_root_has_no_candidates_or_values() -> None:
    tree, _ = _make_tree(START_FEN)

    summary = summarise_search(tree.root, rollouts_this_search=0, elapsed_seconds=0.0)

    assert summary.candidates == []
    assert summary.principal_variation == []
    assert summary.net_value is None and summary.search_value is None
    assert summary.tree_size == 1 and summary.rollouts_per_second == 0.0


def test_numbered_san_line_from_either_side() -> None:
    board = chess.Board()
    assert _numbered_san_line(board, ["e2e4", "e7e5", "g1f3"]) == ["1.", "♟e4", "♟e5", "2.", "♞f3"]

    board.push_uci("e2e4")
    assert _numbered_san_line(board, ["e7e5", "g1f3"]) == ["1...", "♟e5", "2.", "♞f3"]


@pytest.mark.parametrize(
    "fen, uci, expected",
    [
        pytest.param("4k3/8/8/3p4/4P3/8/8/4K3 w - - 0 1", "e4d5", "♟exd5", id="pawn-capture"),
        pytest.param("7k/4P3/8/8/8/8/8/4K3 w - - 0 1", "e7e8q", "♟e8=Q+", id="promotion"),
        pytest.param("4k3/8/8/8/8/8/8/R3K2R w KQ - 0 1", "e1g1", "O-O", id="castling"),
        pytest.param("4k3/8/8/8/8/8/8/R3K2R w KQ - 0 1", "a1a8", "♜a8+", id="rook-check"),
    ],
)
def test_figurine_san_names_the_moving_piece(fen: str, uci: str, expected: str) -> None:
    assert _figurine_san(chess.Board(fen), chess.Move.from_uci(uci)) == expected


def test_panel_draws_a_summary_and_the_empty_state() -> None:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    pygame.font.init()
    _, summary = _searched_summary(START_FEN)
    screen = pygame.Surface((300, 400))
    panel = SearchPanel(pygame.Rect(0, 0, 300, 400))

    panel.draw(screen, None)
    panel.draw(screen, summary)
