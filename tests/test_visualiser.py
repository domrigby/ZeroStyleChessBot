import os
from typing import Iterator, Optional

import chess
import pygame
import pytest

from zero_style_chess_engine.search_insights import SearchSummary
from zero_style_chess_engine.ui.visualiser import GameHistory, Visualiser, expected_score

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


def _summary(board: chess.Board, search_value: Optional[float]) -> SearchSummary:
    return SearchSummary(
        fen=board.fen(),
        total_visits=10,
        rollouts_this_search=10,
        elapsed_seconds=1.0,
        candidates=[],
        principal_variation=[],
        net_value=None,
        search_value=search_value,
        tree_size=10,
        max_depth=2,
    )


def _after(*moves: str) -> chess.Board:
    board = chess.Board()
    for move in moves:
        board.push_uci(move)
    return board


def _key(key: int) -> pygame.event.Event:
    return pygame.event.Event(pygame.KEYDOWN, key=key)


def _click(pos: tuple[int, int]) -> pygame.event.Event:
    return pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=pos, button=1)


@pytest.fixture
def visualiser() -> Iterator[Visualiser]:
    """Analysis mode, after 1. e4 e5 2. Nf3: four positions, plies 0 to 3"""
    vis = Visualiser("test", "analysis", show_search=True, left_width=140)
    for moves in (["e2e4"], ["e2e4", "e7e5"], ["e2e4", "e7e5", "g1f3"]):
        vis.history.push(_after(*moves))
    yield vis
    pygame.quit()


@pytest.mark.parametrize(
    "board, value, expected",
    [(chess.Board(), 0.6, 0.8), (_after("e2e4"), 0.6, 0.2), (chess.Board(), 0.0, 0.5), (chess.Board(), None, None)],
)
def test_expected_score_is_from_whites_point_of_view(
    board: chess.Board, value: Optional[float], expected: Optional[float]
) -> None:
    assert expected_score(_summary(board, value)) == pytest.approx(expected)


def test_history_keeps_the_latest_search_of_the_latest_position_only() -> None:
    history = GameHistory(chess.Board())
    history.record_search(_summary(chess.Board(), 0.2))
    history.record_search(_summary(chess.Board(), 0.4))
    history.push(_after("e2e4"))
    history.record_search(_summary(chess.Board(), 0.9))  # Stale: a search of the position before the move

    assert history.expected_scores() == [pytest.approx(0.7), None]
    assert history.summary(1) is None


def test_history_copies_boards() -> None:
    board = chess.Board()
    history = GameHistory(board)
    board.push_uci("e2e4")

    assert history.board(0) == chess.Board()


def test_arrow_keys_step_through_the_game_and_end_returns_live(visualiser: Visualiser) -> None:
    assert visualiser.is_live and visualiser.viewed_ply == 3

    assert visualiser.handle_event(_key(pygame.K_LEFT))
    assert not visualiser.is_live and visualiser.viewed_ply == 2

    visualiser.handle_event(_key(pygame.K_HOME))
    visualiser.handle_event(_key(pygame.K_LEFT))  # Already at the start
    assert visualiser.viewed_ply == 0

    visualiser.handle_event(_key(pygame.K_END))
    assert visualiser.is_live

    visualiser.handle_event(_key(pygame.K_LEFT))
    visualiser.handle_event(_key(pygame.K_RIGHT))  # Back onto the latest position
    assert visualiser.is_live


def test_a_move_while_looking_back_leaves_the_view_where_it_is(visualiser: Visualiser) -> None:
    visualiser.handle_event(_key(pygame.K_LEFT))
    visualiser.history.push(_after("e2e4", "e7e5", "g1f3", "b8c6"))

    assert not visualiser.is_live and visualiser.viewed_ply == 2
    visualiser.handle_event(_key(pygame.K_END))
    assert visualiser.viewed_ply == 4


def test_clicking_the_graph_jumps_to_that_ply(visualiser: Visualiser) -> None:
    graph = visualiser.graph_rect
    assert graph is not None

    assert visualiser.handle_event(_click((graph.x + 1, graph.centery)))
    assert visualiser.viewed_ply == 0
    visualiser.handle_event(_click((graph.right - 1, graph.centery)))
    assert visualiser.is_live


def test_other_events_are_left_to_the_caller(visualiser: Visualiser) -> None:
    on_board = (int(visualiser.square_center(chess.E2)[0]), int(visualiser.square_center(chess.E2)[1]))

    assert not visualiser.handle_event(_click(on_board))
    assert not visualiser.handle_event(_key(pygame.K_a))
    assert visualiser.is_live


def test_play_mode_has_no_graph() -> None:
    play = Visualiser("test", "play", show_search=True)
    analysis = Visualiser("test", "analysis", show_search=True)
    try:
        assert play.graph_rect is None
        assert play.height < analysis.height
        assert not play.handle_event(_click((play.board_rect.x, play.height - 1)))
    finally:
        pygame.quit()


def test_square_at_inverts_square_center(visualiser: Visualiser) -> None:
    for square in (chess.A1, chess.H1, chess.E4, chess.H8):
        x, y = visualiser.square_center(square)
        assert visualiser.square_at((int(x), int(y))) == square
    assert visualiser.square_at((0, 0)) is None  # Left of the board, where the caller's buttons go


def test_draws_the_live_and_a_viewed_position(visualiser: Visualiser) -> None:
    visualiser.history.record_search(_summary(visualiser.history.board(3), 0.1))
    live = visualiser.history.board(3)

    visualiser.draw(live, None, [(chess.G1, chess.F3, (255, 0, 0))], show_candidates=True)
    visualiser.handle_event(_key(pygame.K_HOME))
    visualiser.draw(live, None)
