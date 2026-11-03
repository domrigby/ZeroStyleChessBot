import multiprocessing
import multiprocessing.queues
import queue
from pathlib import Path
from typing import Any, List, cast

import chess_moves
import pytest

from zero_style_chess_engine.messages import PositionRecord
from zero_style_chess_engine.mcts import Node
from zero_style_chess_engine.self_play import SelfPlayWorker


@pytest.fixture
def worker(tmp_path: Path) -> SelfPlayWorker:
    # A thread queue stands in for the multiprocessing queue: same put/get interface, no feeder thread delay
    game_record_queue: "queue.Queue[Any]" = queue.Queue()
    return SelfPlayWorker(
        save_dir=str(tmp_path),
        agent_id=0,
        stop_event=multiprocessing.Event(),
        game_record_queue=cast(multiprocessing.queues.Queue, game_record_queue),
    )


def _optimal_line(root: Node, depth: int) -> Node:
    """Expand move 0 repeatedly. With all-zero Qs, argmax picks move 0, so every node is its parent's best move."""
    engine = chess_moves.ChessEngine()
    node = root
    for _ in range(depth):
        node, _, _ = node.expand(0, engine, state=node.state)
    return node


@pytest.mark.parametrize(
    "positions_on_path, expected_saved",
    [
        pytest.param(4, 4, id="even_keeps_all"),
        pytest.param(5, 4, id="odd_drops_one"),
        pytest.param(1, 0, id="single_position_drops_to_nothing"),
    ],
)
def test_optimal_path_record_keeps_an_even_number_of_positions(
    worker: SelfPlayWorker, positions_on_path: int, expected_saved: int
) -> None:
    # Saved positions are the nodes strictly between the root and the end node
    end_node = _optimal_line(worker.mcts.root, depth=positions_on_path + 1)

    worker._record_from(end_node, stop_when_suboptimal=True)
    assert worker.game_record_queue is not None

    if expected_saved == 0:
        assert worker.game_record_queue.empty()
    else:
        game_states = worker.game_record_queue.get_nowait().positions
        assert len(game_states) == expected_saved


FOOLS_MATE_MOVES = ["f2f3", "e7e5", "g2g4", "d8h4"]
STALEMATE_START = "7k/8/6K1/8/8/8/8/5Q2 w - - 0 1"
STALEMATE_MOVES = ["f1f7"]
WIN_VALUE = 1
LOSS_VALUE = -1
DRAW_VALUE = 0


def _play_scripted_game(worker: SelfPlayWorker, moves: List[str], monkeypatch: pytest.MonkeyPatch) -> None:
    """Run SelfPlayWorker.train with the search replaced by one that puts every visit on the next scripted move.

    The first search of the following game sets the stop event, so train() returns after one complete game.
    """
    engine = chess_moves.ChessEngine()
    remaining_moves = iter(moves)

    def scripted_search(number_of_expansions: int) -> List[Node]:
        current_node = worker.mcts.root
        move = next(remaining_moves, None)
        if move is None:
            worker.stop_event.set()
            move = current_node.moves[0]
        move_idx = current_node.moves.index(move)
        current_node.expand(move_idx, engine, state=current_node.state)
        current_node.Ns[move_idx] = 1
        return []

    monkeypatch.setattr(worker.mcts, "search", scripted_search)
    worker.train()


def _saved_games(worker: SelfPlayWorker) -> List[List[PositionRecord]]:
    assert worker.game_record_queue is not None
    games = []
    while not worker.game_record_queue.empty():
        game_states = worker.game_record_queue.get_nowait().positions
        games.append(game_states)
    return games


def test_played_checkmate_game_is_saved_with_outcome_values(
    worker: SelfPlayWorker, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression (D3): a game that ended in mate on the board was never sent to the trainer."""
    _play_scripted_game(worker, FOOLS_MATE_MOVES, monkeypatch)

    games = _saved_games(worker)
    assert len(games) == 1
    # Every position before the mate, starting position included; black delivered mate
    saved_positions = [(state.fen.split()[1], state.value) for state in games[0]]
    assert sorted(saved_positions) == sorted([("w", LOSS_VALUE), ("b", WIN_VALUE), ("w", LOSS_VALUE), ("b", WIN_VALUE)])
    assert worker.start_state in [state.fen for state in games[0]]


def test_played_stalemate_game_is_saved_as_a_draw(worker: SelfPlayWorker, monkeypatch: pytest.MonkeyPatch) -> None:
    worker.start_state = STALEMATE_START
    _play_scripted_game(worker, STALEMATE_MOVES, monkeypatch)

    games = _saved_games(worker)
    assert len(games) == 1
    assert [(state.fen, state.value) for state in games[0]] == [(STALEMATE_START, DRAW_VALUE)]
