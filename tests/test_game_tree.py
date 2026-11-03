import queue
from pathlib import Path

import chess_moves
import pytest

from zero_style_chess_engine.parallel_game_tree import GameTree, Node


@pytest.fixture
def game_tree(tmp_path: Path) -> GameTree:
    return GameTree(save_dir=str(tmp_path), experience_queue=queue.Queue())


def _optimal_line(root: Node, depth: int) -> Node:
    """Expand move 0 repeatedly. With all-zero Qs, argmax picks move 0, so every node is its parent's best move."""
    engine = chess_moves.ChessEngine()
    node = root
    for _ in range(depth):
        node, _, _ = node.expand(0, engine, state=node.state)
    return node


@pytest.mark.parametrize("positions_on_path, expected_saved", [
    pytest.param(4, 4, id="even_keeps_all"),
    pytest.param(5, 4, id="odd_drops_one"),
    pytest.param(1, 0, id="single_position_drops_to_nothing"),
])
def test_save_path_whilst_optimal_keeps_an_even_number_of_positions(game_tree: GameTree, positions_on_path: int,
                                                                    expected_saved: int) -> None:
    # Saved positions are the nodes strictly between the root and the end node
    end_node = _optimal_line(game_tree.root, depth=positions_on_path + 1)

    game_tree.save_path_whilst_optimal(end_node)

    if expected_saved == 0:
        assert game_tree.experience_queue.empty()
    else:
        game_states, _ = game_tree.experience_queue.get_nowait()
        assert len(game_states) == expected_saved
