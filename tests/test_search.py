import multiprocessing.queues
import queue
from pathlib import Path
from typing import Any, Iterator, Tuple, cast

import numpy as np

from zero_style_chess_engine.messages import EvalRequest, EvalResult
from zero_style_chess_engine.parallel_game_tree import GameTree, Node

START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
# Back-rank mate: Ra8# is white's only mating move
MATE_IN_ONE_FEN = "6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1"
MATING_MOVE = "a1a8"
NUM_ROLLOUTS = 200


class InstantEvaluator:
    """Stands in for the evaluator's request queue: answers every request at once with uniform priors and value 0.

    The answer lands on the tree's results queue, so the tree picks it up on its next poll, the same way it would a
    batch from the real evaluator. No thread, so the search is deterministic apart from PUCT tie-breaking.
    """

    def __init__(self, results_queue: "queue.Queue[Any]") -> None:
        self.results_queue = results_queue
        self.requests_answered = 0

    def put(self, request: EvalRequest) -> None:
        num_moves = len(request.legal_moves)
        uniform_priors = np.full(num_moves, 1.0 / num_moves) if num_moves else np.zeros(0)
        self.results_queue.put([EvalResult(request.node_id, 0.0, uniform_priors)])
        self.requests_answered += 1


def _make_tree(tmp_path: Path, start_state: str) -> Tuple[GameTree, InstantEvaluator]:
    # Thread queues stand in for the multiprocessing ones: same put/get interface, no feeder thread delay
    results_queue: "queue.Queue[Any]" = queue.Queue()
    evaluator = InstantEvaluator(results_queue)
    tree = GameTree(
        save_dir=str(tmp_path),
        eval_request_queue=cast(multiprocessing.queues.Queue, evaluator),
        eval_result_queue=cast(multiprocessing.queues.Queue, results_queue),
        start_state=start_state,
    )
    return tree, evaluator


def _all_nodes(root: Node) -> Iterator[Node]:
    stack = [root]
    while stack:
        node = stack.pop()
        yield node
        stack.extend(node.child_nodes.values())


def _drain(tree: GameTree) -> None:
    """The last rollout's evaluation is still on the results queue when search returns; apply it."""
    tree.apply_neural_net_results()


def test_root_visits_equal_completed_rollouts_and_every_q_is_bounded(tmp_path: Path) -> None:
    tree, _ = _make_tree(tmp_path, START_FEN)

    tree.parallel_search(number_of_expansions=NUM_ROLLOUTS)
    _drain(tree)

    assert tree.root.Ns.sum() == NUM_ROLLOUTS
    for node in _all_nodes(tree.root):
        assert ((node.Qs >= -1.0) & (node.Qs <= 1.0)).all()


def test_greedy_selection_finds_mate_in_one(tmp_path: Path) -> None:
    tree, _ = _make_tree(tmp_path, MATE_IN_ONE_FEN)

    tree.parallel_search(number_of_expansions=NUM_ROLLOUTS)
    _drain(tree)

    _, move, _ = tree.root.greedy_select_new_root_node()
    assert move == MATING_MOVE


def test_no_evaluations_pending_and_no_virtual_loss_left_after_drain(tmp_path: Path) -> None:
    tree, evaluator = _make_tree(tmp_path, START_FEN)

    tree.parallel_search(number_of_expansions=NUM_ROLLOUTS)
    _drain(tree)

    assert evaluator.requests_answered > 0
    assert tree.number_sent == tree.number_returned == evaluator.requests_answered
    assert not any(node.awaiting_processing for node in _all_nodes(tree.root))
    for node in _all_nodes(tree.root):
        assert (node.virtual_losses == 0).all()


def test_search_returns_at_once_when_stop_event_is_set(tmp_path: Path) -> None:
    tree, evaluator = _make_tree(tmp_path, START_FEN)
    tree.stop_event.set()

    tree.parallel_search(number_of_expansions=NUM_ROLLOUTS)

    assert tree.root.Ns.sum() == 0
    assert evaluator.requests_answered == 0
