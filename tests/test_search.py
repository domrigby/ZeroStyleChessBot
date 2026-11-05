import multiprocessing
import multiprocessing.queues
import queue
from typing import Any, Iterator, List, Tuple, cast

import numpy as np
import pytest

from zero_style_chess_engine.config import Backup, SearchConfig
from zero_style_chess_engine.messages import EvalRequest, EvalResult
from zero_style_chess_engine.mcts import MCTS, Node
import chess_moves

START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
# Back-rank mate: Ra8# is white's only mating move
MATE_IN_ONE_FEN = "6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1"
MATING_MOVE = "a1a8"
NUM_ROLLOUTS = 200


class InstantEvaluator:
    """Stands in for the evaluator's request queue: answers every request at once with uniform priors and value 0.

    The answer lands on the search's results queue, so the search picks it up on its next poll, the same way it would a
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


def _make_tree(start_state: str) -> Tuple[MCTS, InstantEvaluator]:
    # Thread queues stand in for the multiprocessing ones: same put/get interface, no feeder thread delay
    results_queue: "queue.Queue[Any]" = queue.Queue()
    evaluator = InstantEvaluator(results_queue)
    tree = MCTS(
        cast(multiprocessing.queues.Queue, evaluator),
        cast(multiprocessing.queues.Queue, results_queue),
        agent_id=0,
        stop_event=multiprocessing.Event(),
    )
    tree.reset(start_state)
    return tree, evaluator


def _all_nodes(root: Node) -> Iterator[Node]:
    stack = [root]
    while stack:
        node = stack.pop()
        yield node
        stack.extend(node.child_nodes.values())


def _drain(tree: MCTS) -> None:
    """The last rollout's evaluation is still on the results queue when search returns; apply it."""
    tree.apply_eval_results()


def test_root_visits_equal_completed_rollouts_and_every_q_is_bounded() -> None:
    tree, _ = _make_tree(START_FEN)

    tree.search(number_of_expansions=NUM_ROLLOUTS)
    _drain(tree)

    assert tree.root.Ns.sum() == NUM_ROLLOUTS
    for node in _all_nodes(tree.root):
        assert ((node.Qs >= -1.0) & (node.Qs <= 1.0)).all()


def test_greedy_selection_finds_mate_in_one() -> None:
    tree, _ = _make_tree(MATE_IN_ONE_FEN)

    tree.search(number_of_expansions=NUM_ROLLOUTS)
    _drain(tree)

    _, move, _ = tree.select_greedy()
    assert move == MATING_MOVE


def test_no_evaluations_pending_and_no_virtual_loss_left_after_drain() -> None:
    tree, evaluator = _make_tree(START_FEN)

    tree.search(number_of_expansions=NUM_ROLLOUTS)
    _drain(tree)

    assert evaluator.requests_answered > 0
    assert tree.number_sent == tree.number_returned == evaluator.requests_answered
    assert not any(node.awaiting_processing for node in _all_nodes(tree.root))
    for node in _all_nodes(tree.root):
        assert (node.virtual_losses == 0).all()


def test_search_returns_at_once_when_stop_event_is_set() -> None:
    tree, evaluator = _make_tree(START_FEN)
    tree.stop_event.set()

    tree.search(number_of_expansions=NUM_ROLLOUTS)

    assert tree.root.Ns.sum() == 0
    assert evaluator.requests_answered == 0


class SkewedEvaluator(InstantEvaluator):
    """Answers with priors proportional to 1, 2, ..., N so they can't be mistaken for the uniform initial priors"""

    def put(self, request: EvalRequest) -> None:
        num_moves = len(request.legal_moves)
        self.results_queue.put([EvalResult(request.node_id, 0.0, np.arange(1.0, num_moves + 1))])
        self.requests_answered += 1


def test_root_priors_come_from_the_evaluator() -> None:
    results_queue: "queue.Queue[Any]" = queue.Queue()
    tree = MCTS(
        cast(multiprocessing.queues.Queue, SkewedEvaluator(results_queue)),
        cast(multiprocessing.queues.Queue, results_queue),
        agent_id=0,
        stop_event=multiprocessing.Event(),
    )
    tree.reset(START_FEN)

    tree.search(number_of_expansions=1)

    expected = np.arange(1.0, tree.root.number_legal_moves + 1)
    np.testing.assert_allclose(tree.root.Ps, expected / expected.sum())


# Qf7 stalemates the black king on h8
STALEMATE_IN_ONE_FEN = "7k/8/5QK1/8/8/8/8/8 w - - 0 1"
STALEMATING_MOVE = "f6f7"


class ConstantValueEvaluator(InstantEvaluator):
    """Answers every request with uniform priors and a fixed value, recording each request's legal moves"""

    def __init__(self, results_queue: "queue.Queue[Any]", value: float) -> None:
        super().__init__(results_queue)
        self.value = value
        self.requested_legal_moves: List[List[str]] = []

    def put(self, request: EvalRequest) -> None:
        self.requested_legal_moves.append(request.legal_moves)
        num_moves = len(request.legal_moves)
        uniform_priors = np.full(num_moves, 1.0 / num_moves) if num_moves else np.zeros(0)
        self.results_queue.put([EvalResult(request.node_id, self.value, uniform_priors)])
        self.requests_answered += 1


def _make_tree_with_value(start_state: str, value: float) -> Tuple[MCTS, ConstantValueEvaluator]:
    results_queue: "queue.Queue[Any]" = queue.Queue()
    evaluator = ConstantValueEvaluator(results_queue, value)
    tree = MCTS(
        cast(multiprocessing.queues.Queue, evaluator),
        cast(multiprocessing.queues.Queue, results_queue),
        agent_id=0,
        stop_event=multiprocessing.Event(),
    )
    tree.reset(start_state)
    return tree, evaluator


def test_game_ending_positions_are_not_sent_to_the_evaluator_and_q_stays_bounded() -> None:
    # A value of -1 is what a trained network says for the mated side: added to the mate reward it used to give Q = 2
    tree, evaluator = _make_tree_with_value(MATE_IN_ONE_FEN, value=-1.0)

    tree.search(number_of_expansions=NUM_ROLLOUTS)
    _drain(tree)

    assert all(legal_moves for legal_moves in evaluator.requested_legal_moves)
    for node in _all_nodes(tree.root):
        assert ((node.Qs >= -1.0) & (node.Qs <= 1.0)).all()


def test_stalemate_is_valued_as_a_draw_whatever_the_network_says() -> None:
    tree, _ = _make_tree_with_value(STALEMATE_IN_ONE_FEN, value=0.9)

    tree.search(number_of_expansions=NUM_ROLLOUTS)
    _drain(tree)

    stalemate_idx = tree.root.moves.index(STALEMATING_MOVE)
    assert tree.root.Ns[stalemate_idx] > 0
    assert tree.root.Qs[stalemate_idx] == 0.0


def test_revisiting_a_checkmate_backs_up_the_win_every_time() -> None:
    tree, _ = _make_tree_with_value(MATE_IN_ONE_FEN, value=0.0)

    tree.search(number_of_expansions=NUM_ROLLOUTS)
    _drain(tree)

    mate_idx = tree.root.moves.index(MATING_MOVE)
    # Q is the mean of the values backed up through the move: a win every visit is exactly 1, however many visits
    assert tree.root.Ns[mate_idx] >= 20
    assert tree.root.Qs[mate_idx] == 1.0


def test_q_averages_only_backed_up_values_while_other_visits_are_in_flight() -> None:
    tree, _ = _make_tree(START_FEN)
    root, move_idx = tree.root, 0
    # Three rollouts selected the move (selection counts them in Ns); their evaluations come back one at a time
    root.Ns[move_idx] += 3

    tree.backpropagation([(root, move_idx)], observed_reward=-0.6)
    assert root.Qs[move_idx] == pytest.approx(0.6)

    tree.backpropagation([(root, move_idx)], observed_reward=-0.2)
    assert root.Qs[move_idx] == pytest.approx(0.4)


def test_a_move_still_being_evaluated_does_not_count_towards_the_value_of_its_position() -> None:
    tree, _ = _make_tree(START_FEN)
    root, losing, in_flight = tree.root, 0, 1
    root.Ns[losing] += 1
    root.Ns[in_flight] += 1  # Its evaluation hasn't come back: Q is still the initial 0

    tree.backpropagation([(root, losing)], observed_reward=0.5)

    assert root.V == pytest.approx(-0.5)


def _root_and_child_with_a_strong_reply(backup: Backup) -> Tuple[MCTS, int, Node, int]:
    """A root move whose child position has one well-tested strong move, plus the child's move a rollout now takes"""
    tree, _ = _make_tree(START_FEN)
    tree.search_config = SearchConfig(backup=backup)
    tree.search(number_of_expansions=NUM_ROLLOUTS)
    _drain(tree)
    root = tree.root
    root_move = next(idx for idx, child in root.child_nodes.items() if child.number_legal_moves > 1)
    child = root.child_nodes[root_move]
    strong, taken = 0, 1
    for node, idx in [(root, root_move), (child, strong), (child, taken)]:
        node.Ns[idx] = node.backed_up[idx] = node.Qs[idx] = 0
    child.Ns[:] = child.backed_up[:] = 0
    child.Ns[strong] = child.backed_up[strong] = 3
    child.Qs[strong] = 0.9
    root.Ns[root_move] += 1
    child.Ns[taken] += 1
    return tree, root_move, child, taken


@pytest.mark.parametrize(
    "backup, expected_root_q",
    [
        pytest.param("mean", -0.3, id="mean_passes_up_the_rollouts_own_value"),
        pytest.param("max", -0.9, id="max_passes_up_the_best_moves_q"),
    ],
)
def test_backup_mode_decides_what_a_node_passes_to_its_parent(backup: Backup, expected_root_q: float) -> None:
    tree, root_move, child, taken = _root_and_child_with_a_strong_reply(backup)

    # The rollout through the child's weaker move comes back at 0.3 for the child's mover
    tree.backpropagation([(tree.root, root_move), (child, taken)], observed_reward=-0.3)

    assert child.Qs[taken] == pytest.approx(0.3)
    assert tree.root.Qs[root_move] == pytest.approx(tree.search_config.gamma * expected_root_q)


@pytest.mark.parametrize(
    "backup, expected_v", [pytest.param("mean", (3 * 0.9 + 0.3) / 4, id="mean"), pytest.param("max", 0.9, id="max")]
)
def test_node_value_is_the_visit_weighted_mean_or_the_best_q(backup: Backup, expected_v: float) -> None:
    tree, root_move, child, taken = _root_and_child_with_a_strong_reply(backup)

    tree.backpropagation([(tree.root, root_move), (child, taken)], observed_reward=-0.3)

    assert child.V == pytest.approx(expected_v)


@pytest.mark.parametrize("backup", ["mean", "max"])
@pytest.mark.parametrize("depth", [2, 4, 12])
def test_a_leaf_value_is_discounted_once_per_ply_on_its_way_to_the_root(backup: Backup, depth: int) -> None:
    tree, _ = _make_tree(START_FEN)
    tree.search_config = SearchConfig(backup=backup)
    # A line of fresh positions, each node's first move leading to the next
    board = chess_moves.ChessEngine()
    path: List[Tuple[Node, int]] = []
    node = tree.root
    for _ in range(depth):
        path.append((node, 0))
        node, _, _ = node.expand(0, board, node.state)

    # The leaf is worth +1 to its side to move, so the move into it is worth -1 to the mover
    tree.backpropagation(path, observed_reward=1.0)

    # Each ply up flips the sign and discounts once more: the root's move is worth (-1)^depth * gamma^(depth - 1)
    gamma = tree.search_config.gamma
    assert tree.root.Qs[0] == pytest.approx((-1) ** depth * gamma ** (depth - 1))


def test_search_defaults_to_mean_backup() -> None:
    assert SearchConfig().backup == "mean"


def _root_where_most_visited_and_highest_q_differ() -> Tuple[MCTS, int, int]:
    """A searched start position with the visits and Qs rewritten: one well-tested move and one lucky long shot"""
    tree, _ = _make_tree(START_FEN)
    tree.search(number_of_expansions=NUM_ROLLOUTS)
    _drain(tree)
    well_tested, long_shot = sorted(tree.root.child_nodes)[:2]
    tree.root.Ns[:] = np.where(tree.root.Ns > 0, 1, 0)
    tree.root.Qs[:] = 0.0
    tree.root.Ns[well_tested], tree.root.Qs[well_tested] = 150, 0.2
    tree.root.Ns[long_shot], tree.root.Qs[long_shot] = 2, 0.9
    return tree, well_tested, long_shot


def test_greedy_selection_by_visits_ignores_a_lucky_high_q() -> None:
    tree, well_tested, _ = _root_where_most_visited_and_highest_q_differ()

    node, move, move_idx = tree.select_greedy("visits")

    assert move_idx == well_tested
    assert (node, move) == (tree.root.child_nodes[well_tested], tree.root.moves[well_tested])


def test_greedy_selection_by_q_takes_the_highest_value() -> None:
    tree, _, long_shot = _root_where_most_visited_and_highest_q_differ()

    _, _, move_idx = tree.select_greedy("q")

    assert move_idx == long_shot


def test_greedy_selection_by_q_never_picks_an_unvisited_move() -> None:
    tree, _ = _make_tree(START_FEN)
    tree.search(number_of_expansions=3)
    _drain(tree)
    tree.root.Qs[tree.root.Ns == 0] = 1.0  # Unvisited moves' Qs mean nothing; make them tempting

    _, _, move_idx = tree.select_greedy("q")

    assert tree.root.Ns[move_idx] > 0
